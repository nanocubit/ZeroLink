/*
ZeroLink - Unified Zero-Copy Runtime for CPU/GPU Computing
Copyright (C) 2025 Slava Maltsev <nanotec@live.ru>
SPDX-License-Identifier: MIT

zerolink/core/gpu/ext/ipc_ext.cpp

C++ Extension для ZeroLink v2.0.
Реализует обертки вокруг CUDA Driver API (VMM) для импорта памяти
и создания PyTorch тензоров без копирования (from_blob).
*/

#include <torch/extension.h>
#include <ATen/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime_api.h> // Для cudaMalloc/free если нужно, но VMM через driver API

#include <memory>
#include <vector>
#include <mutex>
#include <stdexcept>

// ============================================================================
// Макрос для проверки ошибок CUDA
// ============================================================================

#define CU_CHECK(expr) do {                          \
  CUresult r = (expr);                                 \
  if (r != CUDA_SUCCESS) {                             \
    const char* name = nullptr;                        \
    const char* msg  = nullptr;                        \
    cuGetErrorName(r, &name);                          \
    cuGetErrorString(r, &msg);                         \
    TORCH_CHECK(false, "CUDA driver error: ",           \
                (name ? name : "<?>"), " / ",           \
                (msg ? msg : "<?>"));                     \
  }                                                       \
} while(0)

// ============================================================================
// Вспомогательная структура для хранения импортированной памяти
// ============================================================================

struct ImportedRegion {
  int device_id = -1;
  size_t total_size = 0;
  CUdeviceptr base = 0;      // Резервированный VA в ЭТОМ процессе

  // Хендлы физической памяти (импортированные из FD)
  std::vector<CUmemGenericAllocationHandle> handles;

  // Информация о замапленных сегментах для корректного Unmap
  struct Seg { size_t dst; size_t len; };
  std::vector<Seg> mapped_segs;

  ImportedRegion() = default;

  ImportedRegion(
      int dev,
      const std::vector<int>& fds,
      const std::vector<size_t>& src_offsets,
      const std::vector<size_t>& dst_offsets,
      const std::vector<size_t>& lengths,
      size_t total)
    : device_id(dev), total_size(total) {

    // 1. Инициализация CUDA (однократно)
    static std::once_flag once;
    std::call_once(once, [](){ CU_CHECK(cuInit(0)); });

    at::cuda::CUDAGuard guard(device);

    // 2. Импорт хендлов физической памяти из FD
    handles.reserve(fds.size());
    for (size_t i = 0; i < fds.size(); ++i) {
      CUmemGenericAllocationHandle h{};
      // Важно: Используем POSIX FD
      CU_CHECK(cuMemImportFromShareableHandle(
        &h,
        (void*)(intptr_t)fds[i],
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
      ));
      handles.push_back(h);
    }

    // 3. Резервирование VA (Виртуального Адресного Пространства)
    // Используем alignment 2MB (стандарт для GPU)
    size_t alignment = 2 * 1024 * 1024;
    CU_CHECK(cuMemAddressReserve(&base, total, alignment, 0, 0));

    // 4. Маппинг сегментов
    // Логика: base (VA) + dst_offset = целевой адрес.
    // Offset в handle = src_offset.
    for (size_t i = 0; i < lengths.size(); ++i) {
      size_t src = src_offsets[i];
      size_t dst = dst_offsets[i];
      size_t len  = lengths[i];

      // Валидация выравнивания (на всякий случай)
      if (src % 4096 != 0 || dst % 4096 != 0 || len % 4096 != 0) {
        // Бросаем исключение, так как VMM требует выравнивания
        TORCH_CHECK(false, "Misalignment in segment mapping");
      }

      // cuMemMap(ptr, size, offset_in_ptr, handle, handle_offset)
      // Используем рекомендацию явного указателя:
      CU_CHECK(cuMemMap(base + dst, len, 0, handles[i], src));

      mapped_segs.push_back({dst, len});
    }

    // 5. Установка прав доступа (ReadWrite для устройства)
    CUmemAccessDesc ad{};
    ad.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    ad.location.id   = device_id;
    ad.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Открываем доступ ко всему зарезервированному диапазону сразу
    CU_CHECK(cuMemSetAccess(base, total, &ad, 1));
  }

  // Деструктор: критически важен для очистки при удалении тензора
  ~ImportedRegion() {
    // Предотвращаем исключения в деструкторе
    try {
      if (device_id >= 0) {
        at::cuda::CUDAGuard guard(device);

        // 1. Unmap всех сегментов
        for (auto &s : mapped_segs) {
          cuMemUnmap(base + s.dst, s.len);
        }

        // 2. Освобождение VA (сам диапазон)
        if (base) {
          cuMemAddressFree(base, total);
        }

        // 3. Освобождение импортированных хендлов (физическая память)
        // Важно: Мы не "удаляем" память, мы отпускаем хендл.
        // Реальное удаление памяти происходит на стороне Main-процесса.
        for (auto &h : handles) {
          cuMemRelease(h);
        }
      }
    } catch (...) {
      // Игнорируем ошибки в деструкторе
    }
  }
};

// ============================================================================
// PyBind11 Bindings
// ============================================================================

// 1. Получение Granularity (аллокационной единицы)
static int64_t get_granularity(int device) {
  // Инициализация CUDA
  static std::once_flag once;
  std::call_once(once, [](){ CU_CHECK(cuInit(0)); });

  at::cuda::CUDAGuard guard(device);

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t gran = 0;
  CU_CHECK(cuMemGetAllocationGranularity(
      &gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return (int64_t)gran;
}

// 2. Импорт VMM сегментов (основная функция IPC)
// Возвращает PyCapsule, содержащую shared_ptr на ImportedRegion
static py::capsule import_vmm_segments(
    std::vector<int> fds,
    std::vector<int64_t> src_offsets,
    std::vector<int64_t> dst_offsets,
    std::vector<int64_t> lengths,
    int64_t total_size,
    int device
) {
  // Конвертация в size_t
  std::vector<size_t> src(src_offsets.begin(), src_offsets.end());
  std::vector<size_t> dst(dst_offsets.begin(), dst_offsets.end());
  std::vector<size_t> len(lengths.begin(), lengths.end());

  // Создаем регион (управление памятью)
  auto region = std::make_shared<ImportedRegion>(
      device, fds, src, dst, len, (size_t)total_size
  );

  // Упаковываем shared_ptr в PyCapsule
  // Это позволяет объекту жить в Python, пока Capsule жива
  auto* sp = new std::shared_ptr<ImportedRegion>(std::move(region));

  return py::capsule(sp, "zerolink.ImportedRegion",
    [](PyObject* cap) {
      // Когда Capsule удаляется в Python (refcount=0)
      auto* p = (std::shared_ptr<ImportedRegion>*)PyCapsule_GetPointer(cap, "zerolink.ImportedRegion");
      delete p; // Уменьшает shared_ptr. Если счетчик == 0 -> вызовется ~ImportedRegion
    }
  );
}

// 3. Создание тензора из импортированной памяти
// Это дает zero-copy доступ к GPU данным
static torch::Tensor tensor_from_imported(
    const py::capsule& cap,
    std::vector<int64_t> sizes,
    torch::ScalarType dtype,
    int64_t offset_bytes,
    c10::optional<std::vector<int64_t>> strides
) {
  // Извлекаем регион
  auto* p = (std::shared_ptr<ImportedRegion>*)cap.get_pointer();
  TORCH_CHECK(p && *p, "Invalid ImportedRegion capsule");
  auto region = *p;

  TORCH_CHECK(offset_bytes >= 0, "offset_bytes must be >= 0");
  TORCH_CHECK((size_t)offset_bytes < region->total_size, "offset out of range");

  at::cuda::CUDAGuard guard(region->device_id);

  // Вычисляем реальный указатель на данные
  void* ptr = (void*)((uintptr_t)region->base + (uintptr_t)offset_bytes);

  // Опции тензора
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA, region->device_id);

  // Создаем Deleter: захватывает shared_ptr<ImportedRegion>
  // Пока Tensor жив -> region жив -> память замаплена.
  auto deleter = [region](void*) mutable { 
    region.reset(); 
  };

  if (strides.has_value()) {
    return torch::from_blob(ptr, sizes, *strides, deleter, options);
  }
  return torch::from_blob(ptr, sizes, deleter, options);
}

// 4. Вспомогательная функция для получения указателя (для дебага)
static int64_t ptr(const py::capsule& cap) {
  auto* p = (std::shared_ptr<ImportedRegion>*)cap.get_pointer();
  TORCH_CHECK(p && *p, "Invalid ImportedRegion capsule");
  return (int64_t)(*p)->base;
}

// ============================================================================
// Модуль (pybind11)
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyNexus Rex v2.0 CUDA VMM Extension";

  // Экспорт функций
  m.def("get_granularity", &get_granularity,
        "Get CUDA VMM allocation granularity for a device");

  m.def("import_vmm_segments", &import_vmm_segments,
        py::arg("fds"),
        py::arg("src_offsets"),
        py::arg("dst_offsets"),
        py::arg("lengths"),
        py::arg("total_size"),
        py::arg("device"),
        "Import physical memory chunks via FDs and map them into a contiguous VA range");

  m.def("tensor_from_imported", &tensor_from_imported,
        py::arg("region_capsule"),
        py::arg("sizes"),
        py::arg("dtype"),
        py::arg("offset_bytes") = 0,
        py::arg("strides") = py::none(),
        "Create a torch.Tensor from imported GPU memory without copying");

  m.def("ptr", &ptr, "Get base VA pointer (debugging)");
}