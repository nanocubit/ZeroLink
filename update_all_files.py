import os
import re

def replace_in_file(file_path):
    """Заменяет все вхождения 'ZeroLink' на 'ZeroLink' и 'zerolink' на 'zerolink' в файле"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Заменяем все вхождения ZeroLink на ZeroLink
        updated_content = re.sub(r'ZeroLink', 'ZeroLink', content)
        # Заменяем все вхождения zerolink на zerolink
        updated_content = re.sub(r'\bpynexus_rex\b', 'zerolink', updated_content)
        
        # Сохраняем изменения, только если были изменения
        if content != updated_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"Updated: {file_path}")
        else:
            print(f"No changes: {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    # Обрабатываем все файлы в проекте
    for root, dirs, files in os.walk("."):
        # Исключаем директории __pycache__ и .git
        dirs[:] = [d for d in dirs if d not in ('__pycache__', '.git')]
        
        for file in files:
            if file.endswith(('.py', '.md', '.txt', '.yaml', '.yml')):
                file_path = os.path.join(root, file)
                replace_in_file(file_path)

if __name__ == "__main__":
    main()