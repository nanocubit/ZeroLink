"""
zerolink/workers/cpu_worker.py

CPU Worker (Обертка над multiprocessing.Pool для простоты).
"""

import multiprocessing as mp

class CPUWorker:
    def __init__(self, num_workers: int = 4):
        self.pool = mp.Pool(processes=num_workers)

    def map(self, func, iterable):
        return self.pool.map(func, iterable)

    def apply(self, func, args):
        return self.pool.apply(func, args)

    def close(self):
        self.pool.close()
        self.pool.join()