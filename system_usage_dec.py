import time
import psutil
import GPUtil
import pandas as pd
from functools import wraps
import os


def monitor_performance(csv_file="performance_stats.csv"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            mem_usage = []
            gpu_usage = []
            start_time = time.time()

            process = psutil.Process(os.getpid())

            def collect_stats():
                mem_info = process.memory_info()
                mem_usage.append((mem_info.rss / (1024 ** 2), mem_info.vms / (1024 ** 2)))  # Convert to MB

                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_info = gpus[0]
                    gpu_usage.append((gpu_info.memoryUsed, gpu_info.memoryUtil * gpu_info.memoryTotal / (1024)))  # Convert to MB
                else:
                    gpu_usage.append((0, 0))

            from threading import Thread, Event

            stop_event = Event()

            def monitor():
                while not stop_event.is_set():
                    collect_stats()
                    time.sleep(30)

            monitor_thread = Thread(target=monitor)
            monitor_thread.start()

            result = func(*args, **kwargs)
            stop_event.set()
            monitor_thread.join()

            end_time = time.time()
            runtime = end_time - start_time

            peak_mem_usage = max(mem_usage, key=lambda x: x[0])[0]
            mean_mem_usage = sum(x[0] for x in mem_usage) / len(mem_usage)
            peak_gpu_usage = max(gpu_usage, key=lambda x: x[0])[0]
            mean_gpu_usage = sum(x[0] for x in gpu_usage) / len(gpu_usage)
            accuracy = result

            data = {
                "function_name": [func.__name__],
                "peak_memory_usage (MB)": [peak_mem_usage],
                "mean_memory_usage (MB)": [mean_mem_usage],
                "peak_gpu_usage (MB)": [peak_gpu_usage],
                "mean_gpu_usage (MB)": [mean_gpu_usage],
                "runtime (s)": [runtime],
                "accuracy": [accuracy]
            }

            df = pd.DataFrame(data)
            df.to_csv(csv_file, index=False)

            return result
        return wrapper
    return decorator


