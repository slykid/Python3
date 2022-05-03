import psutil
import pandas as pd

def cpu_usage():
    return psutil.cpu_percent()

def vitual_memory_usage():
    return psutil.virtual_memory().percent

def disk_usage():
    return psutil.disk_usage('C:').percent

def rss_usage():
    rss = psutil.Process().memory_info().rss / 2 ** 20
    return rss

print(cpu_usage())
print(vitual_memory_usage())
print(disk_usage())
print(rss_usage())