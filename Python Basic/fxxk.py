import multiprocessing
import os
import time

def whoami(name):
    print(f"I'm {name}, in process {os.getpid()}")

def loopy(name):
    whoami(name)
    start = 1
    stop = 1000000

    for step in range(start, stop):
        print(f"\tNumber {step} of {stop}. Honk!")
        time.sleep(5)

if __name__ == "__main__":
    whoami("main")

    p = multiprocessing.Process(target=loopy, args=("loopy",))
    p.start()

    time.sleep(5)
    p.terminate()