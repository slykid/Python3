import multiprocessing
import os

def do_this(what):
    whoami(what)

def whoami(what):
    print(f"Process {os.getpid()} says: {what}")

if __name__ == "__main__":
    whoami("I'm the main program")

    for i in range(4):
        p = multiprocessing.Process(target=do_this, args=(f"I'm function {i}",))
        p.start()