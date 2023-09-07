import time
import os

class Timer:
    def __init__(self, start_time=None, running=0):
        self.start_time = start_time if start_time is not None else time.time()
        self.running = running
    def report(self, annot):
        if str(os.environ["RANK"]) == "0":
            now = time.time()
            duration = now - self.start_time
            self.running += duration
            print("Completed {:<70}{:>12} milliseconds, {:>12} seconds total".format(annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
            self.start_time = now