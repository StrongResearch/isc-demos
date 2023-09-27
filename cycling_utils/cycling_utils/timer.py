import os, time
from datetime import datetime

class Timer:
    '''
    This Timer can be integrated within a training routine to provide point-to-point
    script timing and reporting.

    def main():
        timer = Timer()
        time.sleep(2)
        timer.report("sleeping for 2 seconds")
        time.sleep(3)
        timer.report("sleeping for 3 seconds")

    >>> main()
    Start                                       0.000 ms     0.000 s total
    Completed sleeping for 2 seconds        2,000.000 ms     2.000 s total
    Completed sleeping for 3 seconds        3,000.000 ms     5.000 s total
    '''
    def __init__(self, report=None, start_time=None, running=0):
        self.start_time = start_time if start_time is not None else time.time()
        self.running = running
        if str(os.environ["RANK"]) == "0":
            report = report if report else "Start"
            print("[{:<80}] {:>12} ms, {:>12} s total".format(report, f'{0.0:,.3f}', f'{0.0:,.2f}'))
    def report(self, annot):
        if str(os.environ["RANK"]) == "0":
            now = time.time()
            duration = now - self.start_time
            self.running += duration
            print("Completed {:<70}{:>12} ms, {:>12} s total".format(annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
            self.start_time = now

class TimestampedTimer:
    '''
    This TimestampedTimer can be integrated within a training routine to provide 
    point-to-point script timing and reporting.

    def main():
        timer = TimestampedTimer()
        time.sleep(2)
        timer.report("sleeping for 2 seconds")
        time.sleep(3)
        timer.report("sleeping for 3 seconds")

    >>> main()
    [TIME] Start                                       0.000 ms     0.000 s total
    [TIME] Completed sleeping for 2 seconds        2,000.000 ms     2.000 s total
    [TIME] Completed sleeping for 3 seconds        3,000.000 ms     5.000 s total
    '''
    def __init__(self, report=None, start_time=None, running=0):
        if str(os.environ.get("RANK","NONE")) in ["0", "NONE"]:
            self.start_time = start_time if start_time is not None else time.time()
            self.running = running
            report = report if report else "Start"
            print("[ {} ] Completed {:<70}{:>12} ms, {:>12} s total".format(time.strftime("%Y-%m-%d %H:%M:%S"), report, f'{0.0:,.3f}', f'{0.0:,.2f}'))
    def report(self, annot):
        if str(os.environ.get("RANK","NONE")) in ["0", "NONE"]:
            now = time.time()
            duration = now - self.start_time
            self.running += duration
            print("[ {} ] Completed {:<70}{:>12} ms, {:>12} s total".format(time.strftime("%Y-%m-%d %H:%M:%S"), annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
            self.start_time = now