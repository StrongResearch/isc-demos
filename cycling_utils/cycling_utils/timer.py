import time
import os

class Timer:
    def __init__(self, report=None, start_time=None, running=0):
        self.start_time = start_time if start_time is not None else time.time()
        self.running = running
        if str(os.environ["RANK"]) == "0":
            report = report if report else "Start"
            print("{:<80}{:>12} ms, {:>12} s total".format(report, f'{0.0:,.3f}', f'{0.0:,.2f}'))
    def report(self, annot):
        if str(os.environ["RANK"]) == "0":
            now = time.time()
            duration = now - self.start_time
            self.running += duration
            print("Completed {:<70}{:>12} ms, {:>12} s total".format(annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
            self.start_time = now

# import os
# from dateime import datetime
# class Timer:
#     def __init__(self, report=None, start_time=None, running=0):
#         self.start_time = start_time if start_time is not None else datetime.now()
#         self.running = running
#         if str(os.environ["RANK"]) == "0":
#             report = report if report else "Start"
#             print("{:<30} {:<70}{:>12} ms, {:>12} s total".format(self.start_time, report, f'{0.0:,.3f}', f'{0.0:,.2f}'))
#     def report(self, annot):
#         if str(os.environ["RANK"]) == "0":
#             now = datetime.now()
#             duration = (now - self.start_time).total_seconds()
#             self.running += duration
#             print("{:<30} Completed {:<70}{:>12} ms, {:>12} s total".format(now, annot, f'{1000*duration:,.3f}', f'{self.running:,.2f}'))
#             self.start_time = now