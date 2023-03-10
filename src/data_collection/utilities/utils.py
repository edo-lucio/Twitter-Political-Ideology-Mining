import csv
import time

class Timer:
    def __init__(self, seconds):
        self.seconds = seconds
        self.start_time = time.time()

    def timer_ended(self):
        now = time.time()

        if now - self.start_time >= self.seconds:
            return True
        return False


def write_file(file, *args):
    writer = csv.writer(file)
    writer.writerow(args)