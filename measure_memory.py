import argparse
import time

import psutil
import numpy as np

AVERAGING_FREQ = 2600
REPORT_FREQ = 10


def main():
    parser = argparse.ArgumentParser('Measure average memory usage of the process given')
    parser.add_argument('-p', '--pid', type=int, required=True, help="process id")
    args = parser.parse_args()

    averages = []
    times = []
    print(f"Reporting mean memory of pid {args.pid} per {REPORT_FREQ} s ...")

    while True:
        try:
            mem = psutil.Process(args.pid).memory_info().rss / 1024 ** 2
        except psutil.NoSuchProcess:
            print("the process finished")
            break
        times.append(mem)
        if len(times) == AVERAGING_FREQ:
            mean = float(np.mean(np.array(times)))
            averages.append(mean)
            times = []
        if len(times) % REPORT_FREQ == 0:
            mean = (sum(times) + AVERAGING_FREQ * sum(averages)) / (len(times) + AVERAGING_FREQ * len(averages))
            print(f"average = {mean} MB")
        time.sleep(1)


if __name__ == '__main__':
    main()
