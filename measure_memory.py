import argparse
import time

import psutil
import numpy as np

AVERAGING_FREQ = 2600
REPORT_FREQ = 10


def main():
    parser = argparse.ArgumentParser('Measure average memory usage of the process given')
    parser.add_argument('-c', '--command', required=True, help="part of process command")
    args = parser.parse_args()

    proc_iter = psutil.process_iter(attrs=["pid", "cmdline"])
    processes = [p for p in proc_iter if
                 args.command in " ".join(p.info["cmdline"]) and "measure_memory" not in " ".join(p.info["cmdline"])]

    for p in processes:
        print(p.info["cmdline"])

    averages = []
    times = []
    print(f"Reporting mean memory of {len(processes)} process(es) containing '{args.command}' per {REPORT_FREQ} s ...")

    while True:
        try:
            mem = sum(p.memory_info().rss / 1024 ** 2 for p in processes)
            # mem = psutil.Process(args.pid).memory_info().rss / 1024 ** 2
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
