import argparse
from pathlib import Path
import re

import numpy as np


FIRST_MINUTE = 5 * 60
END_MINUTE = 23 * 60


def get_tap_in(flow, time_granularity, station_num):
    if time_granularity < 1 or station_num < 1:
        raise ValueError('Time interval and station count must be positive')
    intervals = (END_MINUTE - FIRST_MINUTE + time_granularity - 1) // time_granularity
    counts = np.zeros((intervals, station_num, station_num), dtype=np.int32)
    if len(flow) == 0:
        return counts.sum(axis=2).T, counts
    rows = np.asarray(flow, dtype=np.int64)
    entrance, exit_station, minute = rows[:, 1], rows[:, 2], rows[:, 3]
    valid = ((entrance >= 1) & (entrance <= station_num) &
             (exit_station >= 1) & (exit_station <= station_num) &
             (minute >= FIRST_MINUTE) & (minute < END_MINUTE))
    window = (minute[valid] - FIRST_MINUTE) // time_granularity
    np.add.at(counts, (window, entrance[valid] - 1, exit_station[valid] - 1), 1)
    return counts.sum(axis=2).T, counts


def chronological_key(path):
    match = re.search(r'_(\d{4})$', path.stem)
    if match is None:
        raise ValueError(f'Cannot identify MMDD from {path.name}')
    return match.group(1)


def main():
    local = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Aggregate daily fare-card logs by entry station and time slice.')
    parser.add_argument('--input-dir', type=Path, default=local/'data')
    parser.add_argument('--output-dir', type=Path, default=local/'generated')
    parser.add_argument('--intervals', nargs='+', type=int, default=[5, 10, 15, 30, 60])
    args = parser.parse_args()
    if min(args.intervals) < 1:
        parser.error('Intervals must be positive')
    inputs = sorted(args.input_dir.glob('*.csv'), key=chronological_key)
    if not inputs:
        parser.error(f'No daily CSV files found in {args.input_dir}')

    daily = [np.loadtxt(path, delimiter=',', quotechar='"', skiprows=1, dtype=np.int64, ndmin=2)
             for path in inputs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for interval in args.intervals:
        pieces = []
        for data in daily:
            inflow, _ = get_tap_in(data, interval, station_num=276)
            pieces.append(inflow)
        aggregate = np.concatenate(pieces, axis=1)
        destination = args.output_dir / f'O_flow_{interval}.csv'
        np.savetxt(destination, aggregate, delimiter=',', fmt='%i')
        print(f'{destination}: {aggregate.shape}; counted {aggregate.sum()} trips across {len(inputs)} days', flush=True)


if __name__ == '__main__':
    main()

