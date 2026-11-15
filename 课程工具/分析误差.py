from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


WINDOWS_A_DAY = 72
METHODS = ('persistence', 'both', 'inflow')
COMPARISONS = (('both', 'persistence'), ('both', 'inflow'))


def read_predictions(dual_path, inflow_path, flow_path, station_table):
    with np.load(dual_path) as both, np.load(inflow_path) as inflow:
        target_index = both['target_index']
        actual = both['actual'][..., 0]
        np.testing.assert_array_equal(target_index, inflow['target_index'])
        np.testing.assert_array_equal(actual, inflow['actual'][..., 0])
        np.testing.assert_array_equal(both['last_observation'], inflow['last_observation'])
        predictions = {'both': both['predicted'][..., 0], 'inflow': inflow['predicted'][..., 0]}
        recorded_reference = both['last_observation'][..., 0]

    raw = np.loadtxt(flow_path, delimiter=',')
    np.testing.assert_array_equal(actual, raw[:, target_index].T)
    # Compare on the exact discrete observations, without float32 scaling round-trip.
    predictions['persistence'] = raw[:, target_index - 1].T
    np.testing.assert_allclose(recorded_reference, predictions['persistence'], rtol=1e-7, atol=1e-5)
    np.testing.assert_array_equal(target_index, np.arange(1440, 1800))
    assert all(pred.shape == actual.shape and np.isfinite(pred).all() for pred in predictions.values())

    with station_table.open(encoding='gbk', newline='') as stream:
        rows = csv.reader(stream)
        names = next(rows)[:raw.shape[0]]
        source_rows = np.array([[float(cell) for cell in row[:raw.shape[0]]] for row in rows])
    np.testing.assert_array_equal(source_rows.T, raw)
    assert len(names) == len(set(names)) == actual.shape[1]
    return actual, predictions, target_index, names


def build_stats(actual, predictions):
    total = actual.sum()
    row = {'n': actual.size, 'actual_sum': total, 'actual_mean': actual.mean()}
    for mode in METHODS:
        signed_error = predictions[mode] - actual
        abs_error = np.abs(signed_error).sum()
        squared_error = np.square(signed_error).sum()
        row.update({
            f'{mode}_rmse': np.sqrt(squared_error / actual.size),
            f'{mode}_mae': abs_error / actual.size,
            f'{mode}_wmape_pct': 100. * abs_error / total if total != 0 else np.nan,
            f'{mode}_bias': signed_error.mean(),
            f'{mode}_sae': abs_error,
            f'{mode}_sse': squared_error,
        })
    for model, reference in COMPARISONS:
        for metric in ('rmse', 'mae', 'wmape_pct', 'sae', 'sse'):
            row[f'{model}_minus_{reference}_{metric}'] = row[f'{model}_{metric}'] - row[f'{reference}_{metric}']
    return row


def stats_for(actual, predictions, selector):
    return build_stats(actual[selector], {key: value[selector] for key, value in predictions.items()})


def write_table(output, name, rows):
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f'{name}.csv', index=False, encoding='utf-8-sig', float_format='%.9f')
    return frame


def clock_label(slot):
    minute = 5 * 60 + 15 * slot
    return f'{minute // 60:02d}:{minute % 60:02d}'


def build_change_table(output, actual, predictions, slots):
    change = actual - predictions['persistence']
    magnitude = np.abs(change)
    rows = []
    for group, label, selector in [
        ('change_magnitude', '0–24', magnitude < 25),
        ('change_magnitude', '25–99', (magnitude >= 25) & (magnitude < 100)),
        ('change_magnitude', '100–249', (magnitude >= 100) & (magnitude < 250)),
        ('change_magnitude', '>=250', magnitude >= 250),
        ('change_direction', 'rise>=250', change >= 250),
        ('change_direction', 'drop>=250', change <= -250),
        ('observed_flow', '0', actual == 0),
        ('observed_flow', '1–49', (actual > 0) & (actual < 50)),
        ('observed_flow', '50–249', (actual >= 50) & (actual < 250)),
        ('observed_flow', '250–999', (actual >= 250) & (actual < 1000)),
        ('observed_flow', '>=1000', actual >= 1000),
        ('day_boundary', '05:00', slots == 0),
        ('day_boundary', 'exclude_05:00', slots != 0),
    ]:
        rows.append({'group': group, 'bin': label, **stats_for(actual, predictions, selector)})
    return write_table(output, '分组误差分解', rows)


def plot_comparisons(output, times, stations):
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in ('Microsoft YaHei', 'Noto Sans CJK SC', 'PingFang SC', 'SimHei'):
        if family in installed:
            plt.rcParams.update({'font.family': family})
            break
    plt.rcParams.update({'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.6), layout='constrained',
                             gridspec_kw={'height_ratios': (0.9, 1.1)})
    for column, (model, reference) in enumerate(COMPARISONS):
        prefix = f'{model}_minus_{reference}_'
        label = '双输入减上一观测基线' if reference == 'persistence' else '双输入减仅进站'
        period = axes[0, column]
        period.axhline(0, color='#677483', linewidth=1, zorder=1)
        period.plot(times['slot'], times[prefix + 'rmse'], color='#006699', label='RMSE',
                    linewidth=2, marker='o', markersize=3.2, markevery=4)
        period.plot(times['slot'], times[prefix + 'mae'], color='#a84f14', label='MAE',
                    linewidth=1.9, linestyle='--', marker='s', markersize=2.8, markevery=4)
        period.set(title=label, ylabel='误差差值（人）', xlabel='时段起点',
                   xlim=(-1, 72))
        period.set_xticks(np.arange(0, 72, 8), [clock_label(i) for i in range(0, 72, 8)])
        period.legend(loc='best', frameon=False, ncols=2)

        station = axes[1, column]
        station.axhline(0, color='#677483', linewidth=1, zorder=1)
        station.axvline(0, color='#677483', linewidth=1, zorder=1)
        station.scatter(stations[prefix + 'rmse'], stations[prefix + 'mae'],
                        s=17, alpha=0.60, color='#006699', edgecolor='none', zorder=2)
        for name in ('国贸', '西单') if column == 0 else ('宋家庄', '阜成门'):
            row = stations.loc[stations['station'] == name].iloc[0]
            x, y = row[prefix + 'rmse'], row[prefix + 'mae']
            station.scatter(x, y, marker='D', s=26, color='#a84f14', zorder=3)
            station.annotate(name, (x, y), xytext=(6 if x < 0 else -6, -9 if y > 0 else 7),
                             textcoords='offset points', ha='left' if x < 0 else 'right', fontsize=9)
        station.set(xlabel='站点 RMSE 差值（人）', ylabel='站点 MAE 差值（人）')
    axes[0, 1].sharey(axes[0, 0])
    axes[1, 1].sharex(axes[1, 0])
    axes[1, 1].sharey(axes[1, 0])
    for axis in axes.ravel():
        axis.grid(True, color='#e9edf1', linewidth=0.65, zorder=0)
        axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle('地铁进站客流预测误差｜5 个测试日段，276 个站点', fontsize=16)
    fig.savefig(output / '站点与时段误差.png', dpi=170, facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Compare the course metro inflow prediction runs.')
    parser.add_argument('--dual', type=Path, required=True, help='Dual-input predictions.npz')
    parser.add_argument('--inflow', type=Path, required=True, help='Inflow-only predictions.npz')
    parser.add_argument('--flow', type=Path, required=True, help='Metro in_15min.csv')
    parser.add_argument('--station-table', type=Path, required=True, help='Named 15min_in.csv from the CNN chapter')
    parser.add_argument('--output-dir', type=Path, default=Path('课程输出/地铁误差分析'))
    args = parser.parse_args()
    actual, predictions, target_index, names = read_predictions(args.dual, args.inflow, args.flow, args.station_table)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    slots = target_index % WINDOWS_A_DAY
    day = (target_index - target_index[0]) // WINDOWS_A_DAY + 1
    overall = write_table(output, '总体指标', [build_stats(actual, predictions)])
    stations = write_table(output, '逐站误差', [
        {'station_id': i + 1, 'station': name,
         **stats_for(actual, predictions, (slice(None), i))}
        for i, name in enumerate(names)
    ])
    times = write_table(output, '15分钟分时误差', [
        {'slot': slot, 'time': clock_label(slot), **stats_for(actual, predictions, slots == slot)}
        for slot in range(WINDOWS_A_DAY)
    ])
    hours = write_table(output, '小时分时误差', [
        {'hour': clock_label(slot), **stats_for(actual, predictions, slots // 4 == slot // 4)}
        for slot in range(0, WINDOWS_A_DAY, 4)
    ])
    station_hours = write_table(output, '逐站小时误差', [
        {'station_id': i + 1, 'station': name, 'hour': clock_label(4 * hour),
         **stats_for(actual, predictions, (slots // 4 == hour, i))}
        for i, name in enumerate(names)
        for hour in range(WINDOWS_A_DAY // 4)
    ])
    daily = write_table(output, '逐日段误差', [
        {'test_day': label, **stats_for(actual, predictions, day == label)}
        for label in np.unique(day)
    ])
    changes = build_change_table(output, actual, predictions, slots)

    expected_counts = [(stations, 276), (times, 72), (hours, 18), (station_hours, 276 * 18), (daily, 5)]
    for frame, count in expected_counts + [(changes.loc[changes.group == 'change_magnitude'], 4)]:
        assert len(frame) == count
        assert frame['n'].sum() == actual.size
        for method in METHODS:
            for loss in ('sae', 'sse'):
                column = f'{method}_{loss}'
                np.testing.assert_allclose(frame[column].sum(), overall[column].iloc[0])
    plot_comparisons(output, times, stations)
    print(f'{actual.size} matched predictions; verified partition sums and saved tables and figure.')


if __name__ == '__main__':
    main()

