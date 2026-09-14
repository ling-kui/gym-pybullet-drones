"""Combined-fault sweep for the leader-follower formation.

Runs factorial combinations of link faults (delay, packet loss, noise):
single faults, all pairs, the full triple at a moderate level, plus mild and
severe ladders. Delayed configurations are run with and without dead-reckoning
compensation. Outputs a CSV table and a grouped bar chart of the mean vector
formation error.

Example
-------

    $ python sweep_combo.py
"""
import os
import csv
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from formation_flight import run

SEED = 0

# (label, delay_steps, loss, noise, compensate)
CONFIGS = [
    ('baseline',            0,  0.0, 0.00, False),
    ('D8',                  8,  0.0, 0.00, False),
    ('D8+comp',             8,  0.0, 0.00, True),
    ('L30%',                0,  0.3, 0.00, False),
    ('N5cm',                0,  0.0, 0.05, False),
    ('D8+L30%',             8,  0.3, 0.00, False),
    ('D8+L30%+comp',        8,  0.3, 0.00, True),
    ('D8+N5cm',             8,  0.0, 0.05, False),
    ('D8+N5cm+comp',        8,  0.0, 0.05, True),
    ('L30%+N5cm',           0,  0.3, 0.05, False),
    ('D8+L30%+N5cm',        8,  0.3, 0.05, False),
    ('D8+L30%+N5cm+comp',   8,  0.3, 0.05, True),
    ('mild D2+L10%+N1cm',   2,  0.1, 0.01, False),
    ('severe D16+L50%+N10cm',        16, 0.5, 0.10, False),
    ('severe D16+L50%+N10cm+comp',   16, 0.5, 0.10, True),
]


def quiet(**kw):
    kw.setdefault('gui', False)
    kw.setdefault('plot', False)
    kw.setdefault('seed', SEED)
    import io, contextlib, warnings
    warnings.filterwarnings('ignore')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run(**kw)


def main(output_folder='results'):
    os.makedirs(output_folder, exist_ok=True)
    rows = []
    for label, d, p, n, comp in CONFIGS:
        m = quiet(mode='leader-follower', delay_steps=d, loss=p, noise=n, compensate=comp)
        m['label'] = label
        rows.append(m)
        print("[combo] %-26s -> vec rms %.4f m | mean %.4f m | max %.4f m"
              % (label, m['vec_err_rms'], m['vec_err_mean'], m['vec_err_max']))

    #### CSV ############################################################
    csv_path = os.path.join(output_folder, 'fault_combo.csv')
    keys = ['label', 'mode', 'delay_steps', 'loss', 'noise', 'compensate',
            'vec_err_mean', 'vec_err_max', 'vec_err_rms', 'dist_err_mean', 'dist_err_max', 'leader_rmse']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print('[combo] CSV written to', csv_path)

    #### Figure: grouped bars, mean error in mm ##########################
    base = rows[0]['vec_err_mean'] * 1000
    labels = [r['label'] for r in rows]
    means = [r['vec_err_mean'] * 1000 for r in rows]
    colors = ['#4a90d9' if not r['compensate'] else '#2ca02c' for r in rows]

    fig, ax = plt.subplots(figsize=(13, 4.8))
    bars = ax.bar(range(len(rows)), means, color=colors)
    ax.axhline(base, color='gray', linestyle='--', linewidth=1, label='baseline (%.1f mm)' % base)
    for i, (bar, r) in enumerate(zip(bars, rows)):
        ax.text(i, bar.get_height() + 0.5, '%.1f' % bar.get_height(), ha='center', fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('mean vector formation error (mm)')
    ax.set_title('Combined communication faults, leader-follower formation (blue=raw, green=compensated)')
    ax.legend(); ax.grid(axis='y', alpha=.3)
    fig.tight_layout()
    fig_path = os.path.join(output_folder, 'fault_combo.png')
    fig.savefig(fig_path, dpi=150)
    print('[combo] figure written to', fig_path)

    #### Interaction summary ############################################
    lookup = {r['label']: r['vec_err_mean'] for r in rows}
    b = lookup['baseline']
    print('[combo] interaction analysis (mean error, mm):')
    print('  singles:      D8 %.1f | L30%% %.1f | N5cm %.1f' %
          (lookup['D8']*1000, lookup['L30%']*1000, lookup['N5cm']*1000))
    print('  pairs:        D8+L30%% %.1f | D8+N5cm %.1f | L30%%+N5cm %.1f' %
          (lookup['D8+L30%']*1000, lookup['D8+N5cm']*1000, lookup['L30%+N5cm']*1000))
    print('  additive pred D8+L30%%: %.1f | D8+N5cm: %.1f | L30%%+N5cm: %.1f' %
          ((lookup['D8']+lookup['L30%']-b)*1000, (lookup['D8']+lookup['N5cm']-b)*1000,
           (lookup['L30%']+lookup['N5cm']-b)*1000))
    print('  triple:       D8+L30%%+N5cm %.1f (comp %.1f)' %
          (lookup['D8+L30%+N5cm']*1000, lookup['D8+L30%+N5cm+comp']*1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined-fault sweep for leader-follower formation')
    parser.add_argument('--output_folder', default='results', type=str, help='Folder for CSV and figure (default: "results")', metavar='')
    ARGS = parser.parse_args()
    main(ARGS.output_folder)
