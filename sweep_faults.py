"""Batch sweep of communication faults over the leader-follower formation.

Runs `formation_flight.run()` in leader-follower mode across a grid of link
conditions (delay, packet loss, additive noise), collects the formation
metrics, writes a CSV table and a degradation-curve figure:

- delay sweep with and without dead-reckoning compensation
- packet-loss sweep
- noise sweep

Example
-------

    $ python sweep_faults.py
"""
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from formation_flight import run

DELAY_STEPS = [0, 1, 2, 4, 8, 12, 16]
LOSS_RATES = [0.0, 0.1, 0.2, 0.3, 0.5]
NOISE_STDS = [0.0, 0.01, 0.02, 0.05, 0.1]
SEED = 0


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

    def do(**kw):
        m = quiet(mode='leader-follower', **kw)
        rows.append(m)
        print("[sweep] delay=%-2d loss=%.2f noise=%.3f comp=%-5s -> vec rms %.4f m | mean %.4f m | max %.4f m"
              % (m['delay_steps'], m['loss'], m['noise'], m['compensate'],
                 m['vec_err_rms'], m['vec_err_mean'], m['vec_err_max']))
        return m

    print('=== delay sweep (with / without compensation) ===')
    delay_rows = {'off': [], 'on': []}
    for d in DELAY_STEPS:
        delay_rows['off'].append(do(delay_steps=d))
        delay_rows['on'].append(do(delay_steps=d, compensate=True))

    print('=== packet loss sweep ===')
    loss_rows = []
    for p in LOSS_RATES:
        loss_rows.append(do(loss=p))

    print('=== noise sweep ===')
    noise_rows = []
    for n in NOISE_STDS:
        noise_rows.append(do(noise=n))

    #### CSV ############################################################
    csv_path = os.path.join(output_folder, 'fault_sweep.csv')
    keys = ['mode', 'delay_steps', 'loss', 'noise', 'compensate',
            'vec_err_mean', 'vec_err_max', 'vec_err_rms', 'dist_err_mean', 'dist_err_max', 'leader_rmse']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print('[sweep] CSV written to', csv_path)

    #### Figure ##########################################################
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    delay_s = [d / 48.0 * 1000 for d in DELAY_STEPS]  # control period = 1/48 s
    axes[0].plot(delay_s, [m['vec_err_mean'] for m in delay_rows['off']], 'o-', label='raw')
    axes[0].plot(delay_s, [m['vec_err_mean'] for m in delay_rows['on']], 's-', label='compensated')
    axes[0].set_xlabel('link delay (ms)')
    axes[0].set_ylabel('formation error, mean (m)')
    axes[0].set_title('Delay')
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot([p * 100 for p in LOSS_RATES], [m['vec_err_mean'] for m in loss_rows], 'o-')
    axes[1].set_xlabel('packet loss rate (%)')
    axes[1].set_ylabel('formation error, mean (m)')
    axes[1].set_title('Packet loss')
    axes[1].grid(alpha=.3)

    axes[2].plot([n * 100 for n in NOISE_STDS], [m['vec_err_mean'] for m in noise_rows], 'o-')
    axes[2].set_xlabel('state noise std (cm)')
    axes[2].set_ylabel('formation error, mean (m)')
    axes[2].set_title('Noise')
    axes[2].grid(alpha=.3)

    fig.suptitle('Leader-follower formation under communication faults (2 drones, circle R=0.3m, 48Hz)')
    fig.tight_layout()
    fig_path = os.path.join(output_folder, 'fault_sweep.png')
    fig.savefig(fig_path, dpi=150)
    print('[sweep] figure written to', fig_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sweep communication faults over leader-follower formation')
    parser.add_argument('--output_folder', default='results', type=str, help='Folder for CSV and figure (default: "results")', metavar='')
    ARGS = parser.parse_args()
    main(ARGS.output_folder)
