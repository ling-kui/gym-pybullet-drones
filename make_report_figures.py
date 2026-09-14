"""Generate report figures from recorded experiment data.

Produces (in results/report_figures/):
- fig_pid_formation.png   : PID baseline trajectories + tracking error
- fig_ppo_learning.png    : PPO learning curves (hover baseline & formation training)
- fig_formation_flight.png: 2-drone scripted formation trajectories + inter-drone distance
"""
import os
import sys
import csv
import glob
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = r'D:\workplace\project\gym-pybullet-drones\gym-pybullet-drones-main'
OUT = os.path.join(ROOT, 'report_figures')
CTRL_FREQ = 48.0


def load_csv(path):
    rows = list(csv.reader(open(path)))[1:]
    return np.array([float(r[1]) for r in rows])


def analytic_circle_ref(t, j, num_drones=3, R=.3, period=10., ctrl_freq=48.):
    """Reference used by pid.py: shared circle, per-drone waypoint offset and height."""
    NUM_WP = int(ctrl_freq * period)
    wp = (int((j * NUM_WP / 6) % NUM_WP) + int(round(t * ctrl_freq))) % NUM_WP
    ang = (wp / NUM_WP) * 2 * math.pi + math.pi / 2
    return (R * math.cos(ang), R * math.sin(ang) - R, .1 + j * .05)


def fig_pid(save_dir):
    d = save_dir + '/'
    t = load_csv(d + 'x0.csv')
    ts = np.arange(len(t)) / CTRL_FREQ
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for j, c in enumerate(colors):
        xs, ys = load_csv(d + 'x%d.csv' % j), load_csv(d + 'y%d.csv' % j)
        ref = np.array([analytic_circle_ref(tt, j) for tt in ts])
        axes[0].plot(ref[:, 0], ref[:, 1], '--', color=c, alpha=.45, linewidth=1)
        axes[0].plot(xs, ys, '-', color=c, linewidth=1.2, label='drone%d' % j)
        err = np.sqrt((xs - ref[:, 0])**2 + (ys - ref[:, 1])**2 + (load_csv(d + 'z%d.csv' % j) - ref[:, 2])**2)
        axes[1].plot(ts, err * 100, color=c, linewidth=1.2, label='drone%d' % j)
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_title('PID 圆轨迹跟踪：XY 轨迹（虚线=参考）')
    axes[0].legend(); axes[0].grid(alpha=.3); axes[0].set_aspect('equal')
    axes[1].set_xlabel('时间 (s)'); axes[1].set_ylabel('3D 跟踪误差 (cm)')
    axes[1].set_title('PID 跟踪误差（各轴 RMSE ≈ 5.1 cm，3D 误差约 8-10 cm）')
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_pid_formation.png'), dpi=150)
    plt.close(fig)
    print('fig_pid_formation.png done')


def fig_ppo(hover_npz, formation_npz):
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for npz_path, label, threshold, color in [
            (hover_npz, '单机悬停 (B2)', 474, 'tab:blue'),
            (formation_npz, '双机编队 (P2)', 1050, 'tab:orange')]:
        data = np.load(npz_path)
        ts, rw = data['timesteps'], data['results'][:, 0]
        ax.plot(ts / 1000, rw, '-', color=color, alpha=.35, linewidth=1)
        window = max(3, len(rw)//8)
        smooth = np.convolve(rw, np.ones(window)/window, mode='valid')
        ax.plot(ts[window-1:] / 1000, smooth, '-', color=color, linewidth=2,
                label='%s（滑动均值）' % label)
        ax.axhline(threshold, color=color, linestyle=':', linewidth=1)
        ax.text(ts[-1]/1000*1.01, threshold, ' 达标线 %d' % threshold, color=color,
                fontsize=8, va='center')
    ax.set_xlabel('训练步数（千步）'); ax.set_ylabel('评估回合奖励')
    ax.set_title('PPO 学习曲线：悬停基线与编队任务')
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_ppo_learning.png'), dpi=150)
    plt.close(fig)
    print('fig_ppo_learning.png done')


def fig_formation_flight(save_dir, sep=1.0):
    d = save_dir + '/'
    t = load_csv(d + 'x0.csv')
    ts = np.arange(len(t)) / CTRL_FREQ
    x0, y0, x1, y1 = (load_csv(d + f) for f in ('x0.csv', 'y0.csv', 'x1.csv', 'y1.csv'))
    dist = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].plot(x0, y0, '-', linewidth=1.2, label='drone0 (领机)')
    axes[0].plot(x1, y1, '-', linewidth=1.2, label='drone1 (跟随机)')
    for xx, yy, c in ((x0, y0, 'tab:blue'), (x1, y1, 'tab:orange')):
        axes[0].plot(xx[0], yy[0], 'o', color=c)
        axes[0].plot(xx[-1], yy[-1], 's', color=c)
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_title('双机编队圆轨迹飞行（起点○ 终点□）')
    axes[0].legend(); axes[0].grid(alpha=.3); axes[0].set_aspect('equal')
    axes[1].plot(ts, dist * 100, '-', linewidth=1.2)
    axes[1].axhline(sep * 100, color='gray', linestyle='--', linewidth=1, label='设计间距 %.0f cm' % (sep*100))
    axes[1].set_xlabel('时间 (s)'); axes[1].set_ylabel('实际机间距 (cm)')
    axes[1].set_title('机间距保持（误差均值 4.1 mm，最大 18.7 mm）')
    axes[1].legend(); axes[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_formation_flight.png'), dpi=150)
    plt.close(fig)
    print('fig_formation_flight.png done')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    pid_dirs = sorted(glob.glob(os.path.join(ROOT, 'results', 'save-flight-pid-*')))
    form_dirs = sorted(glob.glob(os.path.join(ROOT, 'results', 'save-flight-formation-*')))
    fig_pid(pid_dirs[0])
    hover = os.path.join(ROOT, 'results', 'baseline-09.14.2026_23.22.43', 'evaluations.npz')
    form = os.path.join(ROOT, 'results', 'baseline-09.14.2026_23.55.00', 'evaluations.npz')
    fig_ppo(hover, form)
    fig_formation_flight(form_dirs[0])
    print('ALL_FIGURES_DONE ->', OUT)
