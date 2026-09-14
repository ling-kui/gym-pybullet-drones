# 基线记录（BASELINE）

课题：通信受限条件下的多无人机协同编队控制与鲁棒性评估
本文件记录所有基线实验的配置与结果，作为后续通信故障注入实验的对照参照系。

## 环境（所有基线共享）

| 项 | 值 |
| --- | --- |
| 代码提交 | `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`（main, "typos"） |
| 平台 | Windows 11 / x64，CPU only |
| Python | 3.12（`.venv` 虚拟环境） |
| 关键依赖 | pybullet 3.2.7（本地编译）、stable-baselines3 2.9.0、torch 2.14.0+cpu、gymnasium 1.3.0、numpy 2.5.3 |
| 随机种子 | env seed=0，PPO seed=0 |

## 基线 B1：单机/多机 PID 圆轨迹跟踪（传统控制）

- 命令：`python -c "from gym_pybullet_drones.examples.pid import run; run(plot=False)"`
- 配置：`CtrlAviary` + `DSLPIDControl`，3 架 cf2x，圆形轨迹 R=0.3m 周期 10s，控制 48Hz / 仿真 240Hz，时长 12s，层高 0.05m 间隔
- 结果目录：`results/save-flight-pid-09.14.2026_23.17.46/`（含各信号 CSV）
- 指标（对解析圆参考的跟踪 RMSE，N=575 控制步）：

| 无人机 | x RMSE | y RMSE | z RMSE |
| --- | --- | --- | --- |
| drone0 | 0.0611 m | 0.0617 m | 0.0017 m |
| drone1 | 0.0556 m | 0.0683 m | 0.0016 m |
| drone2 | 0.0654 m | 0.0605 m | 0.0016 m |
| **整体 3D** | **0.0508 m** | | |

结论：DSL-PID 水平跟踪误差约 6cm、高度跟踪优于 2mm，与文献量级一致，可作为无故障参照。

## 基线 B2：PPO 单机悬停（学习控制）

- 命令：`python baseline_train.py --multiagent false --timesteps 200000`
- 配置：`HoverAviary`，观测 KIN、动作 ONE_D_RPM，PPO MlpPolicy（seed=0），评估回调 eval_freq=1000、达标阈值 474
- 产物：`results/baseline-*/best_model.zip`（达标即停）、`final_model.zip`、`evaluations.npz`
- 指标：见训练日志 `results/baseline_train_log.txt` 与下方"结果"（训练完成后回填）

结果（2026-09-14，200k 步预算，提前达标停止）：

- 达标情况：42,000 步时评估奖励 474.15 ≥ 474 阈值，`StopTrainingOnRewardThreshold` 触发提前停止
- 训练速度：约 620 steps/s（CPU，单环境）
- 评估曲线：初始 ≈ 336 → 15k 步 ≈ 442 → 31k 步起稳定在 ≈ 472-474 → 42k 步达标
- 产物目录：`results/baseline-09.14.2026_23.22.43/`（best_model.zip、final_model.zip、evaluations.npz）
- 备注：评估曲线中段（9k-14k、25k-30k）出现回落，为 PPO 探索波动，正常现象
- 模型验证：`PPO.load(best_model.zip)` 加载 + 240 步（5s）确定性回放回报 469.94（12s 满奖励 474），链路完整

## 已知兼容性问题（已修复）

torch 2.14 无法直接读取 stable-baselines3 2.9.0 传入的 zip 文件流（`PytorchStreamReader ... .data/serialization_id` miniz 错误），导致 `PPO.load()` 报"checkpoint corrupted"。
修复方式：在 `.venv\Lib\site-packages\sitecustomize.py` 放置自动加载的垫片，将所有非 BytesIO 的可读流包装后调用原始 `torch.load`。该文件属于虚拟环境、不进仓库；重建环境时需按此说明重新放置。

## 多机冒烟

- `HoverAviary` 无头 reset+step 验证通过（obs shape (1,72)）
- `baseline_train.py --timesteps 300` 冒烟通过：评估回调、checkpoint 保存正常，初始评估奖励 ≈ 336

## 复现方式

```bat
cd gym-pybullet-drones-main
.venv\Scripts\activate
python -c "from gym_pybullet_drones.examples.pid import run; run(plot=False)"
python baseline_train.py --multiagent false --timesteps 200000
```

RMSE 计算方式：按 pid.py 的航点公式解析重建圆形参考（drone j 相位偏移 NUM_WP/6，层高 0.1+0.05j），逐控制步求位置误差均方根。
