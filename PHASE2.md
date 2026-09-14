# 第二阶段记录：两机编队任务（Phase 2）

目标：把"两架无人机各悬停各的"升级为**编队保持**任务，为第三阶段（通信延迟/丢包/噪声注入）提供两类被测对象。

## 新增内容

| 文件 | 说明 |
| --- | --- |
| `gym_pybullet_drones/envs/FormationAviary.py` | 编队保持 RL 环境（新增） |
| `baseline_train.py` | 训练脚本新增 `--task formation`（修改） |
| `formation_flight.py` | 传统控制（PID）双机编队圆轨迹飞行演示（新增） |
| `gym_pybullet_drones/envs/__init__.py` | 注册 FormationAviary（修改） |

## RL 编队环境：FormationAviary

- 队形：沿 X 轴的直线编队，相邻槽位间距 1m（可配 `formation_separation`），悬停高度 1m
- 初始：无人机在地面（z=0.1m）按相同间距排开，需自主起飞到槽位
- 观测：每机 12 维运动学（位置/姿态/速度/角速度）+ 最近 0.5s 动作缓冲 → 双机堆叠为 `(2, 27)`（30Hz 控制频率）
- 动作：ONE_D_RPM（归一化总转速扰动，与基线 B2 一致）
- 回合：8s（30Hz × 240 步）；出安全边界（|x|,|y|>2m、z>2m、倾角>0.4rad）截断
- **奖励 = 槽位项 + 队形项**：
  - 槽位项：每机 `max(0, 2 - d_i^4)`，d_i 为到自身槽位的距离（继承 MultiHover 风格）
  - 队形项：相邻机对 `max(0, 1 - 4·e_j²)`，e_j 为实际相对位置与设计相对位置之差的范数
  - 每步理论上限 5（4 槽位 + 1 队形），整回合上限 1200
- 训练达标阈值：1050

### RL 训练结果

- 配置：`python baseline_train.py --task formation --timesteps 400000`（PPO MlpPolicy，seed=0，与基线 B2 一致）
- 结果：待训练完成后回填（训练动态：开局 ~877（起飞前即站位于队形），5k 步冲至 ~1041，随后探索期回落至 ~500-600，为 PPO 典型现象）

## 传统控制编队演示：formation_flight.py

- 配置：`CtrlAviary` + `DSLPIDControl`，2 架 cf2x，同相位圆轨迹（R=0.3m、周期 10s、高度 1m），编队偏置 ±0.5m，48Hz 控制 / 240Hz 仿真，12s
- 结果（无头模式，2026-09-14）：
  - **机间距误差（vs 设计 1.000m）：平均 4.1mm，最大 18.7mm** —— 队形几何保持极好
  - 单机槽位跟踪 RMSE（t>2s 稳态）：drone0 9.5cm / drone1 10.0cm（含起飞爬升后追赶参考的剩余瞬态）
  - 数据目录：`results/save-flight-formation-*/`

## 与第三阶段的衔接

- `formation_flight.py`（PID 链路）是**通信故障注入的首选测试床**：轨迹确定、指标清晰（机间距误差 + 槽位 RMSE），注入"邻机状态延迟/丢包/噪声"后可直接量化退化。
- `FormationAviary`（RL 链路）第三阶段将把观测中的邻机信息改造为"经通信链路获取"，对比理想通信 vs 受限通信的训练/执行效果。
