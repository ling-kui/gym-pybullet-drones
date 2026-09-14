"""Two-drone formation flight on a circular trajectory with PID control.

Two control architectures are supported:

- `scripted` (default): every drone tracks its own pre-scripted reference
  (the circle shifted by its formation offset). No inter-drone communication
  is needed; this is the ideal reference run.
- `leader-follower`: drone 0 (leader) tracks the circle; drone 1 (follower)
  targets `leader position + formation offset`, where the leader position is
  received over a `CommLink` that can delay, drop, and corrupt packets. All
  communication faults live in that link only—each drone's own sensing is
  perfect. With `compensate=True` the follower dead-reckons the leader
  position from the (old) received velocity while the sample ages.

Prints formation metrics and returns them as a dict:
- vector formation error ||p1 - p0 - offset|| (direction-aware)
- inter-drone distance error | ||p1 - p0|| - separation |
- leader slot tracking RMSE (steady state, t > 2 s)

Example
-------

    $ python formation_flight.py
    $ python formation_flight.py --gui false --plot false
    $ python formation_flight.py --mode leader-follower --delay_steps 8 --loss 0.3 --compensate true
"""
import os
import time
import argparse
from datetime import datetime
import numpy as np

from gym_pybullet_drones.utils.comm_link import CommLink
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_PLOT = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False
DEFAULT_SEPARATION = 1.0
DEFAULT_H = 1.0
DEFAULT_MODE = 'scripted'


def run(drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        plot=DEFAULT_PLOT,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB,
        separation=DEFAULT_SEPARATION,
        h=DEFAULT_H,
        mode=DEFAULT_MODE,
        delay_steps=0,
        loss=0.0,
        noise=0.0,
        compensate=False,
        seed=0
        ):
    assert mode in ('scripted', 'leader-follower'), "mode must be 'scripted' or 'leader-follower'"
    assert num_drones == 2 or mode == 'scripted', "leader-follower mode currently supports exactly 2 drones"

    #### Line formation offsets along X, centered on the reference drone ####
    FORMATION_OFFSET = np.array([[(i - (num_drones-1)/2) * separation, 0, 0]
                                 for i in range(num_drones)])

    #### Circular reference (passes through the origin), period 10 s #######
    R, PERIOD = .3, 10
    INIT_XYZS = np.hstack([np.zeros((num_drones, 2)) + R*np.array([np.cos(np.pi/2), np.sin(np.pi/2)-1]),
                           .1*np.ones((num_drones, 1))]) + FORMATION_OFFSET
    INIT_RPYS = np.zeros((num_drones, 3))
    NUM_WP = control_freq_hz*PERIOD
    TARGET_XY = np.zeros((NUM_WP, 2))
    for i in range(NUM_WP):
        ang = (i/NUM_WP)*(2*np.pi)+np.pi/2
        TARGET_XY[i, :] = R*np.cos(ang), R*np.sin(ang)-R

    #### Create the environment ############################################
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     obstacles=False
                     )

    #### Initialize the logger and the controllers #########################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )
    ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    #### Communication link (leader-follower mode only) #####################
    if mode == 'leader-follower':
        link = CommLink(delay_steps=delay_steps, loss_prob=loss,
                        noise_std=noise, seed=seed)
        link.prefill(np.hstack([INIT_XYZS[0, 0:3], np.zeros(3)]))  # initial leader state known
        last_rx = np.hstack([INIT_XYZS[0, 0:3], np.zeros(3)])  # newest delivered leader (pos, vel)
        last_age = 0

    #### Run the simulation #################################################
    action = np.zeros((num_drones, 4))
    wp = 0
    dist_err, vec_err, leader_sq = [], [], []
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        ang = (wp/NUM_WP)*(2*np.pi)+np.pi/2
        ref_xy = np.array([R*np.cos(ang), R*np.sin(ang)-R])
        obs, reward, terminated, truncated, info = env.step(action)

        if mode == 'leader-follower':
            link.send(np.hstack([obs[0][0:3], obs[0][10:13]]))  # leader pos + vel
            rx, age = link.recv()
            if rx is not None:  # on packet loss, hold the last delivered sample
                last_rx, last_age = rx, age
            rel = FORMATION_OFFSET[1] - FORMATION_OFFSET[0]  # follower slot RELATIVE to the leader
            if compensate:
                est = last_rx[0:3] + last_rx[3:6]*(last_age*env.CTRL_TIMESTEP)  # dead reckoning
                target1_xy, target1_vel = est[0:2] + rel[0:2], last_rx[3:6]
            else:
                target1_xy, target1_vel = last_rx[0:2] + rel[0:2], last_rx[3:6]

        for j in range(num_drones):
            if mode == 'leader-follower' and j == 1:
                state = obs[1]
                action[1, :], _, _ = ctrl[1].computeControl(control_timestep=env.CTRL_TIMESTEP,
                                                            cur_pos=state[0:3],
                                                            cur_quat=state[3:7],
                                                            cur_vel=state[10:13],
                                                            cur_ang_vel=state[13:16],
                                                            target_pos=np.hstack([target1_xy, h]),
                                                            target_rpy=INIT_RPYS[1, :],
                                                            target_vel=np.hstack([target1_vel[0:2], 0.0])
                                                            )
            else:
                target_pos = np.hstack([ref_xy + FORMATION_OFFSET[j, 0:2], h])
                action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                     state=obs[j],
                                                                     target_pos=target_pos,
                                                                     target_rpy=INIT_RPYS[j, :]
                                                                     )
            if j == 0 and i >= 2*control_freq_hz:  # steady-state metric
                leader_sq.append(np.linalg.norm(obs[0][0:3]-target_pos)**2)
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([target_pos, INIT_RPYS[j, :], np.zeros(6)])
                       )
        form_vec = obs[1][0:3] - obs[0][0:3] - (FORMATION_OFFSET[1]-FORMATION_OFFSET[0])
        vec_err.append(np.linalg.norm(form_vec))
        dist_err.append(abs(np.linalg.norm(obs[1][0:3]-obs[0][0:3]) - separation))
        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
        wp = wp + 1 if wp < (NUM_WP-1) else 0

    #### Close the environment and report ###################################
    env.close()
    logger.save()
    logger.save_as_csv("formation")

    metrics = {
        'mode': mode, 'delay_steps': int(delay_steps), 'loss': float(loss),
        'noise': float(noise), 'compensate': bool(compensate),
        'dist_err_mean': float(np.mean(dist_err)), 'dist_err_max': float(np.max(dist_err)),
        'vec_err_mean': float(np.mean(vec_err)), 'vec_err_max': float(np.max(vec_err)),
        'vec_err_rms': float(np.sqrt(np.mean(np.square(vec_err)))),
        'leader_rmse': float(np.sqrt(np.mean(leader_sq))),
    }
    print("[formation] mode=%s delay=%d loss=%.2f noise=%.3f compensate=%s" %
          (mode, delay_steps, loss, noise, compensate))
    print("[formation] designed separation: %.2f m" % separation)
    print("[formation] vector formation error: mean %.4f m | max %.4f m | rms %.4f m" %
          (metrics['vec_err_mean'], metrics['vec_err_max'], metrics['vec_err_rms']))
    print("[formation] inter-drone distance error: mean %.4f m | max %.4f m" %
          (metrics['dist_err_mean'], metrics['dist_err_max']))
    if leader_sq:
        metrics['leader_rmse'] = float(np.sqrt(np.mean(leader_sq)))
        print("[formation] leader slot tracking RMSE (t>2s): %.4f m" % metrics['leader_rmse'])

    if plot:
        logger.plot()
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-drone PID formation flight with optional communication faults')
    parser.add_argument('--mode',                default=DEFAULT_MODE,        type=str,      help='scripted (no comms) or leader-follower (follower needs leader state)', metavar='', choices=['scripted', 'leader-follower'])
    parser.add_argument('--num_drones',          default=DEFAULT_NUM_DRONES,  type=int,      help='Number of drones (default: 2, scripted mode only)', metavar='')
    parser.add_argument('--gui',                 default=DEFAULT_GUI,         type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',                default=DEFAULT_PLOT,        type=str2bool, help='Whether to plot the results (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz',  default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',     default=DEFAULT_CONTROL_FREQ_HZ,    type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',        default=DEFAULT_DURATION_SEC,       type=int, help='Duration of the simulation in seconds (default: 12)', metavar='')
    parser.add_argument('--separation',          default=DEFAULT_SEPARATION,         type=float, help='Formation spacing between neighbours in meters (default: 1.0)', metavar='')
    parser.add_argument('--h',                   default=DEFAULT_H,                  type=float, help='Formation flight height in meters (default: 1.0)', metavar='')
    parser.add_argument('--delay_steps',         default=0,                          type=int,   help='Leader-follower link delay in control steps (default: 0)', metavar='')
    parser.add_argument('--loss',                default=0.0,                        type=float, help='Leader-follower packet loss probability in [0,1] (default: 0)', metavar='')
    parser.add_argument('--noise',               default=0.0,                        type=float, help='Additive Gaussian noise std on received leader state, meters (default: 0)', metavar='')
    parser.add_argument('--compensate',          default=False,                      type=str2bool, help='Follower dead-reckons leader position from received velocity (default: False)', metavar='')
    parser.add_argument('--seed',                default=0,                          type=int,   help='Random seed of the communication link (default: 0)', metavar='')
    parser.add_argument('--output_folder',       default=DEFAULT_OUTPUT_FOLDER,      type=str,   help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
