"""Two-drone formation flight on a circular trajectory with PID control.

Demonstrates a simple classical-control formation: `num_drones` keep a line
formation of `separation` meters between neighbours while every drone tracks
the same circular path (drone 0 is the reference; the others are shifted by
their formation offset). This script is the deterministic testbed where
communication faults (delay, loss, noise on neighbor states) will be
injected in phase 3.

At the end it prints two formation metrics:
- inter-drone distance error against the designed separation
- per-drone tracking RMSE against the analytic slot reference

Example
-------

    $ python formation_flight.py
    $ python formation_flight.py --gui false --plot false
"""
import os
import time
import argparse
from datetime import datetime
import numpy as np

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
        h=DEFAULT_H
        ):
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

    #### Run the simulation #################################################
    action = np.zeros((num_drones, 4))
    wp = 0
    dist_err, slot_sq = [], [[] for i in range(num_drones)]
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        ang = (wp/NUM_WP)*(2*np.pi)+np.pi/2
        ref_xy = np.array([R*np.cos(ang), R*np.sin(ang)-R])
        obs, reward, terminated, truncated, info = env.step(action)
        for j in range(num_drones):
            target_pos = np.hstack([ref_xy + FORMATION_OFFSET[j, 0:2], h])
            action[j, :], _, _ = ctrl[j].computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                                 state=obs[j],
                                                                 target_pos=target_pos,
                                                                 target_rpy=INIT_RPYS[j, :]
                                                                 )
            pos_err = np.linalg.norm(obs[j][0:3]-target_pos)
            if i >= 2*control_freq_hz:  # steady-state metric: skip the take-off transient
                slot_sq[j].append(pos_err**2)
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([target_pos, INIT_RPYS[j, :], np.zeros(6)])
                       )
        dist = np.linalg.norm(obs[0][0:3]-obs[1][0:3])
        dist_err.append(abs(dist - separation))
        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)
        wp = wp + 1 if wp < (NUM_WP-1) else 0

    #### Close the environment and report ###################################
    env.close()
    logger.save()
    logger.save_as_csv("formation")

    print("[formation] designed separation: %.2f m" % separation)
    print("[formation] inter-drone distance error: mean %.4f m | max %.4f m" % (np.mean(dist_err), np.max(dist_err)))
    for j in range(num_drones):
        print("[formation] drone%d slot tracking RMSE (t>2s): %.4f m" % (j, np.sqrt(np.mean(slot_sq[j]))))

    if plot:
        logger.plot()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-drone PID formation flight on a circular trajectory')
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,  type=int,       help='Number of drones (default: 2)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,         type=str2bool,  help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,        type=str2bool,  help='Whether to plot the results (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,    type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,       type=int, help='Duration of the simulation in seconds (default: 12)', metavar='')
    parser.add_argument('--separation',         default=DEFAULT_SEPARATION,         type=float, help='Formation spacing between neighbours in meters (default: 1.0)', metavar='')
    parser.add_argument('--h',                  default=DEFAULT_H,                  type=float, help='Formation flight height in meters (default: 1.0)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,      type=str,   help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))
