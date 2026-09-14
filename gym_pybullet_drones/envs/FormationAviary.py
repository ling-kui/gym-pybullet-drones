import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class FormationAviary(BaseRLAviary):
    """Multi-agent RL problem: drones keep a line formation at fixed slots.

    Each drone must hover at its own slot of a formation whose geometry is
    defined by `FORMATION_OFFSET`. On top of the per-drone slot reward
    inherited in spirit from `MultiHoverAviary`, a relative-position term
    rewards keeping the inter-drone geometry of the design formation. This
    extra term is what communication-constrained experiments (delayed or
    lost neighbor states) are meant to degrade in later phases.
    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 formation_separation: float=1.0,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a formation-keeping RL environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            Number of drones in the formation (default: 2).
        formation_separation : float, optional
            In-plane spacing between adjacent slots of the line formation, in meters (default: 1.0).
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision).
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thrust and torques, or waypoint with PID control).

        """
        self.EPISODE_LEN_SEC = 8
        #### Drones start on the ground, already in formation spacing ####
        if initial_xyzs is None:
            initial_xyzs = np.array([[(i - (num_drones-1)/2) * formation_separation, 0, .1]
                                     for i in range(num_drones)])
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        #### Slots: a line along X, centered on the origin, hovering at 1m ####
        self.FORMATION_SEPARATION = formation_separation
        self.FORMATION_OFFSET = np.array([[(i - (num_drones-1)/2) * formation_separation, 0, 0]
                                          for i in range(num_drones)])
        self.TARGET_POS = np.hstack([self.FORMATION_OFFSET[:, 0:2],
                                     np.ones((num_drones, 1))])

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward: slot-keeping terms plus a formation-geometry term.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = 0
        for i in range(self.NUM_DRONES):
            ret += max(0, 2 - np.linalg.norm(self.TARGET_POS[i, :]-states[i][0:3])**4)
        #### Formation term: consecutive drones keep the designed spacing ####
        if self.NUM_DRONES > 1:
            actual_rel = states[:-1, 0:3] - states[1:, 0:3]
            desired_rel = self.FORMATION_OFFSET[:-1] - self.FORMATION_OFFSET[1:]
            form_err = np.linalg.norm(actual_rel - desired_rel, axis=1)
            ret += float(np.sum(np.maximum(0, 1 - 4*form_err**2)))
        return ret

    ################################################################################

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        dist = 0
        for i in range(self.NUM_DRONES):
            dist += np.linalg.norm(self.TARGET_POS[i, :]-states[i][0:3])
        if dist < .0001:
            return True
        else:
            return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out or a drone left the safety box.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
