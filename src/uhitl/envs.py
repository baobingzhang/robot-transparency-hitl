"""gym-hil Panda pick-cube environment helpers: construction, flat observations, exact state snapshots."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from gym_hil.envs import PandaPickCubeGymEnv
from gym_hil.wrappers.hil_wrappers import (
    DEFAULT_EE_STEP_SIZE,
    EEActionWrapper,
    GripperPenaltyWrapper,
)

OBS_DIM = 21  # agent_pos (7 qpos + 7 qvel + 1 gripper + 3 tcp) + block position (3)
ACT_DIM = 4  # dx, dy, dz in [-1, 1]; gripper in [0, 2]
ACT_LOW = np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32)
ACT_HIGH = np.array([1.0, 1.0, 1.0, 2.0], dtype=np.float32)


def make_env(
    seed: int = 0, random_block_position: bool = True, gripper_penalty: float = -0.02
):
    base = PandaPickCubeGymEnv(
        seed=seed, random_block_position=random_block_position, reward_type="sparse"
    )
    env = GripperPenaltyWrapper(base, penalty=gripper_penalty)
    env = EEActionWrapper(
        env, ee_action_step_size=DEFAULT_EE_STEP_SIZE, use_gripper=True
    )
    return env


def flat_obs(obs: dict) -> np.ndarray:
    return np.concatenate([obs["agent_pos"], obs["environment_state"]]).astype(
        np.float32
    )


def tcp_pos(obs_flat: np.ndarray) -> np.ndarray:
    return obs_flat[15:18]


def block_pos(obs_flat: np.ndarray) -> np.ndarray:
    return obs_flat[18:21]


def gripper_ctrl(obs_flat: np.ndarray) -> float:
    return float(obs_flat[14])


class Snapshot:
    """Exact copy of the MuJoCo physics state plus the task variables needed to resume an episode."""

    def __init__(self, env):
        u = env.unwrapped
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.empty(mujoco.mj_stateSize(u._model, spec))
        mujoco.mj_getState(u._model, u._data, self.state, spec)
        self.z_init = u._z_init
        self.z_success = u._z_success
        self.last_gripper_pos = env.env.last_gripper_pos  # GripperPenaltyWrapper state

    def restore(self, env) -> np.ndarray:
        u = env.unwrapped
        mujoco.mj_setState(
            u._model, u._data, self.state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        mujoco.mj_forward(u._model, u._data)
        u._z_init = self.z_init
        u._z_success = self.z_success
        env.env.last_gripper_pos = self.last_gripper_pos
        return flat_obs(u._compute_observation())
