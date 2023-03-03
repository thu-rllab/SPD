import numpy as np


def assign_reward(traj_data, mode_id, disc_trainer, num_modes=10, pseudo_reward_scale=10., reward_scale=0., traj_reward=None, **kwargs):
    traj_data = np.array(traj_data)
    pseudo_rewards = np.maximum(disc_trainer.score(traj_data, np.array([mode_id])), np.log(1 / num_modes) * 10)

    return pseudo_rewards * pseudo_reward_scale + np.array(traj_reward) * reward_scale, 0.
