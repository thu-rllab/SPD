import numpy as np
import torch
from copy import deepcopy
from functools import reduce

from .wsre import wsre, wsre_batch

def assign_reward(traj_data, target_data_batches, pseudo_reward_scale=10., reward_scale=0., traj_reward=None, use_batch_apwd=False, **kwargs):
    """
    traj_data: [state]
    target_data_batches: [[target_state], ...] for different skills.
    reward_scale: the scale of the original reward
    traj_reward: [reward]
    """
    traj_data = np.array(traj_data)
    
    if use_batch_apwd:
        srs_list = wsre_batch(traj_data, target_data_batches) # skill*bs_D
        sum_srs = srs_list.sum(1) #skill
    else:
        srs_list = []
        sum_srs = []
        for target_data_batch in target_data_batches:
            pseudo_reward = wsre(traj_data, target_data_batch)
            sum_srs.append(np.sum(pseudo_reward))
            srs_list.append(pseudo_reward)

    min_dist_idx = np.argmin(sum_srs)
    srs = srs_list[min_dist_idx]

    rewards = np.zeros_like(srs)
    norm_scale = len(target_data_batches[0]) / len(traj_data)
    for i in range(len(traj_data)):
        rewards[i] = srs[i] * pseudo_reward_scale * norm_scale
        if traj_reward is not None:
            rewards[i] += traj_reward[i] * reward_scale

    return rewards, 0.
