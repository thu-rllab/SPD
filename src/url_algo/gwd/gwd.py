import torch
import numpy as np
import time
from copy import deepcopy
from scipy.optimize import linear_sum_assignment


def node_cost_st(cost_s, cost_t, p_s=None, p_t=None, loss_type="L2"):
    """
    Args:
        cost_s (torch.Tensor): [K, T, N, N], where T is the length of the control trajectory & N is the number of the agents. Here data around dim-K is the same.
        cost_t (torch.Tensor): [K, B, M, M], where K is the number of the skills & B is the length of the sampled batch.
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]
    
    Returns:
        cost_st (torch.Tensor): [K, T, B, N, M]
    """
    dim_n = cost_s.shape[-1]
    dim_k, dim_b, dim_m = cost_t.shape[:3]
    
    with torch.no_grad():
        if loss_type == "L2":
            f1_st = torch.matmul(cost_s ** 2, p_s).repeat(1, 1, 1, dim_m)                            # [K, T, N, 1] -> [K, T, N, M]

            f2_st = torch.matmul(cost_t ** 2, p_t).permute(0, 1, 3, 2).repeat(1, 1, dim_n, 1)        # [K, B, 1, M] -> [K, B, N, M]
            cost_st = f1_st.unsqueeze(2) + f2_st.unsqueeze(1)                                       # [K, T, B, N, M]
        else:
            raise NotImplementedError
    
    return cost_st


def node_cost(cost_s, cost_t, trans, p_s=None, p_t=None, loss_type="L2"):
    """
    Args:
        cost_s (torch.Tensor): [K, T, N, N], where T is the length of the control trajectory & N is the number of the agents. Here data around dim-K is the same.
        cost_t (torch.Tensor): [K, B, M, M], where K is the number of the skills & B is the length of the sampled batch.
        trans (torch.Tensor): [K, T, B, N, M].
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]
    
    Returns:
        cost (torch.Tensor): [K, T, B, N, M]
    """
    dim_t, dim_n = cost_s.shape[1:3]
    dim_k, dim_b, dim_m = cost_t.shape[:3]

    cost_st = node_cost_st(cost_s, cost_t, p_s, p_t, loss_type)
    if loss_type == "L2":
        cost = cost_st - 2 * torch.matmul(
            torch.matmul(cost_s.unsqueeze(2).repeat(1, 1, dim_b, 1, 1), trans),
            cost_t.unsqueeze(1).repeat(1, dim_t, 1, 1, 1)
        )
    else:
        raise NotImplementedError
    return cost


def sinkhorn_knopp_iteration(cost, trans0=None, p_s=None, p_t=None,
                             a: torch.Tensor = None, beta: float = 1e-1,
                             error_bound: float = 1e-3, max_iter: int = 50):
    """
    Sinkhorn-Knopp iteration algorithm

    When initial optimal transport "trans0" is not available, the function solves
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * <log(trans), trans>

    When initial optimal transport "trans0" is given, the function solves:
        min_{trans in Pi(p_s, p_t)} <cost, trans> + beta * KL(trans || trans0)

    Args:
        cost (torch.Tensor): [K, T, B, N, M], representing batch of distance between nodes.
        trans0 (torch.Tensor): [K, T, B, N, M], representing the optimal transport over the episode.
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]

        a: representing the dual variable
        beta: the weight of entropic regularizer
        error_bound: the error bound to check convergence
        max_iter: the maximum number of iterations

    Returns:
        trans: optimal transport
        a: updated dual variable

    """
    dim_k, dim_t, dim_b, dim_n, dim_m = cost.shape

    if a is None:
        a = torch.ones((cost.shape[-2], 1), device=cost.device) / cost.shape[-2]

    if p_s is None:
        p_s = torch.ones((cost.shape[-2], 1), device=cost.device) / cost.shape[-2]

    if p_t is None:
        p_t = torch.ones((cost.shape[-1], 1), device=cost.device) / cost.shape[-1]

    if trans0 is not None:
        kernel = torch.exp(-cost / beta) * trans0
    else:
        kernel = torch.exp(-cost / beta)

    relative_error = torch.ones(dim_k, dim_t, dim_b) * float("inf")
    indicater_mat = relative_error > error_bound
    iter_i = 0
    while torch.sum(indicater_mat) >= 1. and iter_i < max_iter:
        b = torch.div(p_t, torch.matmul(kernel.permute(0, 1, 2, 4, 3), a) + 1e-10)
        a_new = torch.div(p_s, torch.matmul(kernel, b) + 1e-10)

        relative_error = torch.div(torch.sum(torch.abs(a_new - a), dim=(-2, -1)), torch.sum(torch.abs(a), dim=(-2, -1)) + 1e-10)
        indicater_mat = relative_error > error_bound
        a = a_new
        iter_i += 1
    trans = torch.matmul(a, b.permute(0, 1, 2, 4, 3)) * kernel

    return trans, a


def gromov_wasserstein_discrepancy(cost_s, cost_t, ot_hyperparams, trans0=None, p_s=None, p_t=None):
    """
    Args:
        cost_s (torch.Tensor): [K, T, N, N], where T is the length of the control trajectory & N is the number of the agents. Here data around dim-K is the same.
        cost_t (torch.Tensor): [K, B, M, M], where K is the number of the skills & B is the length of the sampled batch.
        ot_hyperparams (dict): hyper-parameters for the optimal transport algorithm.
        trans0 (torch.Tensor): [K, T, B, N, M].
        p_s (torch.Tensor): [N, 1]
        p_t (torch.Tensor): [M, 1]

    Returns:
        trans (torch.Tensor): [K, T, B, N, M].
        d_gw (torch.Tensor): [K, T, B]. The gromov-wasserstein discrepancy between the episode of graphs & batch of target graphs.
    """
    dim_t, dim_n = cost_s.shape[1:3]
    dim_k, dim_b, dim_m = cost_t.shape[:3]

    if p_s is None:
        p_s = (1. / dim_n) * torch.ones(size=[dim_n, 1], device=cost_s.device)
    if p_t is None:
        p_t = (1. / dim_m) * torch.ones(size=[dim_m, 1], device=cost_s.device)
    
    if trans0 is None:
        trans0 = torch.matmul(p_s, p_t.T).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(dim_k, dim_t, dim_b, 1, 1)
    
    relative_error = torch.ones(dim_k, dim_t, dim_b) * float("inf")
    indicater_mat = relative_error > ot_hyperparams["iter_bound"]
    iter_t = 0
    while torch.sum(indicater_mat) >= 1. and iter_t < ot_hyperparams["outer_iteration"]:
        cost = node_cost(cost_s, cost_t, trans0, p_s, p_t, ot_hyperparams["loss_type"])
        trans, a = sinkhorn_knopp_iteration(
            cost=cost,
            trans0=trans0 if ot_hyperparams["ot_method"] == 'proximal' else None,
            p_s=p_s,
            p_t=p_t,
            error_bound=ot_hyperparams["sk_bound"],
            max_iter=ot_hyperparams["inner_iteration"]
        )
        relative_error = torch.div(torch.sum(torch.abs(trans - trans0), dim=(-2, -1)), torch.sum(torch.abs(trans0), dim=(-2, -1)) + 1e-10)
        indicater_mat = relative_error > ot_hyperparams["iter_bound"]
        trans0 = trans
        iter_t += 1
        
    cost = node_cost(cost_s, cost_t, trans, p_s, p_t, ot_hyperparams["loss_type"])
    d_gw = torch.sum(cost * trans, dim=(-2, -1))

    return trans, d_gw


def calc_graph_discrepancy(traj_data, target_data_batches, ot_hyperparams, device="cuda", sparse_return=False, no_match=False):
    """
    Args:
        traj_data: [graph(torch.Tensor)] with graph.shape [N, N]
        target_data_batches: [[target_state], ...] for different skills.
    Returns:
        min_dis_skill: int. The skill with minimal discrepancy.
        pseudo_rewards: np.array with shape [T]. The GWD for the trajectory.
    """
    def convert_to_tensor(list_data, device="cuda"):
        """
        Args:
            list_data (list of torch.Tensor / np.ndarray)
        """
        if isinstance(list_data, torch.Tensor):
            tensor = list_data.to(device)
        elif isinstance(list_data[0], torch.Tensor):
            tensor = torch.stack(list_data).to(device)
        elif isinstance(list_data[0], np.ndarray):
            tensor = torch.as_tensor(list_data, device=device)
        else:
            raise ValueError
        return tensor

    cost_s = convert_to_tensor(traj_data, device)                   # [T, N, N]
    cost_t = convert_to_tensor(target_data_batches, device)         # [K, B, M, M]
    with torch.no_grad():
        _, d_gw = gromov_wasserstein_discrepancy(cost_s.unsqueeze(0).repeat(cost_t.shape[0], 1, 1, 1), cost_t, ot_hyperparams)

    def match_batch(d_gw):
        """
        Use Hungarian Algorithm to find the minimal matching between T and B.
        
        Args:
            d_gw (torch.Tensor): [K, T, B]
        
        Returns:
            pseudo_rewards_all (np.ndarray): [K, T]
        """
        pseudo_rewards_all = np.zeros(d_gw.shape[:2])
        d_gw = d_gw.detach().cpu().numpy()

        for k in range(d_gw.shape[0]):
            row_idx, col_idx = linear_sum_assignment(d_gw[k])
            pseudo_rewards_all[k, :] = d_gw[k, row_idx, col_idx]

        return pseudo_rewards_all

    pseudo_rewards_all = match_batch(d_gw)          # [K, T], np.ndarray
    pseudo_episode_return = np.sum(pseudo_rewards_all, axis=-1)

    min_dis_skill = np.argmin(pseudo_episode_return)
    pseudo_rewards = pseudo_rewards_all[min_dis_skill]
    dis_sp = np.sum(pseudo_rewards)

    assert not (sparse_return and no_match), "only one option can set to be true."
    if sparse_return:
        sparse_pseudo_rewards = np.zeros_like(pseudo_rewards)
        sparse_pseudo_rewards[-1] = dis_sp
        pseudo_rewards = sparse_pseudo_rewards

    if no_match:    
        d_gw = d_gw.detach().cpu().numpy()
        pseudo_rewards_all = np.diagonal(d_gw, axis1=1, axis2=2)
        pseudo_episode_return = np.sum(pseudo_rewards_all, axis=-1)
        min_dis_skill = np.argmin(pseudo_episode_return)
        pseudo_rewards = pseudo_rewards_all[min_dis_skill]

    return min_dis_skill, pseudo_rewards, dis_sp


def assign_reward(traj_data, target_data_batches, ot_hyperparams, pseudo_reward_scale=10.,
                  reward_scale=0., norm_reward=False, traj_reward=None, device="cuda",
                  sparse_return=False, no_match=False, **kwargs):
    """
    traj_data: [graph(torch.Tensor)] with graph.shape = [N, N]
    target_data_batches: [[target_state], ...] for different skills.
    reward_scale: the scale of the original reward
    traj_reward: [reward]
    """
    _, pseudo_rewards, dis_sp = calc_graph_discrepancy(
        traj_data,
        target_data_batches,
        ot_hyperparams,
        device=device,
        sparse_return=sparse_return,
        no_match=no_match
    )
    
    rewards = np.zeros_like(pseudo_rewards)
    norm_scale = len(target_data_batches[0]) / len(traj_data) if norm_reward else 1.
    for i in range(len(traj_data)):
        rewards[i] = pseudo_rewards[i] * pseudo_reward_scale * norm_scale
        if traj_reward is not None:
            rewards[i] += traj_reward[i] * reward_scale

    return rewards, dis_sp


if __name__ == "__main__":
    K, T, B, N, M = 3, 10, 15, 2, 2
    cost_s = torch.rand(size=[K, T, N, N], device="cuda").float()
    cost_t = torch.rand(size=[K, B, M, M], device="cuda").float()
    cost_s[:, :T, :, :] = deepcopy(cost_s)
    trans = torch.rand(size=[K, T, B, N, M], device="cuda").float()
    mu = (1. / N) * torch.ones(size=[N, 1], device="cuda").float()
    nu = (1. / M) * torch.ones(size=[M, 1], device="cuda").float()

    cost_s = torch.as_tensor(cost_s, device="cuda")
    cost_t = torch.as_tensor(cost_t, device="cuda")
    trans = torch.as_tensor(trans, device="cuda")

    ot_hyperparams = {
        "ot_method": "proximal",
        "loss_type": "L2",
        "inner_iteration": 100,
        "outer_iteration": 1000,
        "iter_bound": 1e-3,
        "sk_bound": 1e-3
    }

    # node_cost_st(cost_s, cost_t)
    # cost = node_cost(cost_s, cost_t, trans, mu, mu)
    # sinkhorn_knopp_iteration(cost, trans0=trans)
    gromov_wasserstein_discrepancy(cost_s, cost_t, ot_hyperparams)
