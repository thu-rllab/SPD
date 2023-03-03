# Wasserstein distance based pseudo-reward estimation
import numpy as np
import torch as th
from torch.distributions.multivariate_normal import MultivariateNormal

def match(A, B, D ,T):
    count_A = [T for _ in range(D)]
    count_B = [D for _ in range(T)]
    distance = np.zeros(D)
    i = j = 0
    while i<D:
        if count_A[i] == 0:
            i += 1
        elif count_B[j] == 0:
            j += 1
        else:
            delta = min(count_A[i], count_B[j])
            count_A[i] -= delta
            count_B[j] -= delta
            distance[i] += np.linalg.norm(A[i] - B[j], ord=2, axis=-1)*delta/T
    return distance

def match_batch(A, B, D, T):
    '''
    A: skill*M*D*s_dim
    B: skill*M*T*s_dim
    '''
    # lcm = np.lcm(D, T)
    lcm = D*T
    rA = A.repeat_interleave(lcm//D, dim=2)
    rB = B.repeat_interleave(lcm//T, dim=2)
    distance = th.linalg.norm(rA-rB, ord=2, dim=3)/T #skill*M*lcm
    skill, M, _ = distance.shape
    distance = distance.reshape(skill, M, D, lcm // D).sum(-1)
    return distance #skill*M*D




def wsre(A, B):
    D, d = A.shape
    T = B.shape[0]
    M = 5
    # d: data dimension, D: number of source data points, T: number of benchmark data points, M: number of v
    mean = np.zeros(d)
    cov = np.eye(d)
    wd = np.zeros((M, D))
    v = np.random.multivariate_normal(mean, cov, M)
    l = 1./np.linalg.norm(v, ord=2, axis=-1)
    v = v * l[:, None]
    for i in range(M):
        pA = np.matmul(A, v[i])
        pB = np.matmul(B, v[i])
        iA = np.argsort(pA)
        iB = np.argsort(pB)
        A = A[iA]
        B = B[iB]
        m = match(A, B, D, T)
        wd[i, iA] = m
    return np.mean(wd, axis=0)

def wsre_batch(A, B, device='cuda'):
    '''
    A: bs_D*s_dim
    B: skill*bs_T*s_dim
    '''
    D, d = A.shape
    skill = len(B)
    T, _ = B[0].shape
    M=5
    with th.no_grad():
        mean = th.zeros([skill,M,d], device=device)
        cov = th.eye(d, device=device).reshape(1,1,d,d).expand(skill,M,-1,-1)
        A = th.tensor(A, device=device)
        B = th.tensor(B, device=device)
        v = MultivariateNormal(mean,cov).sample() #skill*M*d
        l = th.linalg.norm(v, ord=2, dim=-1)
        v = v/l.unsqueeze(-1)
        pA = (A.unsqueeze(0).unsqueeze(0)*v.unsqueeze(2)).sum(-1) #skill*M*bs_D
        pB = (B.unsqueeze(1)*v.unsqueeze(2)).sum(-1) #skill*M*bs_T
        iA = th.argsort(pA, dim=-1)
        iB = th.argsort(pB, dim=-1)
        A = A[iA] #skill*M*bs_D*s_sim
        B = th.gather(B.unsqueeze(1).expand(-1,M,-1,-1), 2, iB.unsqueeze(-1).expand(-1,-1,-1,d)) #skill*M*bs_T*s_dim
        m = match_batch(A, B, D, T) #skill*M*D
        m = th.gather(m, 2, iA)
        return m.mean(1).cpu().numpy() #skill*D
