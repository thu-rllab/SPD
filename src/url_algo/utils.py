import math
import torch
import numpy as np
import math
from copy import deepcopy

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def atan(x,y):
    if abs(x)<0.0001 and abs(y)<0.0001:
        return 0.0
    if abs(x)<0.0001:
        if y>0:
            return math.pi/2
        else:
            return math.pi*3/2
    if x>0:
        if y>=0:
            return math.atan(y/x)
        else:
            return math.atan(y/x)+2*math.pi
    else:
        return math.atan(y/x)+math.pi
        
def linespace_by_rad(rad, lmax):
    lmax=abs(lmax)
    z=np.linspace(0,lmax,100)
    x=math.cos(rad)*z
    y=math.sin(rad)*z
    return x,y

def wrapped_obs(obs, label, n):
    label_onehot = np.zeros((label.size, n))
    label_onehot[np.arange(label.size), label] = 1.
    if len(obs.shape) == 1:
        obs = np.expand_dims(obs, axis=0)
    wo = np.concatenate([label_onehot, obs], -1)
    return wo

def convert_to_onehot(label, n):
    if len(label.shape) > 1:
        label = label[:,0]
    label_onehot = np.zeros((label.size, n))
    label_onehot[np.arange(label.size), label] = 1.
    return label_onehot

def sample_spherical(dim_w=5):
    """
    returns:
        vec: [1, dim_w]
    """
    vec = np.random.normal(loc=0, scale=1, size=(1, dim_w))
    vec /= np.linalg.norm(vec, axis=0)
    return vec
