import torch
from torch.distributions.utils import logits_to_probs
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Categorical, Normal, Independent
import numpy as np


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
SIG_MAX_PRED = 10
SIG_MIN_PRED = 0.05
EPS = 1e-6


# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


def weights_init_pred(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, 0)


class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_outputs):
        super(MLP, self).__init__()

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init)

    def forward(self, inputs, output_activation=None):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        if output_activation == 'relu':
            x = F.relu(x)
        elif output_activation == 'softmax':
            x = F.softmax(x)

        return x


class CNN(nn.Module):
    def __init__(self, height, width, hidden_dim, num_outputs):
        super(CNN, self).__init__()

        self.c1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2)
        self.p1 = nn.MaxPool2d(kernel_size=2)
        height = ((height-2)+1)//2
        width = ((width-2)+1)//2
        self.l1 = nn.Linear(height*width*3,hidden_dim)
        self.l2 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init)

    def forward(self, inputs, output_activation=None):
        x = F.relu(self.p1(self.c1(inputs)))
        x = F.relu(self.l1(x))
        x = self.l2(x)
        if output_activation == 'relu':
            x = F.relu(x)
        elif output_activation == 'softmax':
            x = F.softmax(x)

        return x


class SuccessFeatureNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim, num_W):
        super(SuccessFeatureNet, self).__init__()
        self.num_W = num_W

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init)

    def forward(self, inputs, normalize=True):
        x = F.relu(self.l1(inputs))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        
        # reshape to [bs, num_actions, num_W]
        bs = x.shape[0]
        x.reshape(bs, -1, num_W)

        return x


class RepresentationNet(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_outputs, normalize=True):
        super(RepresentationNet, self).__init__()
        self.normalize = normalize

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, num_outputs)

        self.apply(weights_init)

    def forward(self, inputs, output_activation=None):
        x = F.elu(self.l1(inputs))
        x = F.elu(self.l2(x))
        x = self.l3(x)
        
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)

        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Double Q architecture
        self.l10 = nn.Linear(num_inputs, hidden_dim)
        self.l20 = nn.Linear(hidden_dim, hidden_dim)
        self.l30 = nn.Linear(hidden_dim, num_actions)

        self.l11 = nn.Linear(num_inputs, hidden_dim)
        self.l21 = nn.Linear(hidden_dim, hidden_dim)
        self.l31 = nn.Linear(hidden_dim, num_actions)
        self.apply(weights_init)

    def forward(self, states):
        xu = states

        x1 = F.relu(self.l10(xu))
        x1 = F.relu(self.l20(x1))
        x1 = self.l30(x1)

        x2 = F.relu(self.l11(xu))
        x2 = F.relu(self.l21(x2))
        x2 = self.l31(x2)

        return x1, x2


class QNetworkCont(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetworkCont, self).__init__()

        # Double Q architecture
        self.l10 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.l20 = nn.Linear(hidden_dim, hidden_dim)
        self.l30 = nn.Linear(hidden_dim, 1)

        self.l11 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.l21 = nn.Linear(hidden_dim, hidden_dim)
        self.l31 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)

    def forward(self, states, action):
        xu = torch.cat([states, action], 1)

        x1 = F.relu(self.l10(xu))
        x1 = F.relu(self.l20(x1))
        x1 = self.l30(x1)

        x2 = F.relu(self.l11(xu))
        x2 = F.relu(self.l21(x2))
        x2 = self.l31(x2)

        return x1, x2


class VNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(VNetwork, self).__init__()

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init)

    def forward(self, states):
        x = states
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_dim, num_labels):
        super(Discriminator, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_labels)

        self.apply(weights_init)

    def forward(self, state, label=None):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        dist = Categorical(logits=x)
        l = dist.sample()
        log_p = dist.log_prob(l)
        if label is not None:
            loggt = dist.log_prob(label.squeeze())
        else:
            loggt = None
        return label, loggt, log_p


class Predictor(nn.Module):
    def __init__(self, num_inputs, num_modes, hidden_dim, num_outputs, output_std=1.0) -> None:
        super(Predictor, self).__init__()
        # Input: s, z
        # Output: delta s
        self.linear1 = nn.Linear(num_inputs + num_modes, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_outputs)
        if output_std is None:
            self.std = nn.Linear(hidden_dim, num_outputs)
        else:
            self.std = torch.FloatTensor([output_std])
        self.output_std = output_std
        self.output_bn = nn.BatchNorm1d(num_outputs, affine=False)
        self.apply(weights_init_pred)

    def forward(self, state, label):
        # Input: state, one-hot label
        # state = self.bn(state)
        x = torch.cat([state, label], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean(x)
        if self.output_std is None:
            std = self.std(x)
        else:
            std = self.std.repeat(mean.shape)
        return mean, std

    def sample(self, state, label):
        mean, std = self.forward(state, label)
        dist = Independent(Normal(mean, std), 1)
        pred = dist.rsample()
        log_prob = dist.log_prob(pred)
        return pred, log_prob, mean

    def evaluate(self, state, label, pred):
        mean, std = self.forward(state, label)
        pred_bn = self.output_bn(pred) * 1.0
        dist = Independent(Normal(mean, std), 1)
        log_prob = dist.log_prob(pred_bn)
        dist_entropy = dist.entropy()
        return log_prob, dist_entropy        

    def to(self, device):
        if self.output_std is not None:
            self.std = self.std.to(device)
        return super(Predictor, self).to(device)


class GMMPredictor(nn.Module):
    def __init__(self, num_inputs, num_modes, hidden_dim, num_outputs, output_std=1.0, num_components=4) -> None:
        super(GMMPredictor, self).__init__()
        # Input: s, z
        # Output: delta s
        self.linear1 = nn.Linear(num_inputs + num_modes, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, num_components)
        self.means = []
        self.stds = []
        for _ in range(num_components):
            self.means.append(nn.Linear(hidden_dim, num_outputs))
            if output_std is None:
                self.stds.append(nn.Linear(hidden_dim, num_outputs))
            else:
                self.stds.append(torch.FloatTensor([output_std]))
        self.output_std = output_std
        self.num_components = num_components
        self.output_bn = nn.BatchNorm1d(num_outputs)

    def forward(self, state, label):
        # Input: state, one-hot label
        # state = self.bn(state)
        x = torch.cat([state, label], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        means = []
        stds = []
        logits = self.logits(x)
        for i in range(self.num_components):
            means.append(self.means[i](x))
            if self.output_std is None:
                stds.append(self.stds[i](x))
            else:
                stds.append(self.stds[i].repeat(self.means[i].shape))
        return means, stds, logits

    def sample(self, state, label):
        means, stds, logits = self.forward(state, label)
        comp = Independent(Normal(means, stds), 1)
        mix = Categorical(logits=logits)
        # pred = dist.rsample()
        # log_prob = dist.log_prob(pred)
        # return pred, log_prob, mean

    def evaluate(self, state, label, pred):
        mean, std = self.forward(state, label)
        pred_bn = self.output_bn(pred)
        dist = Independent(Normal(mean, std), 1)
        log_prob = dist.log_prob(pred)
        dist_entropy = dist.entropy()
        return log_prob, dist_entropy        

    def to(self, device):
        if self.output_std is not None:
            self.std = self.std.to(device)
        return super(GMMPredictor, self).to(device)


class CategoricalPolicy(nn.Module):
    def __init__(self, num_inputs, action_dims, hidden_dim):
        super(CategoricalPolicy, self).__init__()

        self.l1 = nn.Linear(num_inputs, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dims)

        self.apply(weights_init)

    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = F.relu(self.l2(x))
        logits = self.logits(x)
        prob = F.softmax(logits, dim=-1)

        return logits, prob

    def sample(self, state):
        _, prob = self.forward(state)
        dist = Categorical(probs=prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        mode = dist.mean
        return action, log_prob, mode

    def evaluate(self, state, action):
        _, prob = self.forward(state)
        dist = Categorical(probs=prob)
        action_logprobs = dist.log_prob(action)
        action_logprobs = torch.sum(action_logprobs, dim=-1)
        dist_entropy = dist.entropy()   
        dist_entropy = torch.sum(dist_entropy, dim=-1)  
        return action_logprobs, dist_entropy      

    def to(self, device):
        return super(CategoricalPolicy, self).to(device)


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None, action_std=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)
        self.action_std = action_std # Fixed std

        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        if self.action_std is None:
            log_std = self.log_std_linear(x)
        else:
            log_std = torch.log(self.action_std).repeat(mean.shape)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + EPS)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def evaluate(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        action_logprobs = normal.log_prob(action)
        action_logprobs = torch.sum(action_logprobs, dim=-1)
        dist_entropy = normal.entropy()   
        dist_entropy = torch.sum(dist_entropy, dim=-1)
        return action_logprobs, dist_entropy    

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        if self.action_std is not None:
            self.action_std = self.action_std.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(DeterministicPolicy, self).to(device)
