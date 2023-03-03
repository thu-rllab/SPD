from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from .basic_controller import BasicMAC
import torch as th
import torch.nn.functional as F
from utils.rl_utils import RunningMeanStd
import numpy as np

# This multi-agent controller shares parameters between agents
class RegMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        super(RegMAC, self).__init__(scheme, groups, args)
        self.action_dim = scheme['avail_actions']['vshape'][0]
    
    def _build_inputs(self, batch, t, na, alist):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        #alist: [bs,na]
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t, na]) 
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t, na]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1, na])
        if self.args.obs_agent_id:
            # inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            inputs.append(F.one_hot(th.tensor(na, device=batch.device), self.n_agents).unsqueeze(0).expand(bs, -1))
        if self.args.onehot_action:
            vector = th.zeros([bs, self.n_agents, self.action_dim], device=batch.device)
            vector[:,:na] = F.one_hot(alist[:,:na].long(), self.action_dim)
            inputs.append(vector.view(bs, -1))
        else:
            raise NotImplementedError
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=-1)
        return inputs

    def forward(self, ep_batch, t, t_env, train_mode=False, test_mode=False):
        if test_mode:
            self.agent.eval()
        bs = ep_batch.batch_size
        alist = th.zeros([bs, self.n_agents], device = ep_batch.device)
        agent_outs_list = []
        chosen_action_list = []
        for i in range(self.n_agents):
            agent_inputs = self._build_inputs(ep_batch, t, i, alist) #bs,-1
            agent_outs, self.hidden_states[:,i:i+1] = self.agent(agent_inputs.unsqueeze(1), self.hidden_states[:,i:i+1]) #bs,1,action_dim | bs,1,-1
            agent_outs_list.append(agent_outs)
            if not self.args.sample_on_train and train_mode:
                alist[:,i] = ep_batch['actions'][:,t,i,0]
            else:
                chosen_action = self.action_selector.select_action(agent_outs, ep_batch["avail_actions"][:, t, i:i+1], t_env, test_mode=test_mode) #bs, n_agents
                chosen_action = chosen_action.squeeze(1)
                alist[:,i] = chosen_action
                chosen_action_list.append(chosen_action)
        agent_outs = th.stack(agent_outs_list, dim=1)
        if train_mode:
            return agent_outs
        else:
            chosen_actions = th.stack(chosen_action_list, dim=1) #bs,na
            return agent_outs, chosen_actions
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        _, chosen_actions = self.forward(ep_batch, t_ep, t_env, test_mode=test_mode) #bs,na,dim
        return chosen_actions[bs]
    
    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        if self.args.onehot_action:
            input_shape += scheme['avail_actions']['vshape'][0] * self.n_agents
        else:
            input_shape += self.n_agents
        return input_shape
    
    def save_models(self, path, mode_id):
        th.save(self.agent.state_dict(), "{}/agent_{}.th".format(path, mode_id))

    def load_models(self, path, mode_id):
        self.agent.load_state_dict(th.load("{}/agent_{}.th".format(path, mode_id), map_location=lambda storage, loc: storage))