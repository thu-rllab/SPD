import numpy as np
import torch
import time
from copy import deepcopy

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from .episode_runner import EpisodeRunner
from url_algo import REGISTRY as url_assigner_REGISTRY
from url_algo.buffer import Cache, ReplayMemory


class URLEvaluator(EpisodeRunner):

    def __init__(self, args, logger):
        super(URLEvaluator, self).__init__(args, logger)
        self.num_modes = args.num_modes

        self.mixed_buffer = ReplayMemory(args.cache_size)
        self.indie_buffer_empty = [Cache(args.cache_size) for _ in range(self.num_modes)]
        self.indie_buffer_dict = {}

        self.url_assigner_fn = url_assigner_REGISTRY[self.args.url_algo]
    
    def setup(self, scheme, groups, preprocess, macs):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.macs = macs
    
    def reset(self, mode_id=None):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        self.mode_id = mode_id
        self.pseudo = False

    def run(self, test_mode=False, mode_id=None):
        self.reset(mode_id)

        terminated = False
        episode_return = 0
        self.macs[self.mode_id].init_hidden(batch_size=self.batch_size)

        single_url_traj = []
        graph_traj = []
        control_traj_reward = []
        observations = self.env.get_obs()

        single_url_feature, active_agents = self.build_url_feature(observations)
        graph_feature, _ = self.build_graph(observations)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [observations]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.macs[self.mode_id].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            # Fix memory leak
            cpu_actions = actions.to("cpu").numpy()

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": cpu_actions,
                "ori_reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            single_url_traj.append(single_url_feature)
            graph_traj.append(graph_feature)
            control_traj_reward.append(reward)

            # assert the controlling agents are the same
            new_observations = self.env.get_obs()
            
            new_single_url_feature, new_active_agents = self.build_url_feature(new_observations)
            new_graph_feature, _ = self.build_graph(new_observations)
            
            # insert the url_feature into the cache.
            self.mixed_buffer.push((np.array([self.mode_id]), single_url_feature))
            if active_agents not in self.indie_buffer_dict.keys():
                self.indie_buffer_dict[active_agents] = deepcopy(self.indie_buffer_empty)
            self.indie_buffer_dict[active_agents][self.mode_id].push((graph_feature, ))

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            observations = new_observations
            single_url_feature = new_single_url_feature
            graph_feature = new_graph_feature
            active_agents = new_active_agents

        return self.batch

    def build_url_feature(self, observations):
        if self.args.env == "gfootball":
            assert self.env.n_agents == 2, "only support 2 agents now."
            url_feature_list = []
            for obs in observations:
                url_feature_list.append(obs[0:6])        # [ego_positions, relative_positions_1, relative_positions_2]
                url_feature_list.append(obs[12:16])      # [relative_opponent_positions_1, relative_opponent_positions_2]
                url_feature_list.append(obs[20:22])      # [relative_ball_x, relative_ball_y]

            obs = observations[0]
            url_feature_list.append(obs[22:23])             # [ball_z]

            url_feature_list.append(obs[6:12])           # [left_team_movements]
            url_feature_list.append(obs[16:20])          # [right_team_movements]
            url_feature_list.append(obs[23:])            # other infos
            
            url_feature = np.concatenate(url_feature_list, axis=0)
            active_agents = tuple(url_feature[-3:])
        elif self.args.env == "mpe":
            url_feature_list = []
            for obs in observations:
                url_feature_list.append(obs[2:4])       # [ego_positions]
                if self.args.url_velocity:
                    url_feature_list.append(obs[0:2])   # [ego_velocity]

            url_feature = np.concatenate(url_feature_list, axis=0)
            active_agents = 1
        else:
            raise NotImplementedError

        return url_feature, active_agents
    
    def build_graph(self, observations):
        if self.args.env == "gfootball":
            # assert self.env.n_agents == 3, "only support 3 agents now."
            agents_pos_x, agents_pos_y = [], []
            for obs in observations:
                agents_pos_x.append(obs[0])
                agents_pos_y.append(obs[1])
            
            obs = observations[0]
            if self.args.opponent_graph:
                agents_pos_x.append(obs[12])
                agents_pos_y.append(obs[13])
            
            if self.args.ball_graph:
                agents_pos_x.append(obs[20])
                agents_pos_y.append(obs[21])

            active_agents = tuple(obs[-3:])
        elif self.args.env == "mpe":
            agents_pos_x, agents_pos_y = [], []
            for obs in observations:
                agents_pos_x.append(obs[2])
                agents_pos_y.append(obs[3])
            active_agents = 1
        elif self.args.env == 'sc2':
            if self.args.env_args['map_name'] == 'corridor':
                agents_pos_x, agents_pos_y, health = [], [], []
                
                active_agents=1
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        agents_pos_x = torch.as_tensor(agents_pos_x).reshape(-1, 1)
        agents_pos_y = torch.as_tensor(agents_pos_y).reshape(-1, 1)

        relative_pos_x = agents_pos_x - agents_pos_x.T
        relative_pos_y = agents_pos_y - agents_pos_y.T

        url_graph = torch.sqrt(relative_pos_x ** 2 + relative_pos_y ** 2)

        return url_graph, active_agents
