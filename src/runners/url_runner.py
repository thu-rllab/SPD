import numpy as np
import torch
import time
from copy import deepcopy

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from .episode_runner import EpisodeRunner
from url_algo import REGISTRY as url_assigner_REGISTRY
from url_algo.buffer import Cache


class URLRunner(EpisodeRunner):

    def __init__(self, args, logger):
        super(URLRunner, self).__init__(args, logger)
        self.num_modes = args.num_modes

        if self.args.url_algo == "diayn":
            self.cache = Cache(args.cache_size)
        else:
            self.caches_empty = [Cache(args.cache_size) for _ in range(self.num_modes)]
            self.caches_dict = {}
        
        self.url_train_returns = []
        self.url_test_returns = []

        self.url_assigner_fn = url_assigner_REGISTRY[self.args.url_algo]
    
    def setup(self, scheme, groups, preprocess, macs, disc_trainer=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.macs = macs
        self.disc_trainer = disc_trainer
    
    def create_env(self, env_args):
        del self.env
        self.env = env_REGISTRY[self.args.env](**env_args)
    
    def reset(self, mode_id=None):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        # self.mode_id = np.random.randint(0, high=self.num_modes)
        self.mode_id = mode_id
        self.pseudo = False
        self.d_sp = 0.

    def run(self, test_mode=False, mode_id=None, replay_save_path=None, episode_i=0):
        self.reset(mode_id)

        if replay_save_path is not None:
            self.env.save_replay(step=self.t, replay_save_path=replay_save_path, episode_i=episode_i)

        terminated = False
        episode_return = 0
        episode_pseudo_return = 0
        self.macs[self.mode_id].init_hidden(batch_size=self.batch_size)

        control_traj = []
        control_traj_reward = []
        state = self.env.get_state()
        observations = self.env.get_obs()
        url_feature, active_agents = self.build_graph_or_feature(observations, state)

        while not terminated:

            pre_transition_data = {
                "state": [state],
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

            control_traj.append(url_feature)
            control_traj_reward.append(reward)

            # assert the controlling agents are the same
            new_state = self.env.get_state()
            new_observations = self.env.get_obs()
            new_url_feature, new_active_agents = self.build_graph_or_feature(new_observations, new_state)
                
            if new_active_agents == active_agents and len(control_traj) < self.args.max_control_len:
                controller_updated = False
            else:
                controller_updated = True
            
            # when the control traj ended, calculate the pseudo rewards.
            if terminated or controller_updated:
                if self.t_env >= self.args.start_steps:
                    pseudo_rewards, d_sp = self.calc_pseudo_rewards(active_agents, control_traj, control_traj_reward)
                    if pseudo_rewards is not None:
                        self.d_sp = d_sp
                        self.pseudo = True
                        pseudo_rewards_data = {
                            "reward": pseudo_rewards,
                        }
                        self.batch.update(pseudo_rewards_data, ts=slice((self.t - len(pseudo_rewards)) + 1, self.t + 1))
                        episode_pseudo_return += np.sum(pseudo_rewards)
                control_traj = []
            
            # insert the url_feature into the cache.
            if self.args.url_algo == "diayn":
                self.cache.push((np.array([self.mode_id]), url_feature))
            else:
                if active_agents not in self.caches_dict.keys():
                    self.caches_dict[active_agents] = deepcopy(self.caches_empty)
                self.caches_dict[active_agents][self.mode_id].push((url_feature, ))

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            state=new_state
            observations = new_observations
            url_feature = new_url_feature
            active_agents = new_active_agents

            if replay_save_path is not None:
                self.env.save_replay(step=self.t, replay_save_path=replay_save_path, episode_i=episode_i)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.macs[self.mode_id].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # Fix memory leak
        cpu_actions = actions.to("cpu").numpy()
        self.batch.update({"actions": cpu_actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        env_returns = self.test_returns if test_mode else self.train_returns
        url_returns = self.url_test_returns if test_mode else self.url_train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) for k in set(cur_stats)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        env_returns.append(episode_return)
        url_returns.append(episode_pseudo_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(env_returns, cur_stats, log_prefix + 'env_')
            self._log(url_returns, cur_stats, log_prefix + 'url_')
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(env_returns, cur_stats, log_prefix + 'env_')
            self._log(url_returns, cur_stats, log_prefix + 'url_')
            if hasattr(self.macs[self.mode_id].action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.macs[self.mode_id].action_selector.epsilon, self.t_env)
            if self.pseudo:
                self.logger.log_stat("d_sp", self.d_sp, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch

    def build_graph_or_feature(self, observations, state):
        if self.args.url_algo == "gwd":
            if self.args.env == "gfootball" or self.args.env == "mpe":
                url_feature, active_agents = self.build_graph_by_obs(observations)
            elif self.args.env == "sc2":
                url_feature, active_agents = self.build_graph_by_state(state)
            else:
                raise NotImplementedError
        else:
            url_feature, active_agents = self.build_url_feature(observations)
        return url_feature, active_agents

    def build_url_feature(self, observations):
        if self.args.env == "gfootball":
                       # assert self.env.n_agents == 2, "only support 2 agents now."
            # url_feature_list = []
            # for obs in observations:
            #     url_feature_list.append(obs[0:6])        # [ego_positions, relative_positions_1, relative_positions_2]
            #     url_feature_list.append(obs[12:16])      # [relative_opponent_positions_1, relative_opponent_positions_2]
            #     url_feature_list.append(obs[20:22])      # [relative_ball_x, relative_ball_y]

            # obs = observations[0]
            # url_feature_list.append(obs[22:23])             # [ball_z]

            # url_feature_list.append(obs[6:12])           # [left_team_movements]
            # url_feature_list.append(obs[16:20])          # [right_team_movements]
            # url_feature_list.append(obs[23:])            # other infos
            
            # url_feature = np.concatenate(url_feature_list, axis=0)
            # active_agents = tuple(url_feature[-3:])
            assert self.env.n_agents == 3, "only support academy_3_vs_1_with_keeper now."
            url_feature_list = []
            for obs in observations:
                url_feature_list.append(obs[0:2])           # [ego_positions]
            
            obs = observations[0]
            # abs_pos = relative_pos + ego_pos
            if self.args.opponent_graph:
                url_feature_list.append([obs[16] + obs[0], obs[17] + obs[1]])       # [opponent_position]
            if self.args.ball_graph:
                url_feature_list.append([obs[24] + obs[0], obs[25] + obs[1]])       # [ball_position], only x, y direction.

            url_feature = np.concatenate(url_feature_list, axis=0)
            active_agents = tuple(url_feature[-3:])
        elif self.args.env == "mpe":
            url_feature_list = []
            for obs in observations:
                url_feature_list.append(obs[2:4])       # [ego_positions]
                if self.args.url_velocity:
                    url_feature_list.append(obs[0:2])   # [ego_velocity]
            # observe the pos of the good_agent
            if self.args.env_args['url_downstream']:
                obs = self.env.get_obs_good_agent()
                url_feature_list.append(obs[2:4])

            url_feature = np.concatenate(url_feature_list, axis=0)
            active_agents = 1
        else:
            raise NotImplementedError

        return url_feature, active_agents
    
    def build_graph_by_obs(self, observations):
        if self.args.env == "gfootball":
            assert self.env.n_agents == 3, "only support academy_3_vs_1_with_keeper now."
            agents_pos_x, agents_pos_y = [], []
            for obs in observations:
                agents_pos_x.append(obs[0])
                agents_pos_y.append(obs[1])

            obs = observations[0]
            # abs_pos = relative_pos + ego_pos
            if self.args.opponent_graph:
                agents_pos_x.append(obs[16] + obs[0])
                agents_pos_y.append(obs[17] + obs[1])
            
            if self.args.ball_graph:
                agents_pos_x.append(obs[24] + obs[0])
                agents_pos_y.append(obs[25] + obs[1])

            active_agents = tuple(obs[-3:])
        elif self.args.env == "mpe":
            agents_pos_x, agents_pos_y = [], []
            for obs in observations:
                agents_pos_x.append(obs[2])
                agents_pos_y.append(obs[3])

            # observe the pos of the good_agent
            if self.args.env_args['url_downstream']:
                obs = self.env.get_obs_good_agent()
                agents_pos_x.append(obs[2])
                agents_pos_y.append(obs[3])

            active_agents = 1
        else:
            raise NotImplementedError

        agents_pos_x = torch.as_tensor(agents_pos_x).reshape(-1, 1)
        agents_pos_y = torch.as_tensor(agents_pos_y).reshape(-1, 1)

        relative_pos_x = agents_pos_x - agents_pos_x.T
        relative_pos_y = agents_pos_y - agents_pos_y.T

        url_graph = torch.sqrt(relative_pos_x ** 2 + relative_pos_y ** 2)

        return url_graph, active_agents
    
    def build_graph_by_state(self, state):
        assert self.args.env == 'sc2'
        active_agents=1
        nf_al = 4 + self.env.shield_bits_ally + self.env.unit_type_bits
        nf_en = 3 + self.env.shield_bits_enemy + self.env.unit_type_bits
        agent_num = self.env.n_agents
        enemy_num = self.env.n_enemies
        agent_state = state[:(agent_num*nf_al)].reshape(agent_num, nf_al)
        agent_feature = agent_state[:,(0,2,3)] #6*3        
        if self.args.opponent_graph:
            enemy_state = state[(agent_num*nf_al):(agent_num*nf_al + enemy_num*nf_en)].reshape(enemy_num,nf_en)
            enemy_feature = enemy_feature = enemy_state[:,(0,1,2)]
            agent_feature = np.vstack([agent_feature, enemy_feature])
        with torch.no_grad():
            if self.args.del_death:
                agent_feature=agent_feature[np.where(agent_feature[:,0]>0.001)]
            agent_feature = torch.as_tensor(agent_feature) #bs*3
            url_graph = torch.linalg.norm(agent_feature.unsqueeze(0)-agent_feature.unsqueeze(1), ord=2, dim=2)
        return url_graph, active_agents


    def calc_pseudo_rewards(self, active_agents, control_traj, control_traj_reward=None):
        try:
            target_data_batches = None
            if self.args.url_algo != "diayn":
                target_data_batches = []
                for i in range(self.num_modes):
                    if i == self.mode_id:
                        continue
                    target_data_batches.append(list(self.caches_dict[active_agents][i].dump(self.args.max_control_len))[0])
                    
            pseudo_rewards, d_sp = self.url_assigner_fn(
                traj_data=control_traj,
                target_data_batches=target_data_batches,
                ot_hyperparams=self.args.ot_hyperparams,
                mode_id=self.mode_id,
                disc_trainer=self.disc_trainer,
                num_modes=self.args.num_modes,
                pseudo_reward_scale=self.args.pseudo_reward_scale,
                reward_scale=self.args.reward_scale,
                norm_reward=self.args.norm_reward,
                traj_reward=control_traj_reward,
                device="cuda",
                use_batch_apwd=self.args.batch_apwd,
                sparse_return=self.args.sparse_return,
                no_match=self.args.no_match
            )
        except:
            return None, None

        return pseudo_rewards.reshape(-1, 1), d_sp
