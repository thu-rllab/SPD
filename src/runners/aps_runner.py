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
from url_algo.utils import sample_spherical


class URLRunner(EpisodeRunner):

    def __init__(self, args, logger):
        super(URLRunner, self).__init__(args, logger)
        
        self.url_train_returns = []
        self.url_test_returns = []

        self.url_assigner_fn = url_assigner_REGISTRY[self.args.url_algo]
        self.runnner_algo = args.runner_algo
    
    def setup(self, scheme, groups, preprocess, mac, phi_net=None, success_feature_net=None):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.phi_net = phi_net
        self.success_feature_net = success_feature_net
    
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
        self.reset()

        if replay_save_path is not None:
            self.env.save_replay(step=self.t, replay_save_path=replay_save_path, episode_i=episode_i)

        terminated = False
        episode_return = 0
        episode_pseudo_return = 0
        # self.mac[self.mode_id].init_hidden(batch_size=self.batch_size)

        state = self.env.get_state()
        observations = self.env.get_obs()
        aps_feature = np.concatenate([obs for obs in observations], axis=0)

        while not terminated:
            if self.t % args.aps_n_step == 0:
                task_vector_w = sample_spherical(args.dim_w)

            pre_transition_data = {
                "state": [state],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [observations],
                "task_vector_w": [task_vector_w]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac[self.mode_id].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
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
        actions = self.mac[self.mode_id].select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
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
            if hasattr(self.mac[self.mode_id].action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac[self.mode_id].action_selector.epsilon, self.t_env)
            if self.pseudo:
                self.logger.log_stat("d_sp", self.d_sp, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch


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
                success_feature_net=self.success_feature_net,
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
