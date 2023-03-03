import numpy as np
import torch
import time
from copy import deepcopy

from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from .episode_runner import EpisodeRunner
# from wurl.apwd import assign_reward
from url_algo import REGISTRY as url_assigner_REGISTRY
from url_algo.buffer import Cache

assign_reward = url_assigner_REGISTRY["gwd"]


class URLLoadRunner(EpisodeRunner):

    def __init__(self, args, logger):
        super(URLLoadRunner, self).__init__(args, logger)
        self.num_modes = args.num_modes

        self.caches_empty = [Cache(args.cache_size) for _ in range(self.num_modes)]
        self.caches_dict = {}
        self.train_returns = [[] for _ in range(args.num_modes)]
        self.test_returns = [[] for _ in range(args.num_modes)]
        self.cur_reward = [0 for _ in range(args.num_modes)]
        self.model_choosed = False
    
    def setup(self, scheme, groups, preprocess, macs):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.macs = macs
    
    def create_env(self, env_args):
        del self.env
        self.env = env_REGISTRY[self.args.env](**env_args)
    
    def reset(self, mode_id=None):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0
        # self.mode_id = np.random.randint(0, high=self.num_modes)
        self.mode_id = mode_id
        

    def run(self, test_mode=False, mode_id=None):
        self.reset(mode_id)

        terminated = False
        episode_return = 0
        self.macs[self.mode_id].init_hidden(batch_size=self.batch_size)

        observations = self.env.get_obs()
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
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            # assert the controlling agents are the same
            new_observations = self.env.get_obs()


            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            observations = new_observations

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
        cur_returns = self.test_returns[mode_id] if test_mode else self.train_returns[mode_id]
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode:
            if not self.model_choosed: #should test all
                if (sum(len(tr) for tr in self.test_returns) == self.args.test_nepisode * self.num_modes):
                    for i in range(self.num_modes):
                        r_skill = np.mean(self.test_returns[i])
                        self.cur_reward[i] = self.cur_reward[i]*self.args.reward_alpha + r_skill*(1-self.args.reward_alpha)
                    self._log(self.test_returns, cur_stats, log_prefix)
                    
            else: #should test one
                if len(self.test_returns[mode_id]) == self.args.test_nepisode:
                    self._log(self.test_returns, cur_stats, log_prefix)
        elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(self.train_returns, cur_stats, log_prefix)
            if hasattr(self.macs[self.mode_id].action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.macs[self.mode_id].action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env
        return self.batch
    
    def _log(self, all_returns, stats, prefix):
        for i in range(self.num_modes):
            if len(all_returns[i]) > 0:
                self.logger.log_stat(prefix + "return_mean_" + str(i), np.mean(all_returns[i]), self.t_env)
                self.logger.log_stat(prefix + "return_std_" + str(i), np.std(all_returns[i]), self.t_env)
                all_returns[i].clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
