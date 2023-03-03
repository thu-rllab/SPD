import gym
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

from .. import MultiAgentEnv
from .scenarios import simple_tag_new

class SimpleTag(MultiAgentEnv):
    """Only the adversaries are controlled."""
    def __init__(
        self,
        n_agents=4,
        time_limit=100,
        time_step=0,
        map_name='simple_tag',
        seed=0,
        logdir=None,
        url_downstream=True,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.env_name = map_name
        self.seed_i = seed
        self.logdir = logdir
        self.url_downstream = url_downstream                # if url_downstream, the 'good_agent' is random walking
        self.env = simple_tag_new.parallel_env(
            num_good=1,
            num_adversaries=3,
            num_obstacles=1,
            max_cycles=self.episode_limit,
            continuous_actions=False
        )
        self.env.seed(self.seed_i)

        # self.action_space = [self.env.action_space('adversary_' + str(i)) for i in range(self.n_agents)]
        # self.observation_space = [self.env.observation_space('adversary_' + str(i)) for i in range(self.n_agents)]
        self.agents = self.env.possible_agents[:self.n_agents]
        self.action_space = [self.env.action_space(agent) for agent in self.agents]
        self.observation_space = [self.env.observation_space(agent) for agent in self.agents]

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim = self.observation_space[0].shape[0]
        self.obs_dict = None

        self.replay_fig = None

    def get_global_state(self):
        return self.env.state()

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        actions_list = actions.to('cpu').numpy().tolist()
        if self.url_downstream:
            actions_dict = {agent: actions_list[i] for i, agent in enumerate(self.agents)}
            actions_dict['agent_0'] = self.env.action_space('agent_0').sample()
            self.obs_dict, original_rewards, dones, infos = self.env.step(actions_dict)
        else:
            self.obs_dict, original_rewards, dones, infos = self.env.step({agent: actions_list[i] for i, agent in enumerate(self.agents)})

        # only done when reach the episode_limit
        if self.time_step >= self.episode_limit:
            done = True
        else:
            done = False

        if original_rewards['adversary_0'] != original_rewards['adversary_1'] or original_rewards['adversary_0'] != original_rewards['adversary_2'] or original_rewards['adversary_1'] != original_rewards['adversary_2']:
            print(1)

        return original_rewards['adversary_0'], done, {}

    def get_obs(self):
        """Returns all agent observations in a list."""
        return list(self.obs_dict.values())[:self.n_agents]

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.obs_dict[self.agents[agent_id]]

    def get_obs_good_agent(self):
        """Returns observation for the good agent."""
        return self.obs_dict['agent_0']

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.env.state_space.shape[0]

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]
    
    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.episode_array = None
        self.obs_dict = self.env.reset()

        return self.get_obs(), self.get_global_state()

    def render(self):
        pass

    def close(self):
        plt.close()
        self.env.close()

    def seed(self, seed):
        self.seed_i = seed
        self.env.seed(self.seed_i)

    def save_replay(self, step, replay_save_path, episode_i):
        """Save a replay."""
        if self.replay_fig is None:
            self.replay_fig = plt.figure(dpi=100)
        
        if self.episode_array is None:
            self.episode_array = []

        ax = self.replay_fig.add_axes([0.1, 0.1, 0.8, 0.8])
        rgb_array = self.env.render(mode="rgb_array").transpose([1, 0, 2])
        self.episode_array.append(rgb_array)
        if step + 1 >= self.episode_limit:
            np.save(osp.join(replay_save_path, f"episode_{episode_i}_replay.npy"), np.stack(self.episode_array, axis=0))

        ax.imshow(rgb_array)
        self.replay_fig.suptitle(f"step_{step}/{self.episode_limit}", fontsize=15, y=0.05)
        self.replay_fig.savefig(osp.join(replay_save_path, f"episode_{episode_i}-step_{step}.jpg"))

    def get_stats(self):
        return  {}


if __name__ == "__main__":
    env = SimpleTag()
    env.seed(1234342)
    obs, state = env.reset()
    print('state_size: ', env.get_state_size())
    print('obs_size: ', env.get_obs_size())
    print('avail_actions: ', env.get_avail_actions())
    print('avail_actions for good_agent: ', env.get_avail_agent_actions(3))
    print('obs: ', env.get_obs())
    print('obs for adversary_1: ', env.get_obs_agent(1))
    print('obs for good_agent: ', env.get_obs_agent(3))
    print('step returns: ', env.step(torch.as_tensor([0, 1, 1, 0])))

