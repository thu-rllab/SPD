import os
import sys
import gym
import torch as th
import numpy as np
import gfootball.env as football_env
from gfootball.env import observation_preprocessing

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from multiagentenv import MultiAgentEnv


class AcademyPassAndShootWithKeeper(MultiAgentEnv):
    
    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=0,
        render=False,
        n_agents=2,
        time_limit=150,
        time_step=0,
        env_ball_owner=False,
        map_name='academy_pass_and_shoot_with_keeper',
        stacked=False,
        representation="simple115v2",
        rewards='scoring',
        logdir='football_dumps',
        write_video=False,
        number_of_right_players_agent_controls=0,
        seed=0,
    ):
        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        self.ball_owner = env_ball_owner
        self.env_name = map_name
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed

        self.env = football_env.create_environment(
            write_full_episode_dumps=self.write_full_episode_dumps,
            write_goal_dumps=self.write_goal_dumps,
            env_name=self.env_name,
            stacked=self.stacked,
            representation=self.representation,
            rewards=self.rewards,
            logdir=self.logdir,
            render=self.render,
            write_video=self.write_video,
            dump_frequency=self.dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=self.number_of_right_players_agent_controls,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT)
        )
        self.env.seed(self.seed)

        self.obs_dim = 37 if self.ball_owner else 32

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(
                low=np.array([-float("inf")] * self.obs_dim),
                high=np.array([float("inf")] * self.obs_dim),
                shape=(self.obs_dim, )
            ) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n
        self.unit_dim = self.obs_dim

        self.avail_actions = [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

    def step(self, actions):
        """Returns reward, terminated, info."""
        if th.is_tensor(actions):
            actions = actions.cpu().numpy()

        self.time_step += 1
        _, rewards, done, infos = self.env.step(actions.tolist())

        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True

        return sum(rewards), done, infos
        # if sum(rewards) <= 0:
        #     return -int(done), done, infos

        # # return obs, self.get_global_state(), 100, done, infos
        # return 100, done, infos

    # def get_simple_obs(self, index=-1):
    #     full_obs = self.env.unwrapped.observation()[0]
    #     simple_obs = []

    #     if index == -1:
    #         # global state, absolute position
    #         simple_obs.append(full_obs['left_team']
    #                             [-self.n_agents:].reshape(-1))
    #         simple_obs.append(
    #             full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

    #         simple_obs.append(full_obs['right_team'].reshape(-1))
    #         simple_obs.append(full_obs['right_team_direction'].reshape(-1))

    #         simple_obs.append(full_obs['ball'])
    #         simple_obs.append(full_obs['ball_direction'])

    #     else:
    #         # local state, relative position
    #         ego_position = full_obs['left_team'][-self.n_agents +
    #                                                 index].reshape(-1)
    #         simple_obs.append(ego_position)
    #         simple_obs.append((np.delete(
    #             full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

    #         simple_obs.append(
    #             full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
    #         simple_obs.append(np.delete(
    #             full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

    #         simple_obs.append(
    #             (full_obs['right_team'] - ego_position).reshape(-1))
    #         simple_obs.append(full_obs['right_team_direction'].reshape(-1))

    #         simple_obs.append(full_obs['ball'][:2] - ego_position)
    #         simple_obs.append(full_obs['ball'][-1].reshape(-1))
    #         simple_obs.append(full_obs['ball_direction'])

    #     simple_obs = np.concatenate(simple_obs)
    #     return simple_obs

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()
        assert len(full_obs) == self.n_agents == 2, 'Now only support 2 agents subgroup.'

        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            return obj.flatten()
        
        active = [0] * 3
        for obs in full_obs:
            if obs['active'] != -1:
                active[obs['active']] = 1

        simple_obs = []
        if index == -1:
            obs = full_obs[0]
            # global state, absolute position
            simple_obs.extend(do_flatten(obs['left_team']))
            simple_obs.extend(do_flatten(obs['left_team_direction']))

            simple_obs.extend(do_flatten(obs['right_team']))
            simple_obs.extend(do_flatten(obs['right_team_direction']))

            simple_obs.extend(do_flatten(obs['ball']))
        else:
            obs = full_obs[index]
            if obs['active'] != -1:
                active_id = obs['active']

                ego_position = obs['left_team'][active_id]
                simple_obs.extend(do_flatten(ego_position))
                simple_obs.extend(do_flatten(np.delete(obs['left_team'], active_id, axis=0) - ego_position))
                simple_obs.extend(do_flatten(obs['left_team_direction']))
                # relative velocity.
                # ego_movement = obs['left_team_direction'][active_id]
                # simple_obs.extend(do_flatten(ego_movement))
                # simple_obs.extend(do_flatten(np.delete(obs['left_team_direction'], active_id, axis=0) - ego_movement))

                simple_obs.extend(do_flatten(obs['right_team'] - ego_position))
                simple_obs.extend(do_flatten(obs['right_team_direction']))

                simple_obs.extend(do_flatten(obs['ball'][:2] - ego_position))
                simple_obs.extend(do_flatten(obs['ball'][-1]))
            else:
                simple_obs.extend([-1.] * 23)

        simple_obs.extend(do_flatten(obs['ball_direction']))

        ball_owned_player = [0] * (self.n_agents + 3)
        if obs['ball_owned_team'] == -1:
            simple_obs.extend([1, 0, 0])
        if obs['ball_owned_team'] == 0:
            simple_obs.extend([0, 1, 0])
            ball_owned_player[obs['ball_owned_player']] = 1
        if obs['ball_owned_team'] == 1:
            simple_obs.extend([0, 0, 1])
            ball_owned_player[1 + self.n_agents + obs['ball_owned_player']] = 1
        if self.ball_owner:
            simple_obs.extend(ball_owned_player)

        simple_obs.extend(active)

        return np.array(simple_obs, dtype=np.float32)
    

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def get_obs(self):
        """Returns all agent observations in a list."""
        # obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])
        obs = [self.get_simple_obs(i) for i in range(self.n_agents)]
        return obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.get_simple_obs(agent_id)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        return self.get_global_state()

    def get_state_size(self):
        """Returns the size of the global state."""
        # TODO: in wrapper_grf_3vs1.py, author set state_shape=obs_shape
        return self.obs_dim

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        return self.avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.action_space[0].n

    def reset(self):
        """Returns initial observations and states."""
        self.time_step = 0
        self.env.reset()
        obs = np.array([self.get_simple_obs(i) for i in range(self.n_agents)])

        return obs, self.get_global_state()

    def render(self):
        pass

    def close(self):
        self.env.close()

    def seed(self):
        pass

    def save_replay(self):
        """Save a replay."""
        pass


if __name__ == "__main__":
    env = AcademyPassAndShootWithKeeper(env_ball_owner=False)
    env.reset()
    print(env.get_state().shape[0] - env.get_state_size())
    print(env.get_obs_agent(0).shape)
    # print(env.get_obs_agent(0))
    # print(env.get_obs_agent(0).shape)

