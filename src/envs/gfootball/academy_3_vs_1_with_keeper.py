from .. import MultiAgentEnv
import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np

class Academy_3_vs_1_with_Keeper(MultiAgentEnv):

    def __init__(
        self,
        dense_reward=False,
        write_full_episode_dumps=False,
        write_goal_dumps=False,
        dump_freq=1000,
        render=False,
        n_agents=3,
        time_limit=150,
        time_step=0,
        obs_dim=37,
        map_name='academy_3_vs_1_with_keeper',
        env_ball_owner=False,
        stacked=False,
        representation="simple115",
        rewards='scoring',
        logdir='football_dumps',
        write_video=True,
        number_of_right_players_agent_controls=0,
        seed=0,
        no_reward=False,
        map_style = 0
    ):
        self.map_style = map_style

        if self.map_style == 0: #original map, half-court init fixed.
            scenario_cfg = None
        elif self.map_style == 1: #half-court init random
            scenario_cfg = {'random_init':True, 'random_scale':0.05}
        elif self.map_style == 2: #half-court init large random
            scenario_cfg = {'random_init':True, 'random_scale':0.2}
        elif self.map_style == 3: #half-court init random, possesion change not end
            scenario_cfg = {'random_init':True, 'random_scale':0.05, 'pc_ok':True}
        elif self.map_style == 4: #full-court init fixed
            scenario_cfg = {'full_court':True}
        elif self.map_style == 5: #full-court init random
            scenario_cfg = {'full_court':True, 'random_init': True, 'random_scale':0.05}
        elif self.map_style == 6: #full-court init random, possesion change not end
            scenario_cfg = {'full_court':True, 'random_init': True, 'random_scale':0.05, 'pc_ok':True}
        elif self.map_style == 7: #original map, reward scale 0.1
            scenario_cfg = None
        # elif self.map_style == 6: #full-court init at right

        self.dense_reward = dense_reward
        self.write_full_episode_dumps = write_full_episode_dumps
        self.write_goal_dumps = write_goal_dumps
        self.dump_freq = dump_freq
        self.render = render
        self.n_agents = n_agents
        self.episode_limit = time_limit
        self.time_step = time_step
        # self.obs_dim = obs_dim
        self.env_name = map_name
        self.ball_owner = env_ball_owner
        self.stacked = stacked
        self.representation = representation
        self.rewards = rewards
        self.logdir = logdir
        self.write_video = write_video
        self.number_of_right_players_agent_controls = number_of_right_players_agent_controls
        self.seed = seed
        self.no_reward = no_reward
        self.obs_dim = obs_dim
        self.env = football_env.create_environment(
            write_full_episode_dumps = self.write_full_episode_dumps,
            write_goal_dumps = self.write_goal_dumps,
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
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT),
            scenario_cfg=scenario_cfg,
            other_config_options={})
        self.env.seed(self.seed)

        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

        self.n_actions = self.action_space[0].n

        self.unit_dim = self.obs_dim  # QPLEX unit_dim  for cds_gfootball
        # self.unit_dim = 6  # QPLEX unit_dim set as that in Starcraft II

    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()
        assert len(full_obs) == self.n_agents == 3, 'Now only support 2 agents subgroup.'

        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            return obj.flatten()
        
        active = [0] * 4
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
                simple_obs.extend([-1.] * 27)

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
    '''
    def get_simple_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []

        if index == -1:
            # global state, absolute position
            simple_obs.append(full_obs['left_team']
                              [-self.n_agents:].reshape(-1))
            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents:].reshape(-1))

            simple_obs.append(full_obs['right_team'].reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'])
            simple_obs.append(full_obs['ball_direction'])

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            simple_obs.append(
                (full_obs['right_team'] - ego_position).reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            simple_obs.append(full_obs['ball'][:2] - ego_position)
            simple_obs.append(full_obs['ball'][-1].reshape(-1))
            simple_obs.append(full_obs['ball_direction'])

        simple_obs = np.concatenate(simple_obs)
        return simple_obs
        '''

    def get_global_state(self):
        return self.get_simple_obs(-1)

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if self.map_style in [0,1,2,3,7] and (ball_loc[0] < 0 or any(ours_loc[:, 0] < 0)):
            return True

        return False

    def step(self, actions):
        """Returns reward, terminated, info."""
        self.time_step += 1
        _, original_rewards, done, infos = self.env.step(actions.to('cpu').numpy().tolist())
        rewards = list(original_rewards)
        # obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

        if self.time_step >= self.episode_limit:
            done = True

        if self.check_if_done():
            done = True
        if self.no_reward:
            return 0, done, infos
        if sum(rewards) <= 0:
            # return obs, self.get_global_state(), -int(done), done, infos
            return -int(done), done, infos
        # return obs, self.get_global_state(), 100, done, infos
        win_r = 100
        if self.map_style == 7:
            win_r *= 0.1
        return win_r, done, infos

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
        return [[1 for _ in range(self.n_actions)] for agent_id in range(self.n_agents)]

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

