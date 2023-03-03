import datetime
import os
import pprint
import time
import json
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
from tqdm import tqdm
from pyvirtualdisplay import Display
from sacred.serializer import flatten
import numpy as np

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    if args.env_args["render"]:
        disp = Display(backend="xvfb").start()
    if args.best_mode >= 0:
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True, mode_id=args.best_mode)
    else:
        for _ in range(args.test_nepisode):
            runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()
    if args.env_args["render"]:
        disp.stop()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    if getattr(args, 'agent_own_state_size', False):
        args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "ori_reward": {"vshape": (1,)},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }
    if not args.vis_process:
        return _train(args, logger, runner, env_info, scheme, groups, preprocess)
    else:
        return _visualize(args, logger, runner, env_info, scheme, groups, preprocess)

def _train(args, logger, runner, env_info, scheme, groups, preprocess):
    buffers = [ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                            preprocess=preprocess,
                            device="cpu" if args.buffer_cpu_only else args.device) for _ in range(args.num_modes)]
    # Setup multiagent controller here
    macs = [mac_REGISTRY[args.mac](buffers[0].scheme, groups, args) for _ in range(args.num_modes)]

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, macs=macs)

    # Learner
    learners = [le_REGISTRY[args.learner](macs[i], buffers[0].scheme, logger, args) for i in range(args.num_modes)]

    if args.use_cuda:
        for learner in learners:
            learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        for mode_id in range(args.num_modes):
            learners[mode_id].load_models(model_path, mode_id)
        # runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            logger.log_stat("episode", 0, runner.t_env)
            logger.print_recent_stats()
            return
    else:
        print('Must load pretrained model!')
        return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    cur_mode_id = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:
        if not runner.model_choosed:
            if runner.t_env >= args.judge_t and cur_mode_id == 0: #choose the best model to train
                runner.model_choosed = True
                print('Test rewards for all skills:',{i:runner.cur_reward[i] for i in range(runner.num_modes)})
                cur_mode_id = np.argmax(runner.cur_reward)
                macs[cur_mode_id].action_selector.step_in_second_phase()
                for i in range(args.num_modes):
                    if not i == cur_mode_id:
                        buffers[i] = None #release the memory
                        macs[i] = None
                logger.console_logger.info(\
                    "--------------------------------\n\n\nUsing skill {}!\n\n\n--------------------------------".format(cur_mode_id))
            else:
                cur_mode_id = (cur_mode_id + 1) % args.num_modes

        # Run for a whole episode at a time

        with th.no_grad():
            episode_batch = runner.run(test_mode=False, mode_id=cur_mode_id)
            buffers[cur_mode_id].insert_episode_batch(episode_batch)

        if buffers[cur_mode_id].can_sample(args.batch_size):
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffers[cur_mode_id].sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learners[cur_mode_id].train(episode_sample, runner.t_env, episode)
            del episode_sample

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            if runner.model_choosed:
                for _ in range(n_test_runs):
                    runner.run(test_mode=True, mode_id=cur_mode_id)
            else:
                for _ in range(n_test_runs):
                    for j in range(args.num_modes):
                        runner.run(test_mode=True, mode_id=j)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            if not os.path.exists(os.path.join(args.local_results_path, "models", args.unique_token, "config.json")):
                with open(os.path.join(args.local_results_path, "models", args.unique_token, "config.json"), "w") as f:
                    json.dump(flatten(vars(args)), f, sort_keys=True, indent=2)
                    f.flush()
            if not os.path.exists(os.path.join(args.local_results_path, "tb_logs", args.unique_token, "config.json")):
                with open(os.path.join(args.local_results_path, "tb_logs", args.unique_token, "config.json"), "w") as f:
                    json.dump(flatten(vars(args)), f, sort_keys=True, indent=2)
                    f.flush()
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            if not runner.model_choosed:
                for mode_id in range(args.num_modes):
                    learners[mode_id].save_models(save_path, mode_id)
            else:
                learners[cur_mode_id].save_models(save_path, mode_id)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")

def _visualize(args, logger, runner, env_info, scheme, groups, preprocess):
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                           preprocess=preprocess,
                           device="cpu" if args.buffer_cpu_only else args.device)
    macs = [mac_REGISTRY[args.mac](buffer.scheme, groups, args) for _ in range(args.num_modes)]
    for mac in macs:
        mac.cuda()

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, macs=macs)

    assert args.checkpoint_path != "", "The `checkpoint_path` must be valid."

    timesteps = []
    timestep_to_load = 0

    if not os.path.isdir(args.checkpoint_path):
        logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
        return

    # Go through all files in args.checkpoint_path
    for name in os.listdir(args.checkpoint_path):
        full_name = os.path.join(args.checkpoint_path, name)
        # Check if they are dirs the names of which are numbers
        if os.path.isdir(full_name) and name.isdigit():
            timesteps.append(int(name))

    if args.load_step == 0:
        # choose the max timestep
        timestep_to_load = max(timesteps)
    else:
        # choose the timestep closest to load_step
        timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

    model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

    logger.console_logger.info("Loading model from {}".format(model_path))
    for mode_id in range(args.num_modes):
        macs[mode_id].load_models(model_path, mode_id)
    runner.t_env = timestep_to_load


    logger.console_logger.info("Beginning visualization for {} modes.".format(args.num_modes))

    disp = Display(backend="xvfb").start()
    for mode_id in range(args.num_modes):
        args.env_args["logdir"] = os.path.join(model_path, "replay", str(mode_id))
        runner.create_env(args.env_args)
        for i in tqdm(range(10)):
            runner.run(test_mode=True, mode_id=mode_id)

    runner.close_env()
    disp.stop()
    logger.console_logger.info("Finished Visualization.")

def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
