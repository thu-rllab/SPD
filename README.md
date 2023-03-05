# SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning
This is the source code for reproducing the experimental results in ***SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning*** (NeurIPS 2022).

We custom the code from the Open-source code [PyMarl2](https://github.com/hijkzzz/pymarl2).

If you want to know more information about this work, please refer to our [paper](https://openreview.net/forum?id=jJwy2kcBYv) and visit our [google site](https://sites.google.com/view/spd-umarl/).



## News
* Update on 2023/03/03: Release the original training code for SPD.

## Citation
If you find our project useful in your research, please consider citing us:
```
@inproceedings{jiangspd,
  title={SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning},
  author={Jiang, Yuhang and Shao, Jianzhun and He, Shuncheng and Zhang, Hongchang and Ji, Xiangyang},
  booktitle={Advances in Neural Information Processing Systems}
}
```

## Contents
- [SPD: Synergy Pattern Diversifying Oriented Unsupervised Multi-agent Reinforcement Learning](#spd-synergy-pattern-diversifying-oriented-unsupervised-multi-agent-reinforcement-learning)
  - [News](#news)
  - [Citation](#citation)
  - [Contents](#contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [Depoly with Docker](#depoly-with-docker)
      - [Pull the Docker Image Directly](#pull-the-docker-image-directly)
      - [Build from Dockerfile](#build-from-dockerfile)
    - [Deploy with the source code](#deploy-with-the-source-code)
      - [Install Dependencies](#install-dependencies)
      - [Install Necessary Packages in the Anaconda Environment](#install-necessary-packages-in-the-anaconda-environment)
  - [Usage](#usage)
    - [Create a Docker Container for SPD](#create-a-docker-container-for-spd)
    - [Command Line Tool](#command-line-tool)
      - [MPE](#mpe)
        - [URL Training](#url-training)
        - [URL Evaluation](#url-evaluation)
      - [GRF](#grf)
        - [URL Training](#url-training-1)
        - [Train on Downstream Tasks](#train-on-downstream-tasks)
      - [SMAC](#smac)
  - [Contacts](#contacts)
  - [Main Contributors](#main-contributors)
  - [Acknowledgment](#acknowledgment)
  - [License](#license)


## Requirements
* Python >= 3.8
* [PyTorch](https://pytorch.org/) >= 1.7.1
* [Google Research Football](https://github.com/google-research/football)
* [PettingZoo[mpe] == 1.17.0](https://pettingzoo.farama.org/environments/mpe/)
* StarCraftII & [SMAC](https://github.com/oxwhirl/smac)
* System: Linux (Ubuntu >= 18.04 is recommanded)

## Installation
We provide installation instructions for both deploying with ***Docker*** and deploying with the ***source code***, while the former one is recommended.

### Depoly with Docker
Please ensure that **Nvidia-Docker-2.0** has been installed correctly on your machine before deploying with docker.

Here, two ways to get the docker image for SPD are listed below.

#### Pull the Docker Image Directly
The docker image can be directly pulled from [Dockerhub](https://hub.docker.com/repository/docker/joy1002/spd-umarl/general).

```shell
docker pull joy1002/spd-umarl:latest
```
**Note**: Pytorch-1.7.1 with CUDA-11.0 is installed in this docker image, thus you should make sure that your GPU supports this version of CUDA.
Otherwise, you need to follow the guidance below to build the docker image with a proper Pytorch version.

#### Build from Dockerfile
To build the docker image for SPD and run the experiments with GPUs, you should find a proper version of the basic nvidia docker image from [nvidia/cuda](https://hub.docker.com/r/nvidia/cuda) and then assign the [PyTorch version](https://pytorch.org/get-started/previous-versions/).

Here we use `nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04` and `PYTORCH=1.7.1+cu110` as an example
```shell
# clone the source code and the submodules
git clone --recursive git@github.com:thu-rllab/SPD.git

# build the docker image and it will take a while.
# if you are in China, you could also add `--build-arg USE_TSINGHUA_PIP=True` 
# to use the pip source of Tsinghua and speed up the installation of python packages.
docker build -t spd-umarl:latest \
    --build-arg DOCKER_BASE=nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 \
    --build-arg PYTORCH=1.7.1+cu110 \
    SPD
```


### Deploy with the source code
Also, you could deploy with the source code.
[Anaconda3](https://www.anaconda.com/) is recommanded for managing the experiment environment.
#### Install Dependencies
```bash
# dependencies for gfootball
# you may need to add `sudo` to install these dependencies
apt-get update && apt-get install git cmake build-essential \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc
```

#### Install Necessary Packages in the Anaconda Environment
```bash
# create a new environment
conda create -n spd python=3.8
conda activate spd

# install dependencies for gfootball
pip install --upgrade pip setuptools wheel
pip install psutil

# clone the source code and the submodules
git clone --recursive git@github.com:thu-rllab/SPD.git
cd SPD

# install gfootball
pip install ./third_party/football

# install smac
pip install dm-tree==0.1.7 && pip install ./third_party/smac

# install mpe
pip install pettingzoo[mpe]==1.17.0

# install dependencies for pymarl2
pip install sacred numpy==1.22.4 scipy gym==0.24.1 matplotlib seaborn pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger pyvirtualdisplay tqdm protobuf==3.20.1

# install pytorch, please refer to the official site of PyTorch for a proper version.
# here we also use torch==1.7.1+cu110 as an example.
pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage

### Create a Docker Container for SPD
To carry out the experiments, you should first attatch to an environment with all dependencies installed.
Here we give some guides on how to create a docker container for SPD.
**Note if you *deploy with the source code*, you can just skip this part.**

If you want to reproduce the SMAC experimental results, remember to mount the *StarCraftII* (which can be downloaded by the script `install_sc2.sh`) into the docker container.

```shell
# create a `results` folder in your host first to save the experiments results
mkdir <path_to_results_folder>

# create a container named <your_container_name>
# the third line is to mount the StarCraftII, and the forth line is to mount the `results` folder in your host
docker run -itd --gpus all \
    --name <your_container_name> \
    -v <path_to_StarCraftII>:/root/StarCraftII:ro \
    -v <path_to_results>:/root/spd/results \
    <spd_image> \
    /bin/bash
# attatch to the container
docker exec -it <your_container_name> /bin/bash
# then you can run any commands in the container like you are in Ubuntu-18.04


# or you can just execute the commands to run the experiments as well
# please refer to **Command Line Tool** part for details of <spd_experiments_command>
docker run -d --gpus all \
    --name <your_container_name> \
    -v <path_to_StarCraftII>:/root/StarCraftII:ro \
    -v <path_to_results>:/root/spd/results \
    <spd_image> \
    <spd_experiments_command>
```
### Command Line Tool
Following the instructions below, you could run SPD and other URL approaches to produce the Unsupervised Multi-agent Reinforcement Learning experimental results in our work.

Note: Though we custom the code from PyMarl2, the `url_runner` only inherits from the `episode_runner` provided by it.
Therefore we are not sure about the performance of the `parallel_runner` and we recommend you to not use the parallel running script.

#### MPE
##### URL Training 
In our work, all the URL training for Multi-agent tasks is using QMIX as the baseline.
The config file is in the directory `src/config/algs/url_qmix_mpe.yaml` and you can check it for details about the hyper-parameters.
```bash
# SPD
# argument `url_algo` defaults to `gwd` corresponding to Gromov-Wassersteim Discrepancy (GWD), which is used in our method SPD
# note: all the components of our mathod SPD is named with `gwd`
python src/main.py --config=url_qmix_mpe --env-config=mpe_simple_tag with t_max=4050000

# DIAYN
python src/main.py --config=url_qmix_mpe --env-config=mpe_simple_tag with url_algo=diayn t_max=4050000

# WURL
python src/main.py --config=url_qmix_mpe --env-config=mpe_simple_tag with url_algo=wurl t_max=4050000
```
The learned policies' weights will be stored in the folder `results/models/`, and named with the timestamp such as `gwd_qmix_simple_tag_agents-4__2022-05-12_15-52-17`.


##### URL Evaluation
To reproduce the results in Sec. 5.1 in SPD, you need to specify the model location and carry out the evaluation process.
```bash
# here we give an example
python src/main.py --config=url_eval_mpe \
    --env-config=mpe_simple_tag \
    --exp-config=results/models/gwd_qmix_simple_tag_agents-4__2022-05-12_15-52-17 \
    with eval_process=True
```


#### GRF
##### URL Training
The config file is in the directory `src/config/algs/url_gfootball.yaml` and you can check it for details about the hyper-parameters.
We trained 20 different synergy patterns (joint-policies) in GRF in our paper and you can modify the argument `num_modes` to change it.
```bash
python src/main.py --config=url_gfootball --env-config=academy_3_vs_1_with_keeper with num_modes=20 ball_graph=True
```

Additionally, we also provide visualization scripts for the Fig. 5 in SPD (Position Trajecories of Z=20 synergy patterns) in the Appendix.
```bash
# `<path>` should be absolute path or relative path like `results/models/gwd_qmix_football__2022-04-21_19-38-03`.
# remember to copy the config.json from the `results/sacred/` to `results/models/`
python src/main.py --exp-config=<path> with vis_process=True --env-config=gfootball_vis
```

##### Train on Downstream Tasks
Similarly, you should specify the locatin of the model to load to reproduce the results in Sec. 5.2 in SPD.
```bash
# here we give an example
# the details about `env_args.map_style` please refer to file `src/envs/gfootball/academy_3_vs_1_with_keeper.py`
# Besides, the program will test all the synergy patterns (here the number of them is 20) to select the best one at first
# and then use the selected one as the network initialization with epsilon start with a given rate (here it is 0.2) to run the training process.
python src/main.py --config=url_gfootball_load \
    --env-config=academy_3_vs_1_with_keeper \
    with num_modes=20 env_args.map_style=0 t_max=4050000 \
    test_nepisode=50 epsilon_start=0.2 \
    checkpoint_path=results/models/gwd_qmix_academy_3_vs_1_with_keeper_agents-3__2022-05-07_05-34-39
```

#### SMAC
The commands for runing the experiments on SMAC (in our Appendix. E) are similar to running SPD on GRF except the config files are `url_qmix_sc2.yaml` and `url_qmix_sc2_load.yaml`.

## Contacts
* jiangyh19@mails.tsinghua.edu.cn / jiangyuh1112@gmail.com
* sjz18@mails.tsinghua.edu.cn

Any discussions or concerns are welcomed!

## Main Contributors
* [@Joy1112](https://github.com/Joy1112)
* [@qyz55](https://github.com/qyz55)


## Acknowledgment
This work was supported by the National Key R&D Program of China under Grant 2018AAA0102801,National Natural Science Foundation of China under Grant 61620106005.

Our code is based on the open-source code [PyMarl2](https://github.com/hijkzzz/pymarl2), and we really appreciate @hijkzzz's excellent repository.

## License
Code licensed under the Apache License v2.0.
