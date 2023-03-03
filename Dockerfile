ARG DOCKER_BASE="nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04"
FROM $DOCKER_BASE

LABEL author = 'Joy' \
    description="This image is built for producing the experimental results in SPD."

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTORCH="1.7.1+cu110"
ARG USE_TSINGHUA_PIP="False"

# dependencies installation
RUN apt-get update --fix-missing && \
    apt-get --no-install-recommends install --fix-missing -yq apt-utils ca-certificates && \
    apt-get --no-install-recommends install --fix-missing -yq git cmake build-essential vim tmux \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3.8 \
    python3-pip && \
    rm /usr/bin/python /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python3 && \
    python -m pip install --upgrade pip

COPY ./third_party/ /root/envs

# change the source of pip if necessary
RUN if [ "$USE_TSINGHUA_PIP"="True" ]; \
    then echo "USE_TSINGHUA_PIP=True" && \
    mkdir /root/.pip && \
    # no need to add option `-e` after `echo` in Dockerfile anymore
    echo "[global]\n"\
"index-url = https://pypi.tuna.tsinghua.edu.cn/simple\n"\
"\n"\
"[install]\n"\
"trusted-host = pypi.tuna.tsinghua.edu.cn" > /root/.pip/pip.conf; \
    else echo "USE_TSINGHUA_PIP=False"; fi

RUN pip install setuptools wheel && \
    pip install psutil && \
    pip install /root/envs/football && \
    pip install dm-tree==0.1.7 && pip install /root/envs/smac && \
    pip install pettingzoo[mpe]==1.17.0 && \
    rm -r /root/envs && \
    pip install sacred numpy==1.22.4 scipy gym matplotlib seaborn pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger pyvirtualdisplay tqdm protobuf==3.20.1 && \
    pip install torch==$PYTORCH -f https://download.pytorch.org/whl/torch_stable.html

RUN	rm -rf /tmp/* /var/cache/* /usr/share/doc/* /usr/share/man/* /var/lib/apt/lists/*
COPY . /root/spd
WORKDIR "/root/spd"
