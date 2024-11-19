FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ARG DOCKER_GROUP_ID
ARG USER_ID
ARG USER_NAME

# Install deps
RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y  \
    wget zip unzip tar git less curl \
    build-essential gdb pkg-config \
    apt-transport-https \
    ca-certificates \
    gnupg \
    python3 python3-pip \
    gcc-10 gcc-10-base gcc-10-doc g++-10 \
    libstdc++-10-dev \
    gnupg-agent \
    software-properties-common \
    jq \
    libomp-dev \
    libboost-all-dev \
    time bison flex nasm libnss3 libasound2 \
    iputils-ping avahi-daemon avahi-discover avahi-utils libnss-mdns mdns-scan \
    net-tools sudo

# Install docker
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

RUN apt-get update && \
    apt-get install -y \
    docker-ce \
    docker-ce-cli \
    docker-compose-plugin \
    containerd.io

# Install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-linux-x86_64.sh -O /usr/cmake.sh
WORKDIR /usr
RUN sh ./cmake.sh --skip-license

# Install NVidia debug tools
# RUN apt-get update && apt-get install -y \
#   cuda-nsight-11-4 \
#   cuda-nsight-compute-11-4 \
#   cuda-nsight-systems-11-4

# Install Vulkan SDK
# RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add -
# RUN wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.231-focal.list https://packages.lunarg.com/vulkan/1.3.231/lunarg-vulkan-1.3.231-focal.list
# RUN apt-get update && apt-get install -y vulkan-sdk vulkan-validationlayers-dev spirv-tools glslang-tools
# RUN apt-get update && apt-get install -y mesa-utils

# Gui deps
RUN apt-get update && apt-get install -y xterm

# Python deps
RUN pip3 install plyfile==0.8.1 tqdm
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install scipy
RUN pip3 install bson

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3.10-venv

RUN apt-get install -y \
  libglew-dev libassimp-dev libboost-all-dev \
  libgtk-3-dev libopencv-dev libglfw3-dev \
  libavdevice-dev libavcodec-dev libeigen3-dev \
  libxxf86vm-dev libembree-dev

# Setup docker config dir as home dir will be read only
RUN mkdir /docker_config
RUN chmod 777 /docker_config
ENV DOCKER_CONFIG=/docker_config

# Setup username and id so that it matches host machine (otherwise files will be written as root)
RUN groupmod -g ${DOCKER_GROUP_ID} docker
RUN useradd -u ${USER_ID} ${USER_NAME}
RUN usermod -a -G docker ${USER_NAME}
ENV USER=${USER_NAME}
ENV USER_ID=${USER_ID}
ENV HOME=/home/${USER_NAME}
RUN mkdir -p /home/${USER_NAME}
RUN chown ${USER_NAME}:${USER_NAME} /home/${USER_NAME}
RUN mkdir -p /run/user/${USER_ID}/keyring
RUN chown ${USER_NAME}:${USER_NAME} /run/user/${USER_ID}/keyring

# Give the user permission to start the dbus and avahi-daemon services. This is needed to get mDNS
# to work so that we can connect to addresses like blueos.local
RUN echo "${USER_NAME} ALL = NOPASSWD: /usr/sbin/service dbus start" >> /etc/sudoers
RUN echo "${USER_NAME} ALL = NOPASSWD: /usr/sbin/service avahi-daemon start" >> /etc/sudoers
RUN cat /etc/sudoers
USER ${USER_NAME}

WORKDIR /home/${USER_NAME}
RUN touch .bashrc

# Install NodeJS
# https://stackoverflow.com/a/60137919
SHELL ["/bin/bash", "--login", "-i", "-c"]
RUN wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
RUN source /home/${USER_NAME}/.bashrc && nvm install 16.4.2
RUN source /home/${USER_NAME}/.bashrc && npm install -g gitpkg

WORKDIR /src
CMD sudo service dbus start && sudo service avahi-daemon start && /bin/bash