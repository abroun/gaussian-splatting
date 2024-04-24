#!/usr/bin/env bash

USER_ID=`id -u`
DOCKER_GROUP_ID=`getent group docker | awk '{split($0,a,":"); print a[3]}'`

IMAGE_TAG=env/gaussian_splatting
CONTAINER_NAME_ENV=gaussian_splatting_env

# With Docker BuildKit it seems hard to see progress and caching doesn't
# seem to work so well. Therefore we disable BuildKit and use the legacy
# Docker build system.
# https://stackoverflow.com/questions/64804749/why-is-docker-build-not-showing-any-output-from-commands
export DOCKER_BUILDKIT=0

# Create the docker environment for building Gaussian Splatting
docker build -t $IMAGE_TAG docker/dev_env/. \
    --build-arg "DOCKER_GROUP_ID=${DOCKER_GROUP_ID}" \
    --build-arg "USER_ID=${USER_ID}" \
    --build-arg "USER_NAME=${USER}"

EXTRA_ARGS=

# Enable this for NSight profiling
EXTRA_ARGS+=" --security-opt seccomp=seccomp_chrome_and_perf.json "

# Start the command line
docker run --rm -it \
    --name ${CONTAINER_NAME_ENV} \
    -e DISPLAY \
    -v $(pwd):/src \
    -v /tmp:/tmp \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /home/${USER}/.docker:/home/${USER}/.docker \
    -v /home/${USER}/.gitconfig:/home/${USER}/.gitconfig:ro \
    -v /home/${USER}/.aws:/home/${USER}/.aws:ro \
    -v /home/${USER}/.ssh:/home/${USER}/.ssh:ro \
    -v /run/user/${USER_ID}/keyring:/run/user/${USER_ID}/keyring \
    -v /media/datasets:/media/datasets \
    -v /home/${USER}/dev/personal:/home/${USER}/dev/personal \
    -e SSH_AUTH_SOCK \
    -e OCEAN_LOCKER_HOST_DIR=$(pwd) \
    -e OCEAN_LOCKER_WEB_HOST_DIR=$(pwd)/web \
    -e GUI_HOST_DIR=$(pwd)/gui \
    --network=host \
    ${EXTRA_ARGS} \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus=all $IMAGE_TAG