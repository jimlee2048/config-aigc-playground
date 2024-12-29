FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV WORKDIR=/workspace
WORKDIR ${WORKDIR}
ENV COMFYUI_PATH=${WORKDIR}/ComfyUI
ENV COMFYUI_MN_PATH=${COMFYUI_PATH}/custom_nodes/ComfyUI-Manager

ARG DEBIAN_FRONTEND=noninteractive
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        ca-certificates git cmake ninja-build wget curl aria2 ffmpeg libegl1-mesa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install comfyui - method 1: manually clone
RUN git clone --single-branch https://github.com/comfyanonymous/ComfyUI.git ${COMFYUI_PATH} \
    && git clone --single-branch https://github.com/ltdrdata/ComfyUI-Manager.git ${COMFYUI_MN_PATH}

VOLUME [ "${COMFYUI_PATH}/user", "${COMFYUI_PATH}/output" , "${COMFYUI_PATH}/models", "${COMFYUI_PATH}/custom_nodes"]

# install comfyui - method 2: using comfy-cli
# RUN comfy --skip-prompt install --version=nightly --skip-torch-or-directml --nvidia --cuda-version 12.4

# master(nightly), or version tag like v0.1.0
# https://github.com/comfyanonymous/ComfyUI/tags
ARG COMFYUI_VERSION="master"
RUN git -C ${COMFYUI_PATH} checkout ${COMFYUI_VERSION}
# main(nightly), or version tag like v0.1.0
# https://github.com/ltdrdata/ComfyUI-Manager/tags
ARG COMFYUI_MN_VERSION="main"
RUN git -C ${COMFYUI_MN_PATH} checkout ${COMFYUI_MN_VERSION}

# install comfyui basic requirements
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        -r ${COMFYUI_PATH}/requirements.txt \
        -r ${COMFYUI_MN_PATH}/requirements.txt \
        xformers \
        --index-url https://download.pytorch.org/whl/cu124 \
        --extra-index-url https://pypi.org/simple

# instanll comfy-cli
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install comfy-cli \
    && comfy --skip-prompt tracking disable \
    && comfy set-default ${COMFYUI_PATH}

EXPOSE 8188

# Let .pyc files be stored in one place
ENV PYTHONPYCACHEPREFIX="/root/.cache/pycache"
# Let PIP install packages to /root/.local
ENV PIP_USER=true
# Add above to PATH
ENV PATH="${PATH}:/root/.local/bin"
# Suppress [WARNING: Running pip as the 'root' user]
ENV PIP_ROOT_USER_ACTION=ignore

COPY boot.py .
CMD [ "python", "boot.py" ]