FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates git cmake ninja-build wget curl aria2 ffmpeg libegl1-mesa \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install comfy-cli \
    && comfy --skip-prompt tracking disable

# install method 1: manually clone
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI \
    && git clone https://github.com/ltdrdata/ComfyUI-Manager.git /workspace/ComfyUI/custom_nodes/ComfyUI-Manager \
    && comfy set-default /workspace/ComfyUI

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        -r /workspace/ComfyUI/requirements.txt \
        -r /workspace/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt

# install method 2: using comfy-cli
# RUN comfy --skip-prompt install --version=nightly --skip-torch-or-directml --nvidia --cuda-version 12.4

EXPOSE 8188
ENTRYPOINT ["comfy", "launch", "--", "--listen", "0.0.0.0,::", "--port", "8188"]