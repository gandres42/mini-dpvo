FROM rocm/dev-ubuntu-26.04
ARG USERNAME=USERNAME
ARG CONTAINER_ID=CONTAINER_ID
ARG USER_UID=1000
ARG USER_GID=$USER_UID
ENV DEBIAN_FRONTEND=noninteractive

# create non-root user and add to sudo
RUN if id -u $USER_UID ; then userdel `id -un $USER_UID` ; fi
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# update and install basic tools
RUN apt-get upgrade -y
RUN apt-get install -y \
    ssh \
    curl \
    git \
    wget \
    python3-pip \
    python-is-python3 \
    jq \
    ninja-build \
    nano \
    libnuma-dev \
    libgl1 \
    libglib2.0-0t64

# deps
ENV PYTORCH_ROCM_ARCH=gfx1030
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.14 --break-system-packages
RUN cd /tmp && \
    git clone https://github.com/gandres42/lietorch.git --recurse-submodules && \
    cd lietorch && \
    pip install --no-build-isolation --break-system-packages .

# dev environemnt setup
RUN chsh -s /bin/bash $USERNAME
USER $USERNAME

# non-root `pip install` puts console scripts (`hf`, `mini-dpvo-app`) here
ENV PATH=/home/$USERNAME/.local/bin:$PATH
RUN echo "source /home/$USERNAME/.dotfiles/setup.bash" >> ~/.bashrc && \
    echo 'export PIP_BREAK_SYSTEM_PACKAGES=1' >> ~/.bashrc && \
    echo 'export HSA_OVERRIDE_GFX_VERSION=10.3.0' >> ~/.bashrc
ENV CONTAINER_ID $CONTAINER_ID