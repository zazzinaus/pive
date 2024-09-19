FROM nvidia/cuda:12.2.2-base-ubuntu22.04

# Set the working directory
WORKDIR /app

# Copy your code to the container
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    wget \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.10.14 from source
RUN cd /usr/src \
    && wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz \
    && tar xzf Python-3.10.14.tgz \
    && cd Python-3.10.14 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && rm -rf /usr/src/Python-3.10.14*

# Update alternatives to use Python 3.10.14
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 1

# Install PyTorch and other dependencies
RUN pip3 install --no-cache-dir torch torchvision \
    && pip3 install --no-cache-dir -r requirements.txt

RUN pip3.10 install --upgrade pip
RUN pip3 install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"