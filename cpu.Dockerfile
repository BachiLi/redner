FROM ubuntu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        libtbb-dev \
        pkg-config \
        libopenexr-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------
# Download CMake
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2-Linux-x86_64.tar.gz && \
    tar -xvzf cmake-3.15.2-Linux-x86_64.tar.gz && \
    rm cmake-3.15.2-Linux-x86_64.tar.gz
ENV PATH /cmake-3.15.2-Linux-x86_64/bin:$PATH

#-----------------------------------------------------
# Upgrade to gcc 7
# https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt update \
    && apt install g++-7 -y \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                           --slave /usr/bin/g++ g++ /usr/bin/g++-7 \
    && update-alternatives --config gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------
# Install Miniconda and Python
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh   \
    && chmod +x ~/miniconda.sh  \
    && ~/miniconda.sh -b -p /opt/conda  \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install -y \
        python=$PYTHON_VERSION \
        pyyaml=5.1 \
        scipy=1.2.1 \
        numpy=1.16.4 \
        ipython=7.5 \
        mkl=2019.4 \
        mkl-include=2019.4 \
        cython=0.29.10 \
        typing=3.6.4 \
        ffmpeg=4.0 \
        scikit-image=0.15.0 \
        pybind11=2.2.4 \
    && pip install ninja==1.9.0.post1 \
    && pip install tensorflow==1.14.0 \
    && conda install --yes pytorch-cpu=1.1.0 torchvision-cpu=0.3.0 -c pytorch \
    && /opt/conda/bin/conda clean -ya

#---------------------------------------------------
# Copy Redner app to container
COPY . /app
WORKDIR /app
RUN chmod -R a+w /app

ENV LD_LIBRARY_PATH /usr/local/lib:${LD_LIBRARY_PATH}

#-----------------------------------------------------
# Build Redner C++ code
WORKDIR /app
RUN if [ -d "build" ]; then rm -rf build; fi \
    && mkdir build \
    && cd build \
    && cmake .. -DEMBREE_ISPC_SUPPORT=false -DEMBREE_TUTORIALS=false \
    && make install -j24 \
    && cd / \
    && rm -rf /app/build/

