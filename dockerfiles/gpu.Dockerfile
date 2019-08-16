# Environment without Tensorflow custom ops
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.1ubuntu2 \
        cmake=3.5.1-1ubuntu3 \
        curl=7.47.0-1ubuntu2.13 \
        ca-certificates=20170717~16.04.2 \
        libjpeg-dev=8c-2ubuntu8 \
        libpng-dev \
        libtbb-dev \ 
        pkg-config=0.29.1-0ubuntu1 \ 
        libglfw3-dev=3.1.2-3 \
        libopenexr-dev=2.2.0-10ubuntu2 \ 
        libopenimageio-dev=1.6.11~dfsg0-1ubuntu1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

#-----------------------------------------------------
# Build CMake
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.12.4/cmake-3.12.4.tar.gz && \
    tar -xvzf cmake-3.12.4.tar.gz && \
    rm cmake-3.12.4.tar.gz && \
    cd cmake-3.12.4 && \
    ./bootstrap && make && make install

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
    && conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch \
    && /opt/conda/bin/conda clean -ya


#-----------------------------------------------------
# Install Embree
ARG EMBREE_VERSION=3.2.4
WORKDIR /
RUN curl -L -O https://github.com/embree/embree/releases/download/v${EMBREE_VERSION}/embree-${EMBREE_VERSION}.x86_64.linux.tar.gz \
    && tar -xvzf embree-${EMBREE_VERSION}.x86_64.linux.tar.gz \
    && rm embree-${EMBREE_VERSION}.x86_64.linux.tar.gz

#---------------------------------------------------
# Copy Redner app to container
COPY . /app
WORKDIR /app
RUN chmod -R a+w /app \
    && mv dockerfiles/CMakeLists.txt . \
    && mv dockerfiles/FindEmbree.cmake ./cmake/ \
    && mv dockerfiles/FindThrust.cmake ./cmake/ \
    && mv dockerfiles/embree/CMakeLists.txt ./embree/ \
    && mv /embree-${EMBREE_VERSION}.x86_64.linux  /app/embree/

#-----------------------------------------------------
# Install NVIDIA OptiX
ARG OPTIX_VERSION=5.1.0
WORKDIR /app
RUN mv dockerfiles/dependencies/NVIDIA-OptiX-SDK-${OPTIX_VERSION}-linux64 /usr/local/optix
ENV LD_LIBRARY_PATH /app/embree/embree-${EMBREE_VERSION}.x86_64.linux/lib:/usr/local/optix/lib64:${LD_LIBRARY_PATH}
    

#-----------------------------------------------------
# Build Redner C++ code
WORKDIR /app
RUN if [ -d "build" ]; then rm -rf build; fi \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make install -j 8 \
    && cd / \
    && rm -rf /app/build/

