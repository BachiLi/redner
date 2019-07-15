# Environment without Tensorflow custom ops
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ARG PYTHON_VERSION=3.6
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


#-----------------------------------------------------
# Install Miniconda and Python
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh   \
    && chmod +x ~/miniconda.sh  \
    && ~/miniconda.sh -b -p /opt/conda  \
    && rm ~/miniconda.sh  \
    && /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing  \
    && /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
RUN pip install ninja

#-----------------------------------------------------
# Build CMake
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.12.4/cmake-3.12.4.tar.gz && \
    tar -xvzf cmake-3.12.4.tar.gz && \
    rm cmake-3.12.4.tar.gz && \
    cd cmake-3.12.4 && \
    ./bootstrap && make && make install

#-----------------------------------------------------
# Install Linux stuffs
RUN apt-get update && apt-get install -y \
    libtbb-dev \ 
    pkg-config \ 
    libglfw3-dev \
    libopenexr-dev \ 
    libopenimageio-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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
# Install Python dependencies
#-----------------------------------------------------
# Install Tensorflow
RUN pip install tensorflow==1.14.0 \ 
    && conda install --yes pytorch-cpu torchvision-cpu -c pytorch \
    && conda install --yes \ 
        ffmpeg=4.0 \
        scikit-image=0.15.0 \
        scipy=1.2.1 \ 
    && conda clean -ya

# Copy Redner app to container
COPY . /app
WORKDIR /app
RUN chmod -R a+w /app \
    && cp dockerfiles/CMakeLists.txt . \
    && cp dockerfiles/FindEmbree.cmake ./cmake/ \
    && cp dockerfiles/FindThrust.cmake ./cmake/ \
    && cp dockerfiles/embree/CMakeLists.txt ./embree/

#-----------------------------------------------------
# Install Embree
ARG EMBREE_VERSION=3.2.4
WORKDIR /
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/embree/embree/releases/download/v${EMBREE_VERSION}/embree-${EMBREE_VERSION}.x86_64.linux.tar.gz \
    && tar -xvzf embree-${EMBREE_VERSION}.x86_64.linux.tar.gz \
    && rm embree-${EMBREE_VERSION}.x86_64.linux.tar.gz \
    && mv embree-${EMBREE_VERSION}.x86_64.linux  /app/embree/

#RUN git clone --recursive https://github.com/supershinyeyes/redner.git 
#-----------------------------------------------------
    # && cp -r dockerfiles/dependencies/embree-${EMBREE_VERSION}.x86_64.linux ./embree
    
    
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
    && conda install --yes pybind11=2.2.4 \
    && cmake .. \
    && make install -j 8 \
    && cd /app \
    && rm -rf build

