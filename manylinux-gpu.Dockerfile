FROM tensorflow/tensorflow:custom-op-gpu-ubuntu16

#-----------------------------------------------------
# Download CMake
RUN curl -L -O https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2-Linux-x86_64.tar.gz && \
    tar -xvzf cmake-3.15.2-Linux-x86_64.tar.gz && \
    rm cmake-3.15.2-Linux-x86_64.tar.gz
ENV PATH /cmake-3.15.2-Linux-x86_64/bin:$PATH

#-----------------------------------------------------
# Install Miniconda and Python
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh   \
    && chmod +x ~/miniconda.sh  \
    && ~/miniconda.sh -b -p /opt/conda  \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install -y \
        python=3.7 \
        pytorch \
        pybind11 \
        tensorflow-gpu \
        scikit-image \
    && conda clean -ya

#---------------------------------------------------
# Copy Redner app to container
COPY . /app
WORKDIR /app
RUN chmod -R a+w /app

#-----------------------------------------------------
# Build wheels and install
WORKDIR /app
RUN if [ -d "build" ]; then rm -rf build; fi \
    && REDNER_CUDA=1 PROJECT_NAME=redner-gpu python -m pip wheel -w /dist --verbose . \
    && conda run python -m pip install /dist/redner*.whl

#-----------------------------------------------------
# Create a Python 3.6 environment
RUN conda create -n py36 python=3.6 \
    && conda run -n py36 conda install -y \
        pytorch \
        pybind11 \
        tensorflow-gpu \
        -c pytorch \
    && conda clean -ya

#-----------------------------------------------------
# Build wheels and convert
WORKDIR /app
RUN if [ -d "build" ]; then rm -rf build; fi \
    && REDNER_CUDA=1 PROJECT_NAME=redner-gpu conda run -n py36 python -m pip wheel -w /dist --verbose . \
    && for f in /dist/redner*-linux_*.whl; \
    do \
      auditwheel repair "$f" -w /dist; \
    done

#-----------------------------------------------------
# Create a Python 3.8 environment
RUN conda create -n py38 python=3.8 \
    && conda run -n py38 conda install -y \
        pytorch \
        pybind11 \
        -c pytorch \
    && conda run -n py38 pip install tensorflow-gpu \
    && conda clean -ya

#-----------------------------------------------------
# Build wheels and convert
WORKDIR /app
RUN if [ -d "build" ]; then rm -rf build; fi \
    && REDNER_CUDA=1 PROJECT_NAME=redner-gpu conda run -n py38 python -m pip wheel -w /dist --verbose . \
    && for f in /dist/redner*-linux_*.whl; \
    do \
      auditwheel repair "$f" -w /dist; \
    done
