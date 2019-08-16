FROM shinyeyes/redner:cpu-without-optix

ARG EMBREE_VERSION=3.2.4

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

