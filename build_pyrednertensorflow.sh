#!/bin/bash

python setup.py install
cd pyrednertensorflow/custom-ops
TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') 
g++ -std=c++11 -shared data_ptr.cc -o data_ptr.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
mv data_ptr.so /
g++ -std=c++11 -shared pytorch_scatter_add.cc -o pytorch_scatter_add.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
mv pytorch_scatter_add.so /

