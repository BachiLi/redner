# https://github.com/PatWie/tensorflow-cmake/blob/master/cmake/modules/FindTensorFlow.cmake

if(APPLE)
  execute_process(
    COMMAND python -c "import tensorflow as tf; print(tf.__version__); print(tf.__cxx11_abi_flag__); print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib() + '/libtensorflow_framework.dylib')"
    OUTPUT_VARIABLE TF_INFORMATION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)
else()
  execute_process(
    COMMAND python -c "import tensorflow as tf; print(tf.__version__); print(tf.__cxx11_abi_flag__); print(tf.sysconfig.get_include()); print(tf.sysconfig.get_lib() + '/libtensorflow_framework.so')"
    OUTPUT_VARIABLE TF_INFORMATION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)
endif()

if("${retcode}" STREQUAL "0")
  string(REPLACE "\n" ";" TF_INFORMATION_LIST ${TF_INFORMATION_STRING})
  list(GET TF_INFORMATION_LIST 0 TF_DETECTED_VERSION)
  list(GET TF_INFORMATION_LIST 1 TF_DETECTED_ABI)
  list(GET TF_INFORMATION_LIST 2 TF_DETECTED_INCLUDE_DIR)
  list(GET TF_INFORMATION_LIST 3 TF_DETECTED_LIBRARY)
  set(TensorFlow_VERSION ${TF_DETECTED_VERSION})
  set(TensorFlow_ABI ${TF_DETECTED_ABI})
  set(TensorFlow_INCLUDE_DIR ${TF_DETECTED_INCLUDE_DIR})
  set(TensorFlow_LIBRARY ${TF_DETECTED_LIBRARY})
  set(TensorFlow_FOUND TRUE)
endif()
