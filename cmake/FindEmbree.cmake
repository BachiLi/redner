find_path(EMBREE_INCLUDE_PATH embree3/rtcore.h
  /usr/include
  /usr/local/include
  /opt/local/include)

find_library(EMBREE_LIBRARY NAMES embree3 PATHS 
  /usr/lib 
  /usr/local/lib 
  /opt/local/lib)

if (EMBREE_INCLUDE_PATH AND EMBREE_LIBRARY)
  set(EMBREE_FOUND TRUE)
endif ()
