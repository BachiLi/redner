#pragma once

#ifdef WIN32
#define DISPATCH(use_gpu, f, ...) \
    ((use_gpu) ? f(thrust::device, ##__VA_ARGS__	) : f(thrust::host, ##__VA_ARGS__))
#else
#define DISPATCH(use_gpu, f, args...) \
    ((use_gpu) ? f(thrust::device, args) : f(thrust::host, args))
#endif

