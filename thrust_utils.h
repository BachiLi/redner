#pragma once

#define DISPATCH(use_gpu, f, args...) \
    ((use_gpu) ? f(thrust::device, args) : f(thrust::host, args))

