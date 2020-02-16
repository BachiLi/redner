#pragma once

#include "buffer.h"
#include "vector.h"
#include "ray.h"
#include "intersection.h"
#include "thrust_utils.h"

void init_active_pixels(const BufferView<Ray> &rays,
                        BufferView<int> &active_pixels,
                        bool use_gpu,
                        ThrustCachedAllocator &thrust_alloc);
void update_active_pixels(const BufferView<int> &active_pixels,
                          const BufferView<Intersection> &isects,
                          BufferView<int> &new_active,
                          bool use_gpu);

void test_active_pixels(bool use_gpu);
