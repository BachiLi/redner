#include "atomic.h"
#include "test_utils.h"

void test_atomic() {
    float x = 1.f;
    atomic_add(x, 1.f);
    equal_or_error<float>(__FILE__, __LINE__, x, 2.f);
    double y = 1.0;
    atomic_add(y, 1.0);
    equal_or_error<double>(__FILE__, __LINE__, x, 2.0);
}
