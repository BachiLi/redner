#include "aabb.h"

std::ostream& operator<<(std::ostream &os, const AABB3 &bounds) {
    return os << "p_min -> " << bounds.p_min << ", p_max ->" << bounds.p_max;
}

std::ostream& operator<<(std::ostream &os, const AABB6 &bounds) {
    return os << "p_min -> " << bounds.p_min << ", p_max ->" << bounds.p_max <<
                 ", d_min -> " << bounds.d_min << ", d_max ->" << bounds.d_max;
}
