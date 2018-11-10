#pragma once

#include "vector.h"

// From https://www.wikiwand.com/en/Cohen%E2%80%93Sutherland_algorithm
// Thank you wikipedia!

// Cohen–Sutherland clipping algorithm clips a line from
// P0 = (x0, y0) to P1 = (x1, y1) against a rectangle with
// diagonal from (xmin, ymin) to (xmax, ymax).

using OutCode = int;
constexpr int OC_INSIDE = 0;
constexpr int OC_LEFT = 1;
constexpr int OC_RIGHT = 2;
constexpr int OC_BOTTOM = 4;
constexpr int OC_TOP = 8;

DEVICE
OutCode compute_out_code(const Vector2 &v) {
    OutCode code;
    code = OC_INSIDE;           // initialised as being inside of [[clip window]]
    if (v[0] < 0.f) {        // to the left of clip window
        code |= OC_LEFT;
    } else if (v[0] > 1.f) { // to the right of clip window
        code |= OC_RIGHT;
    }
    if (v[1] < 0.f) {        // below the clip window
        code |= OC_BOTTOM;
    } else if (v[1] > 1.f) { // above the clip window
        code |= OC_TOP;
    }
    return code;
};

DEVICE
bool clip_line(const Vector2 &v0, const Vector2 &v1,
               Vector2 &v0c, Vector2 &v1c) {
    // Compute the bit code for a point (x, y) using the clip rectangle
    // bounded diagonally by (xmin, ymin), and (xmax, ymax)

    // compute outcodes for P0, P1, and whatever point lies outside the clip rectangle
    OutCode outcode0 = compute_out_code(v0);
    OutCode outcode1 = compute_out_code(v1);
    bool accept = false;
    v0c = v0;
    v1c = v1;

    while (true) {
        if (!(outcode0 | outcode1)) {
            // bitwise OR is 0: both points inside window; trivially accept and exit loop
            accept = true;
            break;
        } else if (outcode0 & outcode1) {
            // bitwise AND is not 0: both points share an outside zone (LEFT, RIGHT, TOP,
            // or BOTTOM), so both must be outside window; exit loop (accept is false)
            break;
        } else {
            // failed both tests, so calculate the line segment to clip
            // from an outside point to an intersection with clip edge
            auto v = Vector2{0, 0};

            // At least one endpoint is outside the clip rectangle; pick it.
            OutCode outcodeOut = outcode0 ? outcode0 : outcode1;

            // Now find the intersection point;
            // use formulas:
            //   slope = (y1 - y0) / (x1 - x0)
            //   x = x0 + (1 / slope) * (ym - y0), where ym is ymin or ymax
            //   y = y0 + slope * (xm - x0), where xm is xmin or xmax
            // No need to worry about divide-by-zero because, in each case, the
            // outcode bit being tested guarantees the denominator is non-zero
            if (outcodeOut & OC_TOP) {           // point is above the clip window
                v[0] = v0c[0] + (v1c[0] - v0c[0]) * (1.f - v0c[1]) / (v1c[1] - v0c[1]);
                v[1] = 1.f;
            } else if (outcodeOut & OC_BOTTOM) { // point is below the clip window
                v[0] = v0c[0] + (v1c[0] - v0c[0]) * (0.f - v0c[1]) / (v1c[1] - v0c[1]);
                v[1] = 0.f;
            } else if (outcodeOut & OC_RIGHT) {  // point is to the right of clip window
                v[1] = v0c[1] + (v1c[1] - v0c[1]) * (1.f - v0c[0]) / (v1c[0] - v0c[0]);
                v[0] = 1.f;
            } else if (outcodeOut & OC_LEFT) {   // point is to the left of clip window
                v[1] = v0c[1] + (v1c[1] - v0c[1]) * (0.f - v0c[0]) / (v1c[0] - v0c[0]);
                v[0] = 0.f;
            }

            // Now we move outside point to intersection point to clip
            // and get ready for next pass.
            if (outcodeOut == outcode0) {
                v0c[0] = v[0];
                v0c[1] = v[1];
                outcode0 = compute_out_code(v0c);
            } else {
                v1c[0] = v[0];
                v1c[1] = v[1];
                outcode1 = compute_out_code(v1c);
            }
        }
    }

    return accept;
}
