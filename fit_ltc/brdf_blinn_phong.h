#ifndef _BRDF_BLINN_PHONG_
#define _BRDF_BLINN_PHONG_

#include "brdf.h"

class BrdfBlinnPhong : public Brdf
{
public:

	virtual float eval(const vec3& V, const vec3& L, const float alpha, float& pdf) const
	{
		if(V.z <= 0)
		{
			pdf = 0;
			return 0;
		}

		const vec3 H = normalize(V+L);

        auto smithG1 = [&](const vec3 &v) -> float {
            auto cos_theta = v.z;
            if (dot(v, H) * cos_theta <= 0) {
                return 0;
            }
            // tan^2 + 1 = 1/cos^2
            auto tan_theta = sqrt(1.f / (cos_theta * cos_theta) - 1.f);
            if (tan_theta == 0.0f) {
                return 1;
            }
            auto a = 1.f / (alpha * tan_theta);
            if (a >= 1.6f) {
                return 1;
            }
            auto a_sqr = a*a;
            return (3.535f * a + 2.181f * a_sqr)
                 / (1.0f + 2.276f * a + 2.577f * a_sqr);
        };
        auto G = smithG1(V) * smithG1(L);

		// D
        auto phong_exponent = std::max(2.f / (alpha * alpha) - 2, 0.f);
        auto D = pow(H.z, phong_exponent) *
            (phong_exponent + 2.f) / float(2 * M_PI);

		pdf = fabsf(D * H.z / 4.0f / dot(V,H));
		float res = D * G / 4.0f / V.z;

		return res;
	}

	virtual vec3 sample(const vec3& V, const float alpha, const float U1, const float U2) const
	{
		const float phi = 2.0f*3.14159f * U1;
		const float phong_exponent = std::max(2.f / (alpha * alpha), 0.f);
		const float cos_theta = pow(U2, 1.f / (phong_exponent + 2.f));
		const float r = sqrt(1.f / (cos_theta * cos_theta) - 1.f);
		const vec3 N = normalize(vec3(r*cosf(phi), r*sinf(phi), 1.0f));
		const vec3 L = -V + 2.0f * N * dot(N, V);
		return L;
	}

};

#endif
