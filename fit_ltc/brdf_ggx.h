#ifndef _BRDF_GGX_
#define _BRDF_GGX_

#include "brdf.h"

inline auto hypot2(float a, float b) {
    if (fabs(a) > fabs(b)) {
        auto ratio = b/a;
        return fabs(a) * sqrt(1.f + ratio * ratio);
    } else if (b != 0.f) {
        auto ratio = a/b;
        return fabs(b) * sqrt(1.f + ratio * ratio);
    }
    return 0.f;
}

class BrdfGGX : public Brdf
{
public:

	virtual float eval(const vec3& V, const vec3& L, const float alpha, float& pdf) const
	{
        if(V.z <= 0)
        {
            pdf = 0;
            return 0;
        }

        // masking
        const float a_V = 1.0f / alpha / tanf(acosf(V.z));
        const float LambdaV = (V.z<1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f/a_V/a_V)) : 0.0f;
        const float G1 = 1.0f / (1.0f + LambdaV);

        // shadowing
        float G2;
        if(L.z <= 0.0f)
            G2 = 0;
        else
        {
            const float a_L = 1.0f / alpha / tanf(acosf(L.z));
            const float LambdaL = (L.z<1.0f) ? 0.5f * (-1.0f + sqrtf(1.0f + 1.0f/a_L/a_L)) : 0.0f;
            G2 = 1.0f / (1.0f + LambdaV + LambdaL);
        }

        // D
        const vec3 H = normalize(V+L);
        const float slopex = H.x/H.z;
        const float slopey = H.y/H.z;
        float D = 1.0f / (1.0f + (slopex*slopex+slopey*slopey)/alpha/alpha);
        D = D*D;
        D = D / (3.14159f * alpha * alpha * H.z*H.z*H.z*H.z);

        pdf = fabsf(D * H.z / 4.0f / dot(V,H));
        float res = D * G2 / 4.0f / V.z;

        return res;
		// if(V.z <= 0)
		// {
		// 	pdf = 0;
		// 	return 0;
		// }

		// const vec3 H = normalize(V+L);

  //       auto project_roughness = [&](const auto &v) {
  //           auto cos_theta = v.z;
  //           auto sin_theta_sq = 1.f - cos_theta * cos_theta;
  //           if (sin_theta_sq <= 0.f) {
  //               return alpha;
  //           }
  //           auto inv_sin_theta_sq = 1.f / sin_theta_sq;
  //           auto cos_phi_2 = v.x * v.x * inv_sin_theta_sq;
  //           auto sin_phi_2 = v.y * v.y * inv_sin_theta_sq;
  //           return sqrt(cos_phi_2 + sin_phi_2) * alpha;
  //       };
  //       auto smithG1 = [&](const auto &v) {
  //           auto cos_theta = v.z;
  //           if (dot(v, H) * cos_theta <= 0) {
  //               return 0.f;
  //           }
  //           // tan^2 + 1 = 1/cos^2
  //           auto tan_theta = sqrt(1.f / (cos_theta * cos_theta) - 1.f);
  //           if (tan_theta == 0.0f) {
  //               return 1.f;
  //           }
  //           auto root = project_roughness(v) * tan_theta;
  //           return 2.0f / (1.0f + hypot2(1.0f, root));
  //       };
  //       auto G = smithG1(V) * smithG1(L);

		// // D
		// const float slopex = H.x/H.z;
		// const float slopey = H.y/H.z;
		// float D = 1.0f / (1.0f + (slopex*slopex+slopey*slopey)/alpha/alpha);
		// D = D*D;
		// D = D / (float(M_PI) * alpha * alpha * H.z*H.z*H.z*H.z);

		// pdf = fabsf(D * H.z / 4.0f / dot(V,H));
		// float res = D * G / 4.0f / V.z;

		// return res;
	}

	virtual vec3 sample(const vec3& V, const float alpha, const float U1, const float U2) const
	{
		const float phi = 2.0f*3.14159f * U1;
		const float r = alpha*sqrtf(U2/(1.0f-U2));
		const vec3 N = normalize(vec3(r*cosf(phi), r*sinf(phi), 1.0f));
		const vec3 L = -V + 2.0f * N * dot(N, V);
		return L;
	}

};


#endif
