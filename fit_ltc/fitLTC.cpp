// fitLTC.cpp : Defines the entry point for the console application.
//
#include <glm/glm.hpp>
using namespace glm;

#include <algorithm>
#include <fstream>
#include <iomanip>

#include "LTC.h"
#include "brdf.h"
#include "brdf_ggx.h"
#include "brdf_beckmann.h"
#include "brdf_disneyDiffuse.h"
#include "brdf_blinn_phong.h"

#include "nelder_mead.h"

#include "export.h"
#include "plot.h"

// size of precomputed table (theta, alpha)
const int N = 128;
// number of samples used to compute the error during fitting
const int Nsample = 50;
// minimal roughness (avoid singularities)
const float MIN_ALPHA = 0.0001f;


// compute the norm (albedo) of the BRDF
float computeNorm(const Brdf& brdf, const vec3& V, const float alpha)
{
	float norm = 0.0;

	for(int j = 0 ; j < Nsample ; ++j)
	for(int i = 0 ; i < Nsample ; ++i)
	{
		const float U1 = (i+0.5f)/(float)Nsample;
		const float U2 = (j+0.5f)/(float)Nsample;

		// sample
		const vec3 L = brdf.sample(V, alpha, U1, U2);

		// eval
		float pdf;
		float eval = brdf.eval(V, L, alpha, pdf);

		// accumulate
		norm += (pdf > 0) ? eval / pdf : 0.0f;
	}

	return norm / (float)(Nsample*Nsample);
}

// compute the average direction of the BRDF
vec3 computeAverageDir(const Brdf& brdf, const vec3& V, const float alpha)
{
	vec3 averageDir = vec3(0,0,0);

	for(int j = 0 ; j < Nsample ; ++j)
	for(int i = 0 ; i < Nsample ; ++i)
	{
		const float U1 = (i+0.5f)/(float)Nsample;
		const float U2 = (j+0.5f)/(float)Nsample;

		// sample
		const vec3 L = brdf.sample(V, alpha, U1, U2);

		// eval
		float pdf;
		float eval = brdf.eval(V, L, alpha, pdf);

		// accumulate
		averageDir += (pdf > 0) ? eval / pdf * L : vec3(0,0,0);
	}

	// clear y component, which should be zero with isotropic BRDFs
	averageDir.y = 0.0f;

	return normalize(averageDir);
}

// compute the error between the BRDF and the LTC
// using Multiple Importance Sampling
float computeError(const LTC& ltc, const Brdf& brdf, const vec3& V, const float alpha)
{
	double error = 0.0;

	for(int j = 0 ; j < Nsample ; ++j)
	for(int i = 0 ; i < Nsample ; ++i)
	{
		const float U1 = (i+0.5f)/(float)Nsample;
		const float U2 = (j+0.5f)/(float)Nsample;

		// importance sample LTC
		{
			// sample
			const vec3 L = ltc.sample(U1, U2);
				
			// error with MIS weight
			float pdf_brdf;
			float eval_brdf = brdf.eval(V, L, alpha, pdf_brdf);
			float eval_ltc = ltc.eval(L);
			float pdf_ltc = eval_ltc / ltc.amplitude;
			double error_ = fabsf(eval_brdf - eval_ltc);
			error_ = error_*error_*error_;
			error += error_ / (pdf_ltc + pdf_brdf);
		}

		// importance sample BRDF
		{
			// sample
			const vec3 L = brdf.sample(V, alpha, U1, U2);

			// error with MIS weight
			float pdf_brdf;
			float eval_brdf = brdf.eval(V, L, alpha, pdf_brdf);			
			float eval_ltc = ltc.eval(L);
			float pdf_ltc = eval_ltc / ltc.amplitude;
			double error_ = fabsf(eval_brdf - eval_ltc);
			error_ = error_*error_*error_;
			error += error_ / (pdf_ltc + pdf_brdf);
		}
	}

	return (float)error / (float)(Nsample*Nsample);
}

struct FitLTC
{
	FitLTC(LTC& ltc_, const Brdf& brdf, bool isotropic_, const vec3& V_, float alpha_) :
		ltc(ltc_), brdf(brdf), V(V_), alpha(alpha_), isotropic(isotropic_)
	{
	}

	void update(const float * params)
	{
		float m11 = std::max<float>(params[0], MIN_ALPHA);
		float m22 = std::max<float>(params[1], MIN_ALPHA);
		float m13 = params[2];
		float m23 = params[3];

		if(isotropic)
		{
			ltc.m11 = m11;
			ltc.m22 = m11;
			ltc.m13 = 0.0f;
			ltc.m23 = 0.0f;
		}
		else
		{
			ltc.m11 = m11;
			ltc.m22 = m22;
			ltc.m13 = m13;
			ltc.m23 = m23;
		}
		ltc.update();
	}

	float operator()(const float * params)
	{
		update(params);
		return computeError(ltc, brdf, V, alpha);
	}

	const Brdf& brdf;
	LTC& ltc;
	bool isotropic;

	const vec3& V;
	float alpha;
};

// fit brute force
// refine first guess by exploring parameter space
void fit(LTC& ltc, const Brdf& brdf, const vec3& V, const float alpha, const float epsilon = 0.05f, const bool isotropic=false)
{
	float startFit[4] = { ltc.m11, ltc.m22, ltc.m13, ltc.m23 };
	float resultFit[4];

	FitLTC fitter(ltc, brdf, isotropic, V, alpha);

	// Find best-fit LTC lobe (scale, alphax, alphay)
	float error = NelderMead<4>(resultFit, startFit, epsilon, 1e-5f, 100, fitter);

	// Update LTC with best fitting values
	fitter.update(resultFit);
}

// fit data
void fitTab(mat3 * tab, vec2 * tabAmplitude, const int N, const Brdf& brdf)
{
	LTC ltc;

	// loop over theta and alpha
	for(int a = N-1 ; a >= 0   ; --a)
	for(int t =   0 ; t <= N-1 ; ++t)
	{
		float theta = std::min<float>(1.57f, t / float(N-1) * 1.57079f);
		const vec3 V = vec3(sinf(theta),0,cosf(theta));

		// alpha = roughness^2
		float roughness = a / float(N-1);
		float alpha = std::max<float>(roughness*roughness, MIN_ALPHA);

		cout << "a = " << a << "\t t = " << t  << endl;
		cout << "alpha = " << alpha << "\t theta = " << theta << endl;
		cout << endl;

		ltc.amplitude = computeNorm(brdf, V, alpha); 
		const vec3 averageDir = computeAverageDir(brdf, V, alpha);		
		bool isotropic;

		// 1. first guess for the fit
		// init the hemisphere in which the distribution is fitted
		// if theta == 0 the lobe is rotationally symmetric and aligned with Z = (0 0 1)
		if(t == 0)
		{
			ltc.X = vec3(1,0,0);
			ltc.Y = vec3(0,1,0);
			ltc.Z = vec3(0,0,1);

			if(a == N-1) // roughness = 1
			{
				ltc.m11 = 1.0f;
				ltc.m22 = 1.0f;
			}
			else // init with roughness of previous fit
			{
				ltc.m11 = std::max<float>(tab[a+1+t*N][0][0], MIN_ALPHA);
				ltc.m22 = std::max<float>(tab[a+1+t*N][1][1], MIN_ALPHA);
			}
			
			ltc.m13 = 0;
			ltc.m23 = 0;
			ltc.update();

			isotropic = true;
		}
		// otherwise use previous configuration as first guess
		else
		{
			vec3 L = normalize(averageDir);
			vec3 T1(L.z,0,-L.x);
			vec3 T2(0,1,0);
			ltc.X = T1;
			ltc.Y = T2;
			ltc.Z = L;

			ltc.update();

			isotropic = false;
		}

		// 2. fit (explore parameter space and refine first guess)
		float epsilon = 0.05f;
		fit(ltc, brdf, V, alpha, epsilon, isotropic);

		// copy data
		tab[a + t*N] = ltc.M;
		tabAmplitude[a + t*N][0] = ltc.amplitude;
		tabAmplitude[a + t*N][1] = 0;

		// kill useless coefs in matrix and normalize
		tab[a+t*N][0][1] = 0;
		tab[a+t*N][1][0] = 0;
		tab[a+t*N][2][1] = 0;
		tab[a+t*N][1][2] = 0;
		tab[a+t*N] = 1.0f / tab[a+t*N][2][2] * tab[a+t*N];

		cout << tab[a+t*N][0][0] << "\t " << tab[a+t*N][1][0] << "\t " << tab[a+t*N][2][0] << endl;
		cout << tab[a+t*N][0][1] << "\t " << tab[a+t*N][1][1] << "\t " << tab[a+t*N][2][1] << endl;
		cout << tab[a+t*N][0][2] << "\t " << tab[a+t*N][1][2] << "\t " << tab[a+t*N][2][2] << endl;
		cout << endl;
	}
}

int main(int argc, char* argv[])
{
	// BRDF to fit
	// BrdfGGX brdf;
	// BrdfBeckmann brdf;
	// BrdfDisneyDiffuse brdf;
	BrdfBlinnPhong brdf;
	
	// allocate data
	mat3 * tab = new mat3[N*N];
	vec2 * tabAmplitude = new vec2[N*N];
	
	// fit
	fitTab(tab, tabAmplitude, N, brdf);

	// export in C, matlab and DDS
	// writeTabMatlab(tab, tabAmplitude, N);
	writeTabC(tab, tabAmplitude, N);
	// writeDDS(tab, tabAmplitude, N);

	// spherical plots
	// make_spherical_plots(brdf, tab, N);

	// delete data
	delete [] tab;
	delete [] tabAmplitude;

	return 0;
}

