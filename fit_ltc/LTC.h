#ifndef _LTC_
#define _LTC_


#include <glm/glm.hpp>
using namespace glm;

#include <iostream>
using namespace std;

struct LTC {

	// lobe amplitude
	float amplitude;

	// parametric representation
	float m11, m22, m13, m23;
	vec3 X, Y, Z;

	// matrix representation
	mat3 M;
	mat3 invM;
	float detM;

	LTC()
	{
		amplitude = 1;
		m11 = 1;
		m22 = 1;
		m13 = 0;
		m23 = 0;
		X = vec3(1,0,0);
		Y = vec3(0,1,0);
		Z = vec3(0,0,1);
		update();
	}

	void copy(const LTC& ltc)
	{
		this->amplitude = ltc.amplitude;
		this->m11 = ltc.m11;
		this->m22 = ltc.m22;
		this->m13 = ltc.m13;
		this->m23 = ltc.m23;
		this->X = ltc.X;
		this->Y = ltc.Y;
		this->Z = ltc.Z;
		this->M = ltc.M;
		this->invM = ltc.invM;
		this->detM = ltc.detM;
	}

	void update() // compute matrix from parameters
	{
		M = mat3(X, Y, Z) *
			mat3(m11, 0, 0,
				0, m22, 0,
				m13, m23, 1);
		invM = inverse(M);
		detM = abs(glm::determinant(M));
	}

	void update2()
	{
		invM = inverse(M);
		detM = abs(glm::determinant(M));
	}

	float eval(const vec3& L) const
	{
		vec3 Loriginal = normalize(invM * L);
		vec3 L_ = M * Loriginal;

		float l = length(L_);
		float Jacobian = detM / (l*l*l);

		float D = 1.0f / 3.14159f * glm::max<float>(0.0f, Loriginal.z); 
		
		float res = amplitude * D / Jacobian;
		return res;
	}

	vec3 sample(const float U1, const float U2) const
	{
		const float theta = acosf(sqrtf(U1));
		const float phi = 2.0f*3.14159f * U2;
		const vec3 L = normalize(M * vec3(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta)));
		return L;
	}

	void testNormalization() const
	{
		double sum = 0;
		float dtheta = 0.005f;
		float dphi = 0.005f;
		for(float theta = 0.0f ; theta <= 3.14159f ; theta+=dtheta)
		for(float phi = 0.0f ; phi <= 2.0f * 3.14159f ; phi+=dphi)
		{
			vec3 L(cosf(phi)*sinf(theta), sinf(phi)*sinf(theta), cosf(theta));

			sum += sinf(theta) * eval(L);
		}
		sum *= dtheta * dphi;
		cout << "LTC normalization test: " << sum << endl;
		cout << "LTC normalization expected: " << amplitude << endl;
	}
};




/*
#include "ltc_alpha.inc"

 // build orthonormal basis (Building an Orthonormal Basis from a 3D Unit Vector Without Normalization, [Frisvad2012])
 void buildOrthonormalBasis(vec3& omega_1, vec3& omega_2, const vec3& omega_3)
{
	if(omega_3.z < -0.9999999f)
	{
	   omega_1 = vec3 ( 0.0f , -1.0f , 0.0f );
	   omega_2 = vec3 ( -1.0f , 0.0f , 0.0f );
	} else {
	   const float a = 1.0f /(1.0f + omega_3.z );
	   const float b = -omega_3.x*omega_3 .y*a ;
	   omega_1 = vec3 (1.0f - omega_3.x*omega_3. x*a , b , -omega_3.x );
	   omega_2 = vec3 (b , 1.0f - omega_3.y*omega_3.y*a , -omega_3.y );
	}
}

mat3 moment2M(const mat3& Sigma, const vec3& average)
{
	vec3 T1, T2;
	buildOrthonormalBasis(T1, T2, average);

	const float var1 = dot(T1, Sigma * T1);
	const float var2 = dot(T2, Sigma * T2);
	const float c12 = dot(T1, Sigma * T2);

	mat2 Sigma12(var1, c12, c12, var2);
	vec2 eigen1(1,1);
	eigen1 = normalize(Sigma12 * Sigma12 * Sigma12 * Sigma12 * Sigma12 * Sigma12 * Sigma12 * eigen1);
	vec2 eigen2(-eigen1.y, eigen1.x);

	vec3 Teigen1 = eigen1.x * T1 + eigen1.y * T2;
	vec3 Teigen2 = eigen2.x * T1 + eigen2.y * T2;

	float sigma1 = sqrtf(dot(Teigen1, Sigma * Teigen1));
	float sigma2 = sqrtf(dot(Teigen2, Sigma * Teigen2));

	int index1 = std::min(size, std::max(0, (int)floorf(size * sigma1)));
	int index2 = std::min(size, std::max(0, (int)floorf(size * sigma2)));

	return mat3(Teigen1, Teigen2, average) * mat3(tabAlpha[index1+size*index2].x, 0, 0, 0, tabAlpha[index1+size*index2].y, 0, 0, 0, 1.0f);
}
*/

#endif



