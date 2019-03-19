#ifndef _EXPORT_
#define _EXPORT_

// export data in C
void writeTabC(mat3 * tab, vec2 * tabAmplitude, int N)
{
	ofstream file("results/ltc.inc");

	file << std::fixed;
	file << std::setprecision(6);

	file << "namespace ltc {" << endl;

	file << "static const int size = " << N  << ";" << endl << endl;

	file << "static const std::array<float, 9> tabM[size*size] = {" << endl;
	for(int t = 0 ; t < N ; ++t)
	for(int a = 0 ; a < N ; ++a)
	{
		file << "{";
		file << tab[a + t*N][0][0] << ", " << tab[a + t*N][1][0] << ", " << tab[a + t*N][2][0] << ", ";
		file << tab[a + t*N][0][1] << ", " << tab[a + t*N][1][1] << ", " << tab[a + t*N][2][1] << ", ";
        file << tab[a + t*N][0][2] << ", " << tab[a + t*N][1][2] << ", " << tab[a + t*N][2][2] << "}";
		if(a != N-1 || t != N-1)
			file << ", ";
		file << endl;
	}
	file << "};" << endl << endl;

	file << "static const std::array<float, 9> tabMinv[size*size] = {" << endl;
	for(int t = 0 ; t < N ; ++t)
	for(int a = 0 ; a < N ; ++a)
	{
		mat3 Minv = glm::inverse(tab[a + t*N]);

		file << "{";
		file << Minv[0][0] << ", " << Minv[1][0] << ", " << Minv[2][0] << ", ";
		file << Minv[0][1] << ", " << Minv[1][1] << ", " << Minv[2][1] << ", ";
        file << Minv[0][2] << ", " << Minv[1][2] << ", " << Minv[2][2] << "}";
		if(a != N-1 || t != N-1)
			file << ", ";
		file << endl;
	}
	file << "};" << endl << endl;

	file << "static const float tabAmplitude[size*size] = {" << endl;
	for(int t = 0 ; t < N ; ++t)
	for(int a = 0 ; a < N ; ++a)
	{
		file << tabAmplitude[a + t*N][0] << "f";
		if(a != N-1 || t != N-1)
			file << ", ";
		file << endl;
	}
	file << "};" << endl;

	file << "}" << endl;

	file.close();
}

// export data in matlab
void writeTabMatlab(mat3 * tab, vec2 * tabAmplitude, int N)
{
	ofstream file("results/ltc.mat");

	file << "# name: tabAmplitude" << endl;
	file << "# type: matrix" << endl;
	file << "# ndims: 2" << endl;
	file << " " << N << " " << N << endl;

	for(int t = 0 ; t < N ; ++t)
	{
		for(int a = 0 ; a < N ; ++a)
		{
			file << tabAmplitude[a + t*N][0] << " " ;
		}
		file << endl;
	}

	for(int row = 0 ; row<3 ; ++row)
	for(int column = 0 ; column<3 ; ++column)
	{

		file << "# name: tab" << column << row << endl;
		file << "# type: matrix" << endl;
		file << "# ndims: 2" << endl;
		file << " " << N << " " << N << endl;

		for(int t = 0 ; t < N ; ++t)
		{
			for(int a = 0 ; a < N ; ++a)
			{
				file << tab[a + t*N][column][row] << " " ;
			}
			file << endl;
		}

		file << endl;
	}

	file.close();
}

// export data in dds
#include "dds.h"

void writeDDS(mat3 * tab, vec2 * tabAmplitude, int N)
{
	float * data = new float[N*N*4];

	int n = 0;
	for (int i = 0; i < N*N; ++i, n += 4)
	{
		const mat3& m = tab[i];

		float a = m[0][0];
		float b = m[0][2];
		float c = m[1][1];
		float d = m[2][0];

		// Rescaled inverse of m:
		// a 0 b   inverse   1      0      -b
		// 0 c 0     ==>     0 (a - b*d)/c  0
		// d 0 1            -d      0       a

		// Store the variable terms
		data[n + 0] =  a;
		data[n + 1] = -b;
		data[n + 2] = (a - b*d) / c;
		data[n + 3] = -d;
	}

	SaveDDS("results/ltc_mat.dds", DDS_FORMAT_R32G32B32A32_FLOAT, sizeof(float)*4, N, N, data);
	SaveDDS("results/ltc_amp.dds", DDS_FORMAT_R32G32_FLOAT,       sizeof(float)*2, N, N, tabAmplitude);

	delete [] data;
}

void writeSphereTabC(float * tab, int N)
{
	ofstream file("results/ltc_sphere.inc");

	file << std::fixed;
	file << std::setprecision(6);

	file << "namespace ltc {" << endl;

	file << "static const int tab_sphere_size = " << N  << ";" << endl << endl;

	file << "static const float tabSphere[tab_sphere_size*tab_sphere_size] = {" << endl;
	for(int t = 0 ; t < N ; ++t)
	for(int a = 0 ; a < N ; ++a)
	{
		file << tab[a + t*N];
		if(a != N-1 || t != N-1)
			file << ", ";
		file << endl;
	}
	file << "};" << endl << endl;

	file << "}" << endl;

	file.close();
}

#endif
