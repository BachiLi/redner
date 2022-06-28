
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from NVIDIA Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/**
 * @file   optixu_matrix_namespace.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * OptiX public API Reference - Public Matrix namespace
 */

#ifndef __optixu_optixu_matrix_namespace_h__
#define __optixu_optixu_matrix_namespace_h__

#include "optixu_math_namespace.h"

// __forceinline__ works in CUDA, VS, and with gcc.  Leave it as a macro in case
// we need to make this per-platform or we want to switch off inlining globally.
#ifndef OPTIXU_INLINE
#  define OPTIXU_INLINE_DEFINED 1
#  define OPTIXU_INLINE __forceinline__
#endif // OPTIXU_INLINE

#define RT_MATRIX_ACCESS(m,i,j) m[i*N+j]
#define RT_MAT_DECL template <unsigned int M, unsigned int N>

namespace optix {

  template <int DIM> struct VectorDim { };
  template <> struct VectorDim<2> { typedef float2 VectorType; };
  template <> struct VectorDim<3> { typedef float3 VectorType; };
  template <> struct VectorDim<4> { typedef float4 VectorType; };


  template <unsigned int M, unsigned int N> class Matrix;

   template <unsigned int M> OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,M>& operator*=(Matrix<M,M>& m1, const Matrix<M,M>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE bool         operator==(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE bool         operator!=(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>& operator-=(Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>& operator+=(Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>& operator*=(Matrix<M,N>& m1, float f);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>&operator/=(Matrix<M,N>& m1, float f);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N> operator-(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N> operator+(const Matrix<M,N>& m1, const Matrix<M,N>& m2);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N> operator/(const Matrix<M,N>& m, float f);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N> operator*(const Matrix<M,N>& m, float f);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N> operator*(float f, const Matrix<M,N>& m);
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE typename Matrix<M,N>::floatM operator*(const Matrix<M,N>& m, const typename Matrix<M,N>::floatN& v );
   RT_MAT_DECL OPTIXU_INLINE RT_HOSTDEVICE typename Matrix<M,N>::floatN operator*(const typename Matrix<M,N>::floatM& v, const Matrix<M,N>& m);
   template<unsigned int M, unsigned int N, unsigned int R> OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,R> operator*(const Matrix<M,N>& m1, const Matrix<N,R>& m2);


  // Partial specializations to make matrix vector multiplication more efficient
  template <unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float2 operator*(const Matrix<2,N>& m, const typename Matrix<2,N>::floatN& vec );
  template <unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float3 operator*(const Matrix<3,N>& m, const typename Matrix<3,N>::floatN& vec );
  template <unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const Matrix<4,N>& m, const typename Matrix<4,N>::floatN& vec );
  OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const Matrix<4,4>& m, const float4& vec );

  /**
  * @brief A matrix with M rows and N columns
  *
  * @ingroup CUDACTypes
  *
  * <B>Description</B>
  *
  * @ref Matrix provides a utility class for small-dimension floating-point
  * matrices, such as transformation matrices.  @ref Matrix may also be useful
  * in other computation and can be used in both host and device code.
  * Typedefs are provided for 2x2 through 4x4 matrices.
  *
  * <B>History</B>
  *
  * @ref Matrix was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * \a rtVariableSetMatrix*
  *
  */
  template <unsigned int M, unsigned int N>
  class Matrix
  {
  public:
    typedef typename VectorDim<N>::VectorType  floatN; /// A row of the matrix
    typedef typename VectorDim<M>::VectorType  floatM; /// A column of the matrix

	/** Create an uninitialized matrix */
	RT_HOSTDEVICE              Matrix();

	/** Create a matrix from the specified float array */
	RT_HOSTDEVICE explicit     Matrix( const float data[M*N] ) { for(unsigned int i = 0; i < M*N; ++i) m_data[i] = data[i]; }

	/** Copy the matrix */
	RT_HOSTDEVICE              Matrix( const Matrix& m );

	/** Assignment operator */
	RT_HOSTDEVICE Matrix&      operator=( const Matrix& b );

	/** Access the specified element 0..N*M-1  */
	RT_HOSTDEVICE float        operator[]( unsigned int i )const { return m_data[i]; }

	/** Access the specified element 0..N*M-1  */
	RT_HOSTDEVICE float&       operator[]( unsigned int i )      { return m_data[i]; }

	/** Access the specified row 0..M.  Returns float, float2, float3 or float4 depending on the matrix size  */
	RT_HOSTDEVICE floatN       getRow( unsigned int m )const;

	/** Access the specified column 0..N.  Returns float, float2, float3 or float4 depending on the matrix size */
	RT_HOSTDEVICE floatM       getCol( unsigned int n )const;

	/** Returns a pointer to the internal data array.  The data array is stored in row-major order. */
	RT_HOSTDEVICE float*       getData();

	/** Returns a const pointer to the internal data array.  The data array is stored in row-major order. */
	RT_HOSTDEVICE const float* getData()const;

	/** Assign the specified row 0..M.  Takes a float, float2, float3 or float4 depending on the matrix size */
	RT_HOSTDEVICE void         setRow( unsigned int m, const floatN &r );

	/** Assign the specified column 0..N.  Takes a float, float2, float3 or float4 depending on the matrix size */
	RT_HOSTDEVICE void         setCol( unsigned int n, const floatM &c );

	/** Returns the transpose of the matrix */
	RT_HOSTDEVICE Matrix<N,M>         transpose() const;

	/** Returns the inverse of the matrix */
	RT_HOSTDEVICE Matrix<4,4>         inverse() const;

	/** Returns the determinant of the matrix */
	RT_HOSTDEVICE float               det() const;

	/** Returns a rotation matrix */
	RT_HOSTDEVICE static Matrix<4,4>  rotate(const float radians, const float3& axis);

	/** Returns a translation matrix */
	RT_HOSTDEVICE static Matrix<4,4>  translate(const float3& vec);

	/** Returns a scale matrix */
	RT_HOSTDEVICE static Matrix<4,4>  scale(const float3& vec);

    /** Creates a matrix from an ONB and center point */
	RT_HOSTDEVICE static Matrix<4,4>  fromBasis( const float3& u, const float3& v, const float3& w, const float3& c );

	/** Returns the identity matrix */
	RT_HOSTDEVICE static Matrix<N,N>  identity();

	/** Ordered comparison operator so that the matrix can be used in an STL container */
	RT_HOSTDEVICE bool         operator<( const Matrix<M, N>& rhs ) const;
  private:
	  /** The data array is stored in row-major order */
	  float m_data[M*N];
  };



  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>::Matrix()
  {
  }

  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>::Matrix( const Matrix<M,N>& m )
  {
    for(unsigned int i = 0; i < M*N; ++i)
      m_data[i] = m[i];
  }

  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<M,N>&  Matrix<M,N>::operator=( const Matrix& b )
  {
    for(unsigned int i = 0; i < M*N; ++i)
      m_data[i] = b[i];
    return *this;
  }


  /*
  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float Matrix<M,N>::operator[]( unsigned int i )const
  {
  return m_data[i];
  }


  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float& Matrix<M,N>::operator[]( unsigned int i )
  {
  return m_data[i];
  }
  */

  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE typename Matrix<M,N>::floatN Matrix<M,N>::getRow( unsigned int m )const
  {
    typename Matrix<M,N>::floatN temp;
    float* v = reinterpret_cast<float*>( &temp );
    const float* row = &( m_data[m*N] );
    for(unsigned int i = 0; i < N; ++i)
      v[i] = row[i];

    return temp;
  }


  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE typename Matrix<M,N>::floatM Matrix<M,N>::getCol( unsigned int n )const
  {
    typename Matrix<M,N>::floatM temp;
    float* v = reinterpret_cast<float*>( &temp );
    for ( unsigned int i = 0; i < M; ++i )
      v[i] = RT_MATRIX_ACCESS( m_data, i, n );

    return temp;
  }


  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float* Matrix<M,N>::getData()
  {
    return m_data;
  }


  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE const float* Matrix<M,N>::getData() const
  {
    return m_data;
  }


  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE void Matrix<M,N>::setRow( unsigned int m, const typename Matrix<M,N>::floatN &r )
  {
    const float* v = reinterpret_cast<const float*>( &r );
    float* row = &( m_data[m*N] );
    for(unsigned int i = 0; i < N; ++i)
      row[i] = v[i];
  }


  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE void Matrix<M,N>::setCol( unsigned int n, const typename Matrix<M,N>::floatM &c )
  {
    const float* v = reinterpret_cast<const float*>( &c );
    for ( unsigned int i = 0; i < M; ++i )
      RT_MATRIX_ACCESS( m_data, i, n ) = v[i];
  }


  // Compare two matrices using exact float comparison
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE bool operator==(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      if ( m1[i] != m2[i] ) return false;
    return true;
  }

  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE bool operator!=(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      if ( m1[i] != m2[i] ) return true;
    return false;
  }

  // Subtract two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator-(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    Matrix<M,N> temp( m1 );
    temp -= m2;
    return temp;
  }


  // Subtract two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator-=(Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m1[i] -= m2[i];
    return m1;
  }


  // Add two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator+(const Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    Matrix<M,N> temp( m1 );
    temp += m2;
    return temp;
  }


  // Add two matrices of the same size.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator+=(Matrix<M,N>& m1, const Matrix<M,N>& m2)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m1[i] += m2[i];
    return m1;
  }


  // Multiply two compatible matrices.
  template<unsigned int M, unsigned int N, unsigned int R>
  RT_HOSTDEVICE Matrix<M,R> operator*( const Matrix<M,N>& m1, const Matrix<N,R>& m2)
  {
    Matrix<M,R> temp;

    for ( unsigned int i = 0; i < M; ++i ) {
      for ( unsigned int j = 0; j < R; ++j ) {
        float sum = 0.0f;
        for ( unsigned int k = 0; k < N; ++k ) {
          float ik = m1[ i*N+k ];
          float kj = m2[ k*R+j ];
          sum += ik * kj;
        }
        temp[i*R+j] = sum;
      }
    }
    return temp;
  }


  // Multiply two compatible matrices.
  template<unsigned int M>
  RT_HOSTDEVICE Matrix<M,M>& operator*=(Matrix<M,M>& m1, const Matrix<M,M>& m2)
  {
    m1 = m1*m2;
    return m1;
  }


  // Multiply matrix by vector
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE typename Matrix<M,N>::floatM operator*(const Matrix<M,N>& m, const typename Matrix<M,N>::floatN& vec )
  {
    typename Matrix<M,N>::floatM temp;
    float* t = reinterpret_cast<float*>( &temp );
    const float* v = reinterpret_cast<const float*>( &vec );

    for (unsigned int i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (unsigned int j = 0; j < N; ++j) {
        sum += RT_MATRIX_ACCESS( m, i, j ) * v[j];
      }
      t[i] = sum;
    }

    return temp;
  }

  // Multiply matrix2xN by floatN
  template<unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float2 operator*(const Matrix<2,N>& m, const typename Matrix<2,N>::floatN& vec )
  {
    float2 temp = { 0.0f, 0.0f };
    const float* v = reinterpret_cast<const float*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix3xN by floatN
  template<unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float3 operator*(const Matrix<3,N>& m, const typename Matrix<3,N>::floatN& vec )
  {
    float3 temp = { 0.0f, 0.0f, 0.0f };
    const float* v = reinterpret_cast<const float*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.z += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix4xN by floatN
  template<unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const Matrix<4,N>& m, const typename Matrix<4,N>::floatN& vec )
  {
    float4 temp = { 0.0f, 0.0f, 0.0f, 0.0f };

    const float* v = reinterpret_cast<const float*>( &vec );

    int index = 0;
    for (unsigned int j = 0; j < N; ++j)
      temp.x += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.y += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.z += m[index++] * v[j];

    for (unsigned int j = 0; j < N; ++j)
      temp.w += m[index++] * v[j];

    return temp;
  }

  // Multiply matrix4x4 by float4
  OPTIXU_INLINE RT_HOSTDEVICE float4 operator*(const Matrix<4,4>& m, const float4& vec )
  {
    float4 temp;
    temp.x  = m[ 0] * vec.x +
              m[ 1] * vec.y +
              m[ 2] * vec.z +
              m[ 3] * vec.w;
    temp.y  = m[ 4] * vec.x +
              m[ 5] * vec.y +
              m[ 6] * vec.z +
              m[ 7] * vec.w;
    temp.z  = m[ 8] * vec.x +
              m[ 9] * vec.y +
              m[10] * vec.z +
              m[11] * vec.w;
    temp.w  = m[12] * vec.x +
              m[13] * vec.y +
              m[14] * vec.z +
              m[15] * vec.w;

    return temp;
  }

  // Multiply vector by matrix
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE typename Matrix<M,N>::floatN operator*(const typename Matrix<M,N>::floatM& vec, const Matrix<M,N>& m)
  {
    typename Matrix<M,N>::floatN  temp;
    float* t = reinterpret_cast<float*>( &temp );
    const float* v = reinterpret_cast<const float*>( &vec);

    for (unsigned int i = 0; i < N; ++i) {
      float sum = 0.0f;
      for (unsigned int j = 0; j < M; ++j) {
        sum += v[j] * RT_MATRIX_ACCESS( m, j, i ) ;
      }
      t[i] = sum;
    }

    return temp;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator*(const Matrix<M,N>& m, float f)
  {
    Matrix<M,N> temp( m );
    temp *= f;
    return temp;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator*=(Matrix<M,N>& m, float f)
  {
    for ( unsigned int i = 0; i < M*N; ++i )
      m[i] *= f;
    return m;
  }


  // Multply matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>  operator*(float f, const Matrix<M,N>& m)
  {
    Matrix<M,N> temp;

    for ( unsigned int i = 0; i < M*N; ++i )
      temp[i] = m[i]*f;

    return temp;
  }


  // Divide matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N> operator/(const Matrix<M,N>& m, float f)
  {
    Matrix<M,N> temp( m );
    temp /= f;
    return temp;
  }


  // Divide matrix by a scalar.
  template<unsigned int M, unsigned int N>
  RT_HOSTDEVICE Matrix<M,N>& operator/=(Matrix<M,N>& m, float f)
  {
    float inv_f = 1.0f / f;
    for ( unsigned int i = 0; i < M*N; ++i )
      m[i] *= inv_f;
    return m;
  }

  // Returns the transpose of the matrix.
  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<N,M> Matrix<M,N>::transpose() const
  {
    Matrix<N,M> ret;
    for( unsigned int row = 0; row < M; ++row )
      for( unsigned int col = 0; col < N; ++col )
        ret[col*M+row] = m_data[row*N+col];
    return ret;
  }

  // Returns the determinant of the matrix.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE float Matrix<3,3>::det() const
  {
    const float* m   = m_data;
    float d = m[0]*m[4]*m[8] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7]
      - m[0]*m[5]*m[7] - m[1]*m[3]*m[8] - m[2]*m[4]*m[6];
    return d;
  }

  // Returns the determinant of the matrix.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE float Matrix<4,4>::det() const
  {
    const float* m   = m_data;
    float d =
      m[0]*m[5]*m[10]*m[15]-
      m[0]*m[5]*m[11]*m[14]+m[0]*m[9]*m[14]*m[7]-
      m[0]*m[9]*m[6]*m[15]+m[0]*m[13]*m[6]*m[11]-
      m[0]*m[13]*m[10]*m[7]-m[4]*m[1]*m[10]*m[15]+m[4]*m[1]*m[11]*m[14]-
      m[4]*m[9]*m[14]*m[3]+m[4]*m[9]*m[2]*m[15]-
      m[4]*m[13]*m[2]*m[11]+m[4]*m[13]*m[10]*m[3]+m[8]*m[1]*m[6]*m[15]-
      m[8]*m[1]*m[14]*m[7]+m[8]*m[5]*m[14]*m[3]-
      m[8]*m[5]*m[2]*m[15]+m[8]*m[13]*m[2]*m[7]-
      m[8]*m[13]*m[6]*m[3]-
      m[12]*m[1]*m[6]*m[11]+m[12]*m[1]*m[10]*m[7]-
      m[12]*m[5]*m[10]*m[3]+m[12]*m[5]*m[2]*m[11]-
      m[12]*m[9]*m[2]*m[7]+m[12]*m[9]*m[6]*m[3];
    return d;
  }

  // Returns the inverse of the matrix.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<4,4> Matrix<4,4>::inverse() const
  {
    Matrix<4,4> dst;
    const float* m   = m_data;
    const float d = 1.0f / det();

    dst[0]  = d * (m[5] * (m[10] * m[15] - m[14] * m[11]) + m[9] * (m[14] * m[7] - m[6] * m[15]) + m[13] * (m[6] * m[11] - m[10] * m[7]));
    dst[4]  = d * (m[6] * (m[8] * m[15] - m[12] * m[11]) + m[10] * (m[12] * m[7] - m[4] * m[15]) + m[14] * (m[4] * m[11] - m[8] * m[7]));
    dst[8]  = d * (m[7] * (m[8] * m[13] - m[12] * m[9]) + m[11] * (m[12] * m[5] - m[4] * m[13]) + m[15] * (m[4] * m[9] - m[8] * m[5]));
    dst[12] = d * (m[4] * (m[13] * m[10] - m[9] * m[14]) + m[8] * (m[5] * m[14] - m[13] * m[6]) + m[12] * (m[9] * m[6] - m[5] * m[10]));
    dst[1]  = d * (m[9] * (m[2] * m[15] - m[14] * m[3]) + m[13] * (m[10] * m[3] - m[2] * m[11]) + m[1] * (m[14] * m[11] - m[10] * m[15]));
    dst[5]  = d * (m[10] * (m[0] * m[15] - m[12] * m[3]) + m[14] * (m[8] * m[3] - m[0] * m[11]) + m[2] * (m[12] * m[11] - m[8] * m[15]));
    dst[9]  = d * (m[11] * (m[0] * m[13] - m[12] * m[1]) + m[15] * (m[8] * m[1] - m[0] * m[9]) + m[3] * (m[12] * m[9] - m[8] * m[13]));
    dst[13] = d * (m[8] * (m[13] * m[2] - m[1] * m[14]) + m[12] * (m[1] * m[10] - m[9] * m[2]) + m[0] * (m[9] * m[14] - m[13] * m[10]));
    dst[2]  = d * (m[13] * (m[2] * m[7] - m[6] * m[3]) + m[1] * (m[6] * m[15] - m[14] * m[7]) + m[5] * (m[14] * m[3] - m[2] * m[15]));
    dst[6]  = d * (m[14] * (m[0] * m[7] - m[4] * m[3]) + m[2] * (m[4] * m[15] - m[12] * m[7]) + m[6] * (m[12] * m[3] - m[0] * m[15]));
    dst[10] = d * (m[15] * (m[0] * m[5] - m[4] * m[1]) + m[3] * (m[4] * m[13] - m[12] * m[5]) + m[7] * (m[12] * m[1] - m[0] * m[13]));
    dst[14] = d * (m[12] * (m[5] * m[2] - m[1] * m[6]) + m[0] * (m[13] * m[6] - m[5] * m[14]) + m[4] * (m[1] * m[14] - m[13] * m[2]));
    dst[3]  = d * (m[1] * (m[10] * m[7] - m[6] * m[11]) + m[5] * (m[2] * m[11] - m[10] * m[3]) + m[9] * (m[6] * m[3] - m[2] * m[7]));
    dst[7]  = d * (m[2] * (m[8] * m[7] - m[4] * m[11]) + m[6] * (m[0] * m[11] - m[8] * m[3]) + m[10] * (m[4] * m[3] - m[0] * m[7]));
    dst[11] = d * (m[3] * (m[8] * m[5] - m[4] * m[9]) + m[7] * (m[0] * m[9] - m[8] * m[1]) + m[11] * (m[4] * m[1] - m[0] * m[5]));
    dst[15] = d * (m[0] * (m[5] * m[10] - m[9] * m[6]) + m[4] * (m[9] * m[2] - m[1] * m[10]) + m[8] * (m[1] * m[6] - m[5] * m[2]));
    return dst;
  }

  // Returns a rotation matrix.
  // This is a static member.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<4,4> Matrix<4,4>::rotate(const float radians, const float3& axis)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    float *m = Mat.getData();

    // NOTE: Element 0,1 is wrong in Foley and Van Dam, Pg 227!
    float sintheta=sinf(radians);
    float costheta=cosf(radians);
    float ux=axis.x;
    float uy=axis.y;
    float uz=axis.z;
    m[0*4+0]=ux*ux+costheta*(1-ux*ux);
    m[0*4+1]=ux*uy*(1-costheta)-uz*sintheta;
    m[0*4+2]=uz*ux*(1-costheta)+uy*sintheta;
    m[0*4+3]=0;

    m[1*4+0]=ux*uy*(1-costheta)+uz*sintheta;
    m[1*4+1]=uy*uy+costheta*(1-uy*uy);
    m[1*4+2]=uy*uz*(1-costheta)-ux*sintheta;
    m[1*4+3]=0;

    m[2*4+0]=uz*ux*(1-costheta)-uy*sintheta;
    m[2*4+1]=uy*uz*(1-costheta)+ux*sintheta;
    m[2*4+2]=uz*uz+costheta*(1-uz*uz);
    m[2*4+3]=0;

    m[3*4+0]=0;
    m[3*4+1]=0;
    m[3*4+2]=0;
    m[3*4+3]=1;

    return Matrix<4,4>( m );
  }

  // Returns a translation matrix.
  // This is a static member.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<4,4> Matrix<4,4>::translate(const float3& vec)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    float *m = Mat.getData();

    m[3] = vec.x;
    m[7] = vec.y;
    m[11]= vec.z;

    return Matrix<4,4>( m );
  }

  // Returns a scale matrix.
  // This is a static member.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<4,4> Matrix<4,4>::scale(const float3& vec)
  {
    Matrix<4,4> Mat = Matrix<4,4>::identity();
    float *m = Mat.getData();

    m[0] = vec.x;
    m[5] = vec.y;
    m[10]= vec.z;

    return Matrix<4,4>( m );
  }


  // This is a static member.
  template<>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<4,4>  Matrix<4,4>::fromBasis( const float3& u, const float3& v, const float3& w, const float3& c )
  {
    float m[16];                                                                 
    m[ 0] = u.x;                                                                  
    m[ 1] = v.x;                                                                  
    m[ 2] = w.x;                                                                  
    m[ 3] = c.x;                                                                  

    m[ 4] = u.y;                                                                  
    m[ 5] = v.y;                                                                  
    m[ 6] = w.y;                                                                  
    m[ 7] = c.y;                                                                  

    m[ 8] = u.z;                                                                  
    m[ 9] = v.z;                                                                  
    m[10] = w.z;                                                                 
    m[11] = c.z;                                                                 

    m[12] = 0.0f;                                                                
    m[13] = 0.0f;                                                                
    m[14] = 0.0f;                                                                
    m[15] = 1.0f;                                                                

    return Matrix<4,4>( m );      
  }

  // Returns the identity matrix.
  // This is a static member.
  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE Matrix<N,N> Matrix<M,N>::identity()
  {
    float temp[N*N];
    for(unsigned int i = 0; i < N*N; ++i)
      temp[i] = 0;
    for( unsigned int i = 0; i < N; ++i )
      RT_MATRIX_ACCESS( temp,i,i ) = 1.0f;
    return Matrix<N,N>( temp );
  }

  // Ordered comparison operator so that the matrix can be used in an STL container.
  template<unsigned int M, unsigned int N>
  OPTIXU_INLINE RT_HOSTDEVICE bool Matrix<M,N>::operator<( const Matrix<M, N>& rhs ) const
  {
    for( unsigned int i = 0; i < N*M; ++i ) {
      if( m_data[i] < rhs[i] )
        return true;
      else if( m_data[i] > rhs[i] )
        return false;
    }
    return false;
  }

  typedef Matrix<2, 2> Matrix2x2;
  typedef Matrix<2, 3> Matrix2x3;
  typedef Matrix<2, 4> Matrix2x4;
  typedef Matrix<3, 2> Matrix3x2;
  typedef Matrix<3, 3> Matrix3x3;
  typedef Matrix<3, 4> Matrix3x4;
  typedef Matrix<4, 2> Matrix4x2;
  typedef Matrix<4, 3> Matrix4x3;
  typedef Matrix<4, 4> Matrix4x4;


  OPTIXU_INLINE RT_HOSTDEVICE Matrix<3,3> make_matrix3x3(const Matrix<4,4> &matrix)
  {
    Matrix<3,3> Mat;
    float *m = Mat.getData();
    const float *m4x4 = matrix.getData();

    m[0*3+0]=m4x4[0*4+0];
    m[0*3+1]=m4x4[0*4+1];
    m[0*3+2]=m4x4[0*4+2];

    m[1*3+0]=m4x4[1*4+0];
    m[1*3+1]=m4x4[1*4+1];
    m[1*3+2]=m4x4[1*4+2];

    m[2*3+0]=m4x4[2*4+0];
    m[2*3+1]=m4x4[2*4+1];
    m[2*3+2]=m4x4[2*4+2];

    return Mat;
  }

} // end namespace optix

#undef RT_MATRIX_ACCESS
#undef RT_MAT_DECL

#ifdef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE_DEFINED
#  undef OPTIXU_INLINE
#endif

#endif //  __optixu_optixu_matrix_namespace_h__
