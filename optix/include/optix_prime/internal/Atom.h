
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

#ifndef __optix_optix_prime_atom_h__
#define __optix_optix_prime_atom_h__

#if defined(_MSC_VER)
#  define PRIME_ATOM32_MSC
#  include <math.h>
#  include <intrin.h>
#  pragma intrinsic (_InterlockedExchangeAdd)
#  pragma intrinsic (_InterlockedCompareExchange)
typedef unsigned int atomic_word;
#elif (defined(__GNUC__) && defined(__powerpc64__)) || defined(__ICC)
#  include <ext/atomicity.h>
typedef int atomic_word;
#else
#  define PRIME_ATOM32_GCC
typedef unsigned int atomic_word;
#endif


namespace optix {
  namespace prime {
  
    /// @cond

    /// 32-bit unsigned counter with atomic increments, and decrements.
    class Atom32
    {
    public:
      /// Constructor initializes the counter to \c val.
      Atom32(const unsigned int val) : m_atom(val) {}

      /// Increments counter by one.
      unsigned int operator++();

      /// Decrements counter by one.
      unsigned int operator--();

    private:
      volatile atomic_word m_atom;
    };

#if defined(PRIME_ATOM32_MSC)

    //
    // operator++
    //
    inline unsigned int Atom32::operator++() {
      return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),1L) + 1L;
    }

    //
    // operator--
    //
    inline unsigned int Atom32::operator--() {
      return _InterlockedExchangeAdd(reinterpret_cast<volatile long *>(&m_atom),-1L) - 1L;
    }

#elif defined ( __GNUC__ ) && defined ( __powerpc64__) 

    inline unsigned int Atom32::operator++() {
      return (unsigned int)__gnu_cxx::__exchange_and_add( &m_atom, 1) + 1u;
    }

    inline unsigned int Atom32::operator--() {
      return (unsigned int)__gnu_cxx::__exchange_and_add( &m_atom, -1) - 1u;
    }

#elif defined(PRIME_ATOM32_GCC) // defined(PRIME_ATOM32_X86MSC)

    //
    // operator++
    //
    inline unsigned int Atom32::operator++() {
      unsigned int retval;
      asm volatile (
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $1,%0\n"
        : "=&r" (retval), "+m" (m_atom)
        :
        : "cc"
        );
      return retval;
    }

    //
    // operator--
    //
    inline unsigned int Atom32::operator--() {
      unsigned int retval;
      asm volatile (
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $-1,%0\n"
        : "=&r" (retval), "+m" (m_atom)
        :
        : "cc"
        );
      return retval;
    }

#else
#  error One of PRIME_ATOM32_X86MSC, PRIME_ATOM32_X86GCC must be defined.
#endif

#undef PRIME_ATOM32_MSC
#undef PRIME_ATOM32_GCC

    /// @endcond

  } // namespace prime
} // namespace optix

#endif // #ifndef  __optix_optix_prime_atom_h__
