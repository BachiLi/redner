
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

#ifndef __optix_optix_prime_ref_h__
#define __optix_optix_prime_ref_h__

#include "Atom.h"

namespace optix {
  namespace prime {

    /// @cond

    /// Base class for reference counting.
    class RefCountedObj
    {
    public:
      /// Constructor initializes the reference count to \c 1.
      RefCountedObj( unsigned int refCnt=1 ) : m_refCnt( refCnt )  { }

      /// Virtual d'tor
      virtual ~RefCountedObj() { }

    private:
      friend class ContextObj;
      friend class ModelObj;
      template <class RefCountedObj> friend class Handle;

      /// Increment the reference count
      virtual unsigned int ref();

      /// Decrement the reference count
      virtual unsigned int unref();

      mutable Atom32 m_refCnt;
    };


    //
    // Increment the reference count
    //
    inline unsigned int RefCountedObj::ref() {
      return ++m_refCnt;
    }


    //
    // Decrement the reference count
    //
    inline unsigned int RefCountedObj::unref() {
      unsigned int cnt = --m_refCnt;
      if( !cnt) {
        delete this;
      }

      return cnt;
    }
    
    /// @endcond

  } // end namespace prime
} // end namespace optix

#endif // #ifndef __optix_optix_prime_ref_h__
