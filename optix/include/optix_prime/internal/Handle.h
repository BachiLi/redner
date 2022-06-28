
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

#ifndef __optix_optix_prime_handle_h__
#define __optix_optix_prime_handle_h__

#include "Ref.h"

namespace optix {
  namespace prime {
  
    /// @cond

    /// \brief Handle class that wraps reference counted objects
    template <class RefCountedObj>
    class Handle
    {
    public:
      /// Default constructor, initializes handle to hold an invalid interface.
      Handle();

      /// Constructor from Ref pointer, takes ownership of Ref.
      ///
      /// The constructor does not increment the reference count of \p ptr assuming it is
      /// already set properly. It therefore takes over the ownership of the Ref pointer.
      explicit Handle( RefCountedObj* ptr );

      /// Copy constructor, increments reference count if the Ref is valid.
      Handle( const Handle<RefCountedObj>& other);

      /// Swap two Refs.
      void swap( Handle<RefCountedObj>& other);

      /// Assignment operator, releases old Ref and increments
      /// reference count of the new Ref if Ref is valid.
      Handle<RefCountedObj>& operator=( const Handle<RefCountedObj>& other);

      /// Assignment operator from Ref pointer, releases old Ref and assigns
      /// new Ref \p ptr, takes ownership of Ref.
      /// Does not increment reference count of \p ptr assuming it is already set
      /// properly.
      Handle<RefCountedObj>& operator=( RefCountedObj* ptr);

      /// Releases the current Ref, decrementing the reference count.
      void unref();

      /// Destructor, releases Ref if it is valid, which decrements
      /// the reference count, and triggers thus the deletion of the interface
      /// implementation once the reference count reaches zero.
      ~Handle();

      /// Returns \c true if Ref is valid (not zero).
      bool isValid() const;

      /// Static object creation. Only valid for contexts.
      static Handle<RefCountedObj> create( RTPcontexttype type ) { return RefCountedObj::create( type ); }

      /// The arrow operator accesses the interface.
      RefCountedObj* operator->() const;
    private:

      RefCountedObj* m_iptr; // pointer to underlying interface, can be 0
    };


    //
    // Default constructor
    //
    template <class RefCountedObj>
    inline Handle<RefCountedObj>::Handle() : m_iptr(0)
    {

    }

    //
    // Constructor from Ref pointer
    //
    template <class RefCountedObj>
    inline Handle<RefCountedObj>::Handle( RefCountedObj* ptr ) : m_iptr(ptr)
    {

    }

    //
    // Copy constructor
    //
    template <class RefCountedObj>
    inline Handle<RefCountedObj>::Handle( const Handle<RefCountedObj>& other ) : m_iptr(other.m_iptr)
    {
      if ( m_iptr)
        m_iptr->ref();
    }

    //
    // Returns \c true if Ref is valid.
    //
    template <class RefCountedObj>
    inline bool Handle<RefCountedObj>::isValid() const
    {
      return m_iptr != 0;
    }

    //
    // Releases the current Ref
    //
    template <class RefCountedObj>
    inline void Handle<RefCountedObj>::unref()
    {
      if(m_iptr != NULL) {
        m_iptr->unref();
        m_iptr = NULL;
      }
    }

    //
    // Releases the current Ref
    //
    template <class RefCountedObj>
    inline Handle<RefCountedObj>::~Handle()
    {
      if ( m_iptr )
        m_iptr->unref();
    }

    //
    // Assignment operator
    //
    template <class RefCountedObj>
    inline Handle<RefCountedObj>& Handle<RefCountedObj>::operator=( RefCountedObj* ptr )
    {
      Handle<RefCountedObj>(ptr).swap(*this);
      return *this;
    }

    //
    // Assignment operator
    //
    template <class RefCountedObj>
    inline Handle<RefCountedObj>& Handle<RefCountedObj>::operator=( const Handle<RefCountedObj>& other )
    {
      Handle<RefCountedObj>(other).swap(*this);
      return *this;
    }

    //
    // Swap two Refs
    //
    template <class RefCountedObj>
    inline void Handle<RefCountedObj>::swap( Handle<RefCountedObj>& other )
    {
      RefCountedObj* tmp_iptr = m_iptr;
      m_iptr = other.m_iptr;
      other.m_iptr = tmp_iptr;
    }

    //
    // The arrow operator accesses the interface.
    //
    template <class RefCountedObj>
    inline RefCountedObj* Handle<RefCountedObj>::operator->() const
    {
      return  m_iptr;
    }

    /// @endcond

  } // end namespace prime
} // end namespace optix

#endif // #ifndef __optix_optix_prime_handle_h__
