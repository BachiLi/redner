
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
 * @file   optix_defines.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * OptiX public API Reference - Definitions
 */

#ifndef __optix_optix_defines_h__
#define __optix_optix_defines_h__

/*! Transform type */
enum RTtransformkind {
  RT_WORLD_TO_OBJECT = 0xf00, /*!< World to Object transformation */
  RT_OBJECT_TO_WORLD          /*!< Object to World transformation */
};

/*! Transform flags */
enum RTtransformflags {
  RT_INTERNAL_INVERSE_TRANSPOSE = 0x1000 /*!< Inverse transpose flag */
};

namespace rti_internal_typeinfo {
  enum rtiTypeKind {
    _OPTIX_VARIABLE = 0x796152
  };
  struct rti_typeinfo {
    unsigned int kind;
    unsigned int size;
  };
}

namespace rti_internal_typeinfo {

  enum rtiTypeEnum {
    _OPTIX_TYPE_ENUM_UNKNOWN = 0x1337,
    _OPTIX_TYPE_ENUM_PROGRAM_ID,
    _OPTIX_TYPE_ENUM_PROGRAM_AS_ID
  };
  // Specialize this class (struct) to change the value of m_typeenum
  template<typename T>
  struct rti_typeenum
  {
    static const int m_typeenum = _OPTIX_TYPE_ENUM_UNKNOWN;
  };
}

namespace optix {
  /* Internal enums for identifying texture function lookup types.  Don't add new items in
   * the middle.  Only add at the end. */
  enum rtiTexLookupKind {   
    TEX_LOOKUP_1D = 1,
    TEX_LOOKUP_2D = 2,
    TEX_LOOKUP_3D = 3,
    TEX_LOOKUP_A1 = 4,
    TEX_LOOKUP_A2 = 5,
    TEX_LOOKUP_CUBE = 6,
    TEX_LOOKUP_ACUBE = 7
  };
}
  


#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64) || defined(__powerpc64__)
#define OPTIX_ASM_PTR          "l"
#define OPTIX_ASM_SIZE_T       "l"
#define OPTIX_ASM_PTR_SIZE_STR "64"
#define OPTIX_BITNESS_SUFFIX   "_64"
#else
#define OPTIX_ASM_PTR          "r"
#define OPTIX_ASM_SIZE_T       "r"
#define OPTIX_ASM_PTR_SIZE_STR "32"
#define OPTIX_BITNESS_SUFFIX   ""
#endif

namespace optix {
  typedef size_t optix_size_t;
}

#endif /* __optix_optix_defines_h__ */
