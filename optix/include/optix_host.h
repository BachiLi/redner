
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
 * @file   optix_host.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API
 *
 * OptiX public API Reference - Host side
 */


#ifndef __optix_optix_host_h__
#define __optix_optix_host_h__

#ifndef RTAPI
#if defined(_WIN32)
#define RTAPI __declspec(dllimport)
#else
#define RTAPI
#endif
#endif

#include "internal/optix_declarations.h"


/************************************
 **
 **    Platform-Dependent Types
 **
 ***********************************/

#if defined(_WIN64)
typedef unsigned __int64    RTsize;
#elif defined(_WIN32)
typedef unsigned int        RTsize;
#else
typedef long unsigned int   RTsize;
#endif

/************************************
 **
 **    Opaque Object Types
 **
 ***********************************/

/** Opaque type to handle Acceleration Structures - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTacceleration_api       * RTacceleration;
/** Opaque type to handle Buffers - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTbuffer_api             * RTbuffer;
/** Opaque type to handle Contexts - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTcontext_api            * RTcontext;
/** Opaque type to handle Geometry - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTgeometry_api           * RTgeometry;
/** Opaque type to handle GeometryTriangles - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTgeometrytriangles_api  * RTgeometrytriangles;
/** Opaque type to handle Geometry Instance - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTgeometryinstance_api   * RTgeometryinstance;
/** Opaque type to handle Geometry Group - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTgeometrygroup_api      * RTgeometrygroup;
/** Opaque type to handle Group - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTgroup_api              * RTgroup;
/** Opaque type to handle Material - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTmaterial_api           * RTmaterial;
/** Opaque type to handle Program - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTprogram_api            * RTprogram;
/** Opaque type to handle Selector - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTselector_api           * RTselector;
/** Opaque type to handle Texture Sampler - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTtexturesampler_api     * RTtexturesampler;
/** Opaque type to handle Transform - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTtransform_api          * RTtransform;
/** Opaque type to handle Variable - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTvariable_api           * RTvariable;
/** Opaque type to handle Object - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef void                            * RTobject;
/** Opaque type to handle PostprocessingStage - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTpostprocessingstage_api* RTpostprocessingstage;
/** Opaque type to handle CommandList - Note that the *_api type should never be used directly.
Only the typedef target name will be guaranteed to remain unchanged */
typedef struct RTcommandlist_api        * RTcommandlist;

/************************************
 **
 **    Callback Function Types
 **
 ***********************************/

/** Callback signature for use with rtContextSetTimeoutCallback.
 * Deprecated in OptiX 6.0. */
typedef int (*RTtimeoutcallback)(void);

/** Callback signature for use with rtContextSetUsageReportCallback. */
typedef void (*RTusagereportcallback)(int, const char*, const char*, void*);


#ifdef __cplusplus
extern "C" {
#endif

/************************************
 **
 **    Context-free functions
 **
 ***********************************/

  /**
  * @brief Returns the current OptiX version
  *
  * @ingroup ContextFreeFunctions
  *
  * <B>Description</B>
  *
  * @ref rtGetVersion returns in \a version a numerically comparable
  * version number of the current OptiX library.
  *
  * The encoding for the version number prior to OptiX 4.0.0 is major*1000 + minor*10 + micro.
  * For versions 4.0.0 and higher, the encoding is major*10000 + minor*100 + micro.
  * For example, for version 3.5.1 this function would return 3051, and for version 4.5.1 it would return 40501.
  *
  * @param[out]  version   OptiX version number
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGetVersion was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtDeviceGetDeviceCount
  *
  */
  RTresult RTAPI rtGetVersion(unsigned int* version);

  /**
  * @brief Set a global attribute
  *
  * @ingroup ContextFreeFunctions
  *
  * <B>Description</B>
  *
  * @ref rtGlobalSetAttribute sets \a p as the value of the global attribute
  * specified by \a attrib.
  *
  * Each attribute can have a different size.  The sizes are given in the following list:
  *
  *   - @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX          sizeof(int)
  *
  * @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX sets the execution strategy used by Optix for the
  * next context to be created.
  * Possible values: 0 (legacy megakernel execution strategy), 1 (RTX execution strategy).
  *
  * @param[in]   attrib    Attribute to set
  * @param[in]   size      Size of the attribute being set
  * @param[in]   p         Pointer to where the value of the attribute will be copied from.  This must point to at least \a size bytes of memory
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_GLOBAL_ATTRIBUTE - Can be returned if an unknown attribute was addressed.
  * - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, or if \a p
  * is \a NULL
  *
  * <B>History</B>
  *
  * @ref rtGlobalSetAttribute was introduced in OptiX 5.1.
  *
  * <B>See also</B>
  * @ref rtGlobalGetAttribute
  *
  */
  RTresult RTAPI rtGlobalSetAttribute(RTglobalattribute attrib, RTsize size, const void* p);

  /**
  * @brief Returns a global attribute
  *
  * @ingroup ContextFreeFunctions
  *
  * <B>Description</B>
  *
  * @ref rtGlobalGetAttribute returns in \a p the value of the global attribute
  * specified by \a attrib.
  *
  * Each attribute can have a different size. The sizes are given in the following list:
  *
  *   - @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX                            sizeof(int)
  *   - @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR           sizeof(unsigned int)
  *   - @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR           sizeof(unsigend int)
  *
  * @ref RT_GLOBAL_ATTRIBUTE_ENABLE_RTX is an experimental setting which sets the execution strategy
  * used by Optix for the next context to be created.
  *
  * @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR is an attribute to query the major version of the display driver
  * found on the system. It's the first number in the driver version displayed as xxx.yy.
  *
  * @ref RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR is an attribute to query the minor version of the display driver
  * found on the system. It's the second number in the driver version displayed as xxx.yy.
  *
  * @param[in]   attrib    Attribute to query
  * @param[in]   size      Size of the attribute being queried.  Parameter \a p must have at least this much memory allocated
  * @param[out]  p         Return pointer where the value of the attribute will be copied into.  This must point to at least \a size bytes of memory
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_GLOBAL_ATTRIBUTE - Can be returned if an unknown attribute was addressed.
  * - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, if \a p is
  * \a NULL, or if \a attribute+ordinal does not correspond to an OptiX device
  * - @ref RT_ERROR_DRIVER_VERSION_FAILED - Can be returned if the display driver version could not be obtained.
  *
  * <B>History</B>
  *
  * @ref rtGlobalGetAttribute was introduced in OptiX 5.1.
  *
  * <B>See also</B>
  * @ref rtGlobalSetAttribute,
  *
  */
  RTresult RTAPI rtGlobalGetAttribute(RTglobalattribute attrib, RTsize size, void* p);

  /**
  * @brief Returns the number of OptiX capable devices
  *
  * @ingroup ContextFreeFunctions
  *
  * <B>Description</B>
  *
  * @ref rtDeviceGetDeviceCount returns in \a count the number of compute
  * devices that are available in the host system and will be used by
  * OptiX.
  *
  * @param[out]  count   Number devices available for OptiX
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtDeviceGetDeviceCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGetVersion
  *
  */
  RTresult RTAPI rtDeviceGetDeviceCount(unsigned int* count);

  /**
  * @brief Returns an attribute specific to an OptiX device
  *
  * @ingroup ContextFreeFunctions
  *
  * <B>Description</B>
  *
  * @ref rtDeviceGetAttribute returns in \a p the value of the per device attribute
  * specified by \a attrib for device \a ordinal.
  *
  * Each attribute can have a different size.  The sizes are given in the following list:
  *
  *   - @ref RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK        sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_CLOCK_RATE                   sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT         sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED    sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT   sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_NAME                         up to size-1
  *   - @ref RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY           sizeof(int2)
  *   - @ref RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY                 sizeof(RTsize)
  *   - @ref RT_DEVICE_ATTRIBUTE_TCC_DRIVER                   sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL          sizeof(int)
  *   - @ref RT_DEVICE_ATTRIBUTE_PCI_BUS_ID                   up to size-1, at most 13 chars
  *   - @ref RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES           sizeof(int)*(number of devices + 1)
  *
  * For \a RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES, the first \a int returned is the number
  * of compatible device ordinals returned.  A device is always compatible with itself, so
  * the count will always be at least one.  Size the output buffer based on the number of
  * devices as returned by \a rtDeviceGetDeviceCount.
  *
  * @param[in]   ordinal   OptiX device ordinal
  * @param[in]   attrib    Attribute to query
  * @param[in]   size      Size of the attribute being queried.  Parameter \a p must have at least this much memory allocated
  * @param[out]  p         Return pointer where the value of the attribute will be copied into.  This must point to at least \a size bytes of memory
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE - Can be returned if size does not match the proper size of the attribute, if \a p is
  * \a NULL, or if \a ordinal does not correspond to an OptiX device
  *
  * <B>History</B>
  *
  * @ref rtDeviceGetAttribute was introduced in OptiX 2.0.
  * @ref RT_DEVICE_ATTRIBUTE_TCC_DRIVER was introduced in OptiX 3.0.
  * @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL was introduced in OptiX 3.0.
  * @ref RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtDeviceGetDeviceCount,
  * @ref rtContextGetAttribute
  *
  */
  RTresult RTAPI rtDeviceGetAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, void* p);


/************************************
 **
 **    Object Variable Accessors
 **
 ***********************************/

  /* Sets */
  /**
  * @ingroup rtVariableSet Variable setters
  *
  * @brief Functions designed to modify the value of a program variable
  *
  * <B>Description</B>
  *
  * @ref rtVariableSet functions modify the value of a program variable or variable array. The
  * target variable is specificed by \a v, which should be a value returned by
  * @ref rtContextGetVariable.
  *
  * The commands \a rtVariableSet{1-2-3-4}{f-i-ui}v are used to modify the value of a
  * program variable specified by \a v using the values passed as arguments.
  * The number specified in the command should match the number of components in
  * the data type of the specified program variable (e.g., 1 for float, int,
  * unsigned int; 2 for float2, int2, uint2, etc.). The suffix \a f indicates
  * that \a v has floating point type, the suffix \a i indicates that
  * \a v has integral type, and the suffix \a ui indicates that that
  * \a v has unsigned integral type. The \a v variants of this function
  * should be used to load the program variable's value from the array specified by
  * parameter \a v. In this case, the array \a v should contain as many elements as
  * there are program variable components.
  *
  * The commands \a rtVariableSetMatrix{2-3-4}x{2-3-4}fv are used to modify the value
  * of a program variable whose data type is a matrix. The numbers in the command
  * names are the number of rows and columns, respectively.
  * For example, \a 2x4 indicates a matrix with 2 rows and 4 columns (i.e., 8 values).
  * If \a transpose is \a 0, the matrix is specified in row-major order, otherwise
  * in column-major order or, equivalently, as a matrix with the number of rows and
  * columns swapped in row-major order.
  *
  * If \a v is not a valid variable, these calls have no effect and return
  * @ref RT_ERROR_INVALID_VALUE
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtVariableSet were introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtVariableGet,
  * @ref rtVariableSet,
  * @ref rtDeclareVariable
  *
  * @{
  */
  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f1         Specifies the new float value of the program variable
  */
  RTresult RTAPI rtVariableSet1f(RTvariable v, float f1);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f1         Specifies the new float value of the program variable
  * @param[in]   f2         Specifies the new float value of the program variable
  */
  RTresult RTAPI rtVariableSet2f(RTvariable v, float f1, float f2);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f1         Specifies the new float value of the program variable
  * @param[in]   f2         Specifies the new float value of the program variable
  * @param[in]   f3         Specifies the new float value of the program variable
  */
  RTresult RTAPI rtVariableSet3f(RTvariable v, float f1, float f2, float f3);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f1         Specifies the new float value of the program variable
  * @param[in]   f2         Specifies the new float value of the program variable
  * @param[in]   f3         Specifies the new float value of the program variable
  * @param[in]   f4         Specifies the new float value of the program variable
  */
  RTresult RTAPI rtVariableSet4f(RTvariable v, float f1, float f2, float f3, float f4);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f          Array of float values to set the variable to
  */
  RTresult RTAPI rtVariableSet1fv(RTvariable v, const float* f);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f          Array of float values to set the variable to
  */
  RTresult RTAPI rtVariableSet2fv(RTvariable v, const float* f);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f          Array of float values to set the variable to
  */
  RTresult RTAPI rtVariableSet3fv(RTvariable v, const float* f);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   f          Array of float values to set the variable to
  */
  RTresult RTAPI rtVariableSet4fv(RTvariable v, const float* f);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i1         Specifies the new integer value of the program variable
  */
  RTresult RTAPI rtVariableSet1i(RTvariable v, int i1);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i1         Specifies the new integer value of the program variable
  * @param[in]   i2         Specifies the new integer value of the program variable
  */
  RTresult RTAPI rtVariableSet2i(RTvariable v, int i1, int i2);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i1         Specifies the new integer value of the program variable
  * @param[in]   i2         Specifies the new integer value of the program variable
  * @param[in]   i3         Specifies the new integer value of the program variable
  */
  RTresult RTAPI rtVariableSet3i(RTvariable v, int i1, int i2, int i3);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i1         Specifies the new integer value of the program variable
  * @param[in]   i2         Specifies the new integer value of the program variable
  * @param[in]   i3         Specifies the new integer value of the program variable
  * @param[in]   i4         Specifies the new integer value of the program variable
  */
  RTresult RTAPI rtVariableSet4i(RTvariable v, int i1, int i2, int i3, int i4);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i          Array of integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet1iv(RTvariable v, const int* i);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i          Array of integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet2iv(RTvariable v, const int* i);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i          Array of integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet3iv(RTvariable v, const int* i);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   i          Array of integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet4iv(RTvariable v, const int* i);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u1         Specifies the new unsigned integer value of the program variable
  */
  RTresult RTAPI rtVariableSet1ui(RTvariable v, unsigned int u1);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u1         Specifies the new unsigned integer value of the program variable
  * @param[in]   u2         Specifies the new unsigned integer value of the program variable
  */
  RTresult RTAPI rtVariableSet2ui(RTvariable v, unsigned int u1, unsigned int u2);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u1         Specifies the new unsigned integer value of the program variable
  * @param[in]   u2         Specifies the new unsigned integer value of the program variable
  * @param[in]   u3         Specifies the new unsigned integer value of the program variable
  */
  RTresult RTAPI rtVariableSet3ui(RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u1         Specifies the new unsigned integer value of the program variable
  * @param[in]   u2         Specifies the new unsigned integer value of the program variable
  * @param[in]   u3         Specifies the new unsigned integer value of the program variable
  * @param[in]   u4         Specifies the new unsigned integer value of the program variable
  */
  RTresult RTAPI rtVariableSet4ui(RTvariable v, unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u          Array of unsigned integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet1uiv(RTvariable v, const unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u          Array of unsigned integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet2uiv(RTvariable v, const unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u          Array of unsigned integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet3uiv(RTvariable v, const unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   u          Array of unsigned integer values to set the variable to
  */
  RTresult RTAPI rtVariableSet4uiv(RTvariable v, const unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll1        Specifies the new long long value of the program variable
  */
  RTresult RTAPI rtVariableSet1ll(RTvariable v, long long ll1);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll1        Specifies the new long long value of the program variable
  * @param[in]   ll2        Specifies the new long long value of the program variable
  */
  RTresult RTAPI rtVariableSet2ll(RTvariable v, long long ll1, long long ll2);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll1        Specifies the new long long value of the program variable
  * @param[in]   ll2        Specifies the new long long value of the program variable
  * @param[in]   ll3        Specifies the new long long value of the program variable
  */
  RTresult RTAPI rtVariableSet3ll(RTvariable v, long long ll1, long long ll2, long long ll3);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll1        Specifies the new long long value of the program variable
  * @param[in]   ll2        Specifies the new long long value of the program variable
  * @param[in]   ll3        Specifies the new long long value of the program variable
  * @param[in]   ll4        Specifies the new long long value of the program variable
  */
  RTresult RTAPI rtVariableSet4ll(RTvariable v, long long ll1, long long ll2, long long ll3, long long ll4);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll         Array of long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet1llv(RTvariable v, const long long* ll);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll         Array of long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet2llv(RTvariable v, const long long* ll);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll         Array of long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet3llv(RTvariable v, const long long* ll);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ll         Array of long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet4llv(RTvariable v, const long long* ll);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull1       Specifies the new unsigned long long value of the program variable
  */
  RTresult RTAPI rtVariableSet1ull(RTvariable v, unsigned long long ull1);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull1       Specifies the new unsigned long long value of the program variable
  * @param[in]   ull2       Specifies the new unsigned long long value of the program variable
  */
  RTresult RTAPI rtVariableSet2ull(RTvariable v, unsigned long long ull1, unsigned long long ull2);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull1       Specifies the new unsigned long long value of the program variable
  * @param[in]   ull2       Specifies the new unsigned long long value of the program variable
  * @param[in]   ull3       Specifies the new unsigned long long value of the program variable
  */
  RTresult RTAPI rtVariableSet3ull(RTvariable v, unsigned long long ull1, unsigned long long ull2, unsigned long long ull3);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull1       Specifies the new unsigned long long value of the program variable
  * @param[in]   ull2       Specifies the new unsigned long long value of the program variable
  * @param[in]   ull3       Specifies the new unsigned long long value of the program variable
  * @param[in]   ull4       Specifies the new unsigned long long value of the program variable
  */
  RTresult RTAPI rtVariableSet4ull(RTvariable v, unsigned long long ull1, unsigned long long ull2, unsigned long long ull3, unsigned long long ull4);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull        Array of unsigned long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet1ullv(RTvariable v, const unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull        Array of unsigned long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet2ullv(RTvariable v, const unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull        Array of unsigned long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet3ullv(RTvariable v, const unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   ull        Array of unsigned long long values to set the variable to
  */
  RTresult RTAPI rtVariableSet4ullv(RTvariable v, const unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix2x2fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix2x3fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix2x4fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix3x2fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix3x3fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix3x4fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix4x2fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix4x3fv(RTvariable v, int transpose, const float* m);

  /**
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   transpose  Specifies row-major or column-major order
  * @param[in]   m          Array of float values to set the matrix to
  */
  RTresult RTAPI rtVariableSetMatrix4x4fv(RTvariable v, int transpose, const float* m);
  /**
  * @}
  */

  /**
  * @brief Sets a program variable value to a OptiX object
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableSetObject sets a program variable to an OptiX object value.  The target
  * variable is specified by \a v. The new value of the program variable is
  * specified by \a object. The concrete type of \a object can be one of @ref RTbuffer,
  * @ref RTtexturesampler, @ref RTgroup, @ref RTprogram, @ref RTselector, @ref
  * RTgeometrygroup, or @ref RTtransform.  If \a v is not a valid variable or \a
  * object is not a valid OptiX object, this call has no effect and returns @ref
  * RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   v          Specifies the program variable to be set
  * @param[in]   object     Specifies the new value of the program variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtVariableSetObject was introduced in OptiX 1.0.  The ability to bind an @ref
  * RTprogram to a variable was introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtVariableGetObject,
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableSetObject(RTvariable v, RTobject object);

  /**
  * @brief Defined
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableSetUserData modifies the value of a program variable whose data type is
  * user-defined. The value copied into the variable is defined by an arbitrary region of
  * memory, pointed to by \a ptr. The size of the memory region is given by \a size. The
  * target variable is specified by \a v.  If \a v is not a valid variable,
  * this call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   v          Specifies the program variable to be modified
  * @param[in]   size       Specifies the size of the new value, in bytes
  * @param[in]   ptr        Specifies a pointer to the new value of the program variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtVariableSetUserData was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtVariableGetUserData,
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableSetUserData(RTvariable v, RTsize size, const void* ptr);


  /* Gets */
  /**
  * @ingroup rtVariableGet
  *
  * @brief Functions designed to modify the value of a program variable
  *
  * <B>Description</B>
  *
  * @ref rtVariableGet functions return the value of a program variable or variable
  * array. The target variable is specificed by \a v.
  *
  * The commands \a rtVariableGet{1-2-3-4}{f-i-ui}v are used to query the value
  * of a program variable specified by \a v using the pointers passed as arguments
  * as return locations for each component of the vector-typed variable. The number
  * specified in the command should match the number of components in the data type
  * of the specified program variable (e.g., 1 for float, int, unsigned int; 2 for
  * float2, int2, uint2, etc.). The suffix \a f indicates that floating-point
  * values are expected to be returned, the suffix \a i indicates that integer
  * values are expected, and the suffix \a ui indicates that unsigned integer
  * values are expected, and this type should also match the data type of the
  * specified program variable. The \a f variants of this function should be used
  * to query values for program variables defined as float, float2, float3, float4,
  * or arrays of these. The \a i variants of this function should be used to
  * query values for program variables defined as int, int2, int3, int4, or
  * arrays of these. The \a ui variants of this function should be used to query
  * values for program variables defined as unsigned int, uint2, uint3, uint4,
  * or arrays of these. The \a v variants of this function should be used to
  * return the program variable's value to the array specified by parameter
  * \a v. In this case, the array \a v should be large enough to accommodate all
  * of the program variable's components.
  *
  * The commands \a rtVariableGetMatrix{2-3-4}x{2-3-4}fv are used to query the
  * value of a program variable whose data type is a matrix. The numbers in the
  * command names are interpreted as the dimensionality of the matrix. For example,
  * \a 2x4 indicates a 2 x 4 matrix with 2 columns and 4 rows (i.e., 8
  * values). If \a transpose is \a 0, the matrix is returned in row major order,
  * otherwise in column major order.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtVariableGet were introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtVariableSet,
  * @ref rtVariableGetType,
  * @ref rtContextDeclareVariable
  *
  * @{
  */
  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f1         Float value to be returned
  */
  RTresult RTAPI rtVariableGet1f(RTvariable v, float* f1);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f1         Float value to be returned
  * @param[in]   f2         Float value to be returned
  */
  RTresult RTAPI rtVariableGet2f(RTvariable v, float* f1, float* f2);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f1         Float value to be returned
  * @param[in]   f2         Float value to be returned
  * @param[in]   f3         Float value to be returned
  */
  RTresult RTAPI rtVariableGet3f(RTvariable v, float* f1, float* f2, float* f3);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f1         Float value to be returned
  * @param[in]   f2         Float value to be returned
  * @param[in]   f3         Float value to be returned
  * @param[in]   f4         Float value to be returned
  */
  RTresult RTAPI rtVariableGet4f(RTvariable v, float* f1, float* f2, float* f3, float* f4);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f          Array of float value(s) to be returned
  */
  RTresult RTAPI rtVariableGet1fv(RTvariable v, float* f);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f          Array of float value(s) to be returned
  */
  RTresult RTAPI rtVariableGet2fv(RTvariable v, float* f);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f          Array of float value(s) to be returned
  */
  RTresult RTAPI rtVariableGet3fv(RTvariable v, float* f);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   f          Array of float value(s) to be returned
  */
  RTresult RTAPI rtVariableGet4fv(RTvariable v, float* f);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i1         Integer value to be returned
  */
  RTresult RTAPI rtVariableGet1i(RTvariable v, int* i1);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i1         Integer value to be returned
  * @param[in]   i2         Integer value to be returned
  */
  RTresult RTAPI rtVariableGet2i(RTvariable v, int* i1, int* i2);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i1         Integer value to be returned
  * @param[in]   i2         Integer value to be returned
  * @param[in]   i3         Integer value to be returned
  */
  RTresult RTAPI rtVariableGet3i(RTvariable v, int* i1, int* i2, int* i3);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i1         Integer value to be returned
  * @param[in]   i2         Integer value to be returned
  * @param[in]   i3         Integer value to be returned
  * @param[in]   i4         Integer value to be returned
  */
  RTresult RTAPI rtVariableGet4i(RTvariable v, int* i1, int* i2, int* i3, int* i4);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i          Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet1iv(RTvariable v, int* i);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i          Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet2iv(RTvariable v, int* i);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i          Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet3iv(RTvariable v, int* i);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   i          Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet4iv(RTvariable v, int* i);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet1ui(RTvariable v, unsigned int* u1);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  * @param[in]   u2         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet2ui(RTvariable v, unsigned int* u1, unsigned int* u2);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  * @param[in]   u2         Unsigned integer value to be returned
  * @param[in]   u3         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet3ui(RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  * @param[in]   u2         Unsigned integer value to be returned
  * @param[in]   u3         Unsigned integer value to be returned
  * @param[in]   u4         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet4ui(RTvariable v, unsigned int* u1, unsigned int* u2, unsigned int* u3, unsigned int* u4);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u          Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet1uiv(RTvariable v, unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u          Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet2uiv(RTvariable v, unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u          Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet3uiv(RTvariable v, unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u          Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet4uiv(RTvariable v, unsigned int* u);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll1        Integer value to be returned
  */
  RTresult RTAPI rtVariableGet1ll(RTvariable v, long long* ll1);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll1        Integer value to be returned
  * @param[in]   ll2        Integer value to be returned
  */
  RTresult RTAPI rtVariableGet2ll(RTvariable v, long long* ll1, long long* ll2);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll1        Integer value to be returned
  * @param[in]   ll2        Integer value to be returned
  * @param[in]   ll3        Integer value to be returned
  */
  RTresult RTAPI rtVariableGet3ll(RTvariable v, long long* ll1, long long* ll2, long long* ll3);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll1        Integer value to be returned
  * @param[in]   ll2        Integer value to be returned
  * @param[in]   ll3        Integer value to be returned
  * @param[in]   ll4        Integer value to be returned
  */
  RTresult RTAPI rtVariableGet4ll(RTvariable v, long long* ll1, long long* ll2, long long* ll3, long long* ll4);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll         Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet1llv(RTvariable v, long long* ll);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll         Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet2llv(RTvariable v, long long* ll);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll         Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet3llv(RTvariable v, long long* ll);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ll         Array of integer values to be returned
  */
  RTresult RTAPI rtVariableGet4llv(RTvariable v, long long* ll);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet1ull(RTvariable v, unsigned long long* u1);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  * @param[in]   u2         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet2ull(RTvariable v, unsigned long long* u1, unsigned long long* u2);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  * @param[in]   u2         Unsigned integer value to be returned
  * @param[in]   u3         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet3ull(RTvariable v, unsigned long long* u1, unsigned long long* u2, unsigned long long* u3);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   u1         Unsigned integer value to be returned
  * @param[in]   u2         Unsigned integer value to be returned
  * @param[in]   u3         Unsigned integer value to be returned
  * @param[in]   u4         Unsigned integer value to be returned
  */
  RTresult RTAPI rtVariableGet4ull(RTvariable v, unsigned long long* u1, unsigned long long* u2, unsigned long long* u3, unsigned long long* u4);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ull        Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet1ullv(RTvariable v, unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ull        Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet2ullv(RTvariable v, unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ull        Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet3ullv(RTvariable v, unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   ull        Array of unsigned integer values to be returned
  */
  RTresult RTAPI rtVariableGet4ullv(RTvariable v, unsigned long long* ull);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix2x2fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix2x3fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix2x4fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix3x2fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix3x3fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix3x4fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix4x2fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix4x3fv(RTvariable v, int transpose, float* m);

  /**
  * @param[in]   v          Specifies the program variable whose value is to be returned
  * @param[in]   transpose  Specify(ies) row-major or column-major order
  * @param[in]   m          Array of float values to be returned
  */
  RTresult RTAPI rtVariableGetMatrix4x4fv(RTvariable v, int transpose, float* m);
  /**
  * @}
  */

  /**
  * @brief Returns the value of a OptiX object program variable
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableGetObject queries the value of a program variable whose data type is a
  * OptiX object.  The target variable is specified by \a v. The value of the
  * program variable is returned in \a *object. The concrete
  * type of the program variable can be queried using @ref rtVariableGetType, and the @ref
  * RTobject handle returned by @ref rtVariableGetObject may safely be cast to an OptiX
  * handle of corresponding type. If \a v is not a valid variable, this call sets
  * \a *object to \a NULL and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   v          Specifies the program variable to be queried
  * @param[out]  object     Returns the value of the program variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtVariableGetObject was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtVariableSetObject,
  * @ref rtVariableGetType,
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableGetObject(RTvariable v, RTobject* object);

  /**
  * @brief Defined
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableGetUserData queries the value of a program variable whose data type is
  * user-defined. The variable of interest is specified by \a v.  The size of the
  * variable's value must match the value given by the parameter \a size.  The value of
  * the program variable is copied to the memory region pointed to by \a ptr. The storage
  * at location \a ptr must be large enough to accommodate all of the program variable's
  * value data. If \a v is not a valid variable, this call has no effect and
  * returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   v          Specifies the program variable to be queried
  * @param[in]   size       Specifies the size of the program variable, in bytes
  * @param[out]  ptr        Location in which to store the value of the variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtVariableGetUserData was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtVariableSetUserData,
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableGetUserData(RTvariable v, RTsize size, void* ptr);


  /* Other */
  /**
  * @brief Queries the name of a program variable
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * Queries a program variable's name. The variable of interest is specified by \a
  * variable, which should be a value returned by @ref rtContextDeclareVariable. A pointer
  * to the string containing the name of the variable is returned in \a *nameReturn.
  * If \a v is not a valid variable, this
  * call sets \a *nameReturn to \a NULL and returns @ref RT_ERROR_INVALID_VALUE.  \a
  * *nameReturn will point to valid memory until another API function that returns a
  * string is called.
  *
  * @param[in]   v             Specifies the program variable to be queried
  * @param[out]  nameReturn    Returns the program variable's name
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtVariableGetName was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableGetName(RTvariable v, const char** nameReturn);

  /**
  * @brief Queries the annotation string of a program variable
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableGetAnnotation queries a program variable's annotation string. A pointer
  * to the string containing the annotation is returned in \a *annotationReturn.
  * If \a v is not a valid variable, this call sets
  * \a *annotationReturn to \a NULL and returns @ref RT_ERROR_INVALID_VALUE.  \a
  * *annotationReturn will point to valid memory until another API function that returns
  * a string is called.
  *
  * @param[in]   v                   Specifies the program variable to be queried
  * @param[out]  annotationReturn    Returns the program variable's annotation string
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtVariableGetAnnotation was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtDeclareVariable,
  * @ref rtDeclareAnnotation
  *
  */
  RTresult RTAPI rtVariableGetAnnotation(RTvariable v, const char** annotationReturn);

  /**
  * @brief Returns type information about a program variable
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableGetType queries a program variable's type. The variable of interest is
  * specified by \a v. The program variable's type enumeration is returned in \a *typeReturn,
  * if it is not \a NULL. It is one of the following:
  *
  *   - @ref RT_OBJECTTYPE_UNKNOWN
  *   - @ref RT_OBJECTTYPE_GROUP
  *   - @ref RT_OBJECTTYPE_GEOMETRY_GROUP
  *   - @ref RT_OBJECTTYPE_TRANSFORM
  *   - @ref RT_OBJECTTYPE_SELECTOR
  *   - @ref RT_OBJECTTYPE_GEOMETRY_INSTANCE
  *   - @ref RT_OBJECTTYPE_BUFFER
  *   - @ref RT_OBJECTTYPE_TEXTURE_SAMPLER
  *   - @ref RT_OBJECTTYPE_OBJECT
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT2x2
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT2x3
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT2x4
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT3x2
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT3x3
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT3x4
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT4x2
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT4x3
  *   - @ref RT_OBJECTTYPE_MATRIX_FLOAT4x4
  *   - @ref RT_OBJECTTYPE_FLOAT
  *   - @ref RT_OBJECTTYPE_FLOAT2
  *   - @ref RT_OBJECTTYPE_FLOAT3
  *   - @ref RT_OBJECTTYPE_FLOAT4
  *   - @ref RT_OBJECTTYPE_INT
  *   - @ref RT_OBJECTTYPE_INT2
  *   - @ref RT_OBJECTTYPE_INT3
  *   - @ref RT_OBJECTTYPE_INT4
  *   - @ref RT_OBJECTTYPE_UNSIGNED_INT
  *   - @ref RT_OBJECTTYPE_UNSIGNED_INT2
  *   - @ref RT_OBJECTTYPE_UNSIGNED_INT3
  *   - @ref RT_OBJECTTYPE_UNSIGNED_INT4
  *   - @ref RT_OBJECTTYPE_USER
  *
  * Sets \a *typeReturn to @ref RT_OBJECTTYPE_UNKNOWN if \a v is not a valid variable.
  * Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  *
  * @param[in]   v             Specifies the program variable to be queried
  * @param[out]  typeReturn    Returns the type of the program variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtVariableGetType was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableGetType(RTvariable v, RTobjecttype* typeReturn);

  /**
  * @brief Returns the context associated with a program variable
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableGetContext queries the context associated with a program variable.  The
  * target variable is specified by \a v. The context of the program variable is
  * returned to \a *context if the pointer \a context is not \a NULL. If \a v is
  * not a valid variable, \a *context is set to \a NULL and @ref RT_ERROR_INVALID_VALUE is
  * returned.
  *
  * @param[in]   v          Specifies the program variable to be queried
  * @param[out]  context    Returns the context associated with the program variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtVariableGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableGetContext(RTvariable v, RTcontext* context);

  /**
  * @brief Queries the size, in bytes, of a variable
  *
  * @ingroup Variables
  *
  * <B>Description</B>
  *
  * @ref rtVariableGetSize queries a declared program variable for its size in bytes.
  * This is most often used to query the size of a variable that has a user-defined type.
  * Builtin types (int, float, unsigned int, etc.) may be queried, but object typed
  * variables, such as buffers, texture samplers and graph nodes, cannot be queried and
  * will return @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   v          Specifies the program variable to be queried
  * @param[out]  size       Specifies a pointer where the size of the variable, in bytes, will be returned
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtVariableGetSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtVariableGetUserData,
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtVariableGetSize(RTvariable v, RTsize* size);


/************************************
 **
 **    Context object
 **
 ***********************************/

  /**
  * @brief Creates a new context object
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextCreate allocates and returns a handle to a new context object.
  * Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
  *
  * @param[out]  context   Handle to context for return value
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_NO_DEVICE
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  *
  *
  */
  RTresult RTAPI rtContextCreate(RTcontext* context);

  /**
  * @brief Destroys a context and frees all associated resources
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextDestroy frees all resources, including OptiX objects, associated with
  * this object.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL context.  @ref
  * RT_ERROR_LAUNCH_FAILED may be returned if a previous call to @ref rtContextLaunch "rtContextLaunch"
  * failed.
  *
  * @param[in]   context   Handle of the context to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_LAUNCH_FAILED
  *
  * <B>History</B>
  *
  * @ref rtContextDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtContextDestroy(RTcontext context);

  /**
  * @brief Checks the given context for valid internal state
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextValidate checks the the given context and all of its associated OptiX
  * objects for a valid state.  These checks include tests for presence of necessary
  * programs (e.g. an intersection program for a geometry node), invalid internal state
  * such as \a NULL children in graph nodes, and presence of variables required by all
  * specified programs. @ref rtContextGetErrorString can be used to retrieve a description
  * of a validation failure.
  *
  * @param[in]   context   The context to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_SOURCE
  *
  * <B>History</B>
  *
  * @ref rtContextValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetErrorString
  *
  */
  RTresult RTAPI rtContextValidate(RTcontext context);

  /**
  * @brief Returns the error string associated with a given
  * error
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetErrorString return a descriptive string given an error code.  If \a
  * context is valid and additional information is available from the last OptiX failure,
  * it will be appended to the generic error code description.  \a stringReturn will be
  * set to point to this string.  The memory \a stringReturn points to will be valid
  * until the next API call that returns a string.
  *
  * @param[in]   context         The context object to be queried, or \a NULL
  * @param[in]   code            The error code to be converted to string
  * @param[out]  stringReturn    The return parameter for the error string
  *
  * <B>Return values</B>
  *
  * @ref rtContextGetErrorString does not return a value
  *
  * <B>History</B>
  *
  * @ref rtContextGetErrorString was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  *
  *
  */
  void RTAPI rtContextGetErrorString(RTcontext context, RTresult code, const char** stringReturn);

  /**
  * @brief Set an attribute specific to an OptiX context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetAttribute sets \a p as the value of the per context attribute
  * specified by \a attrib.
  *
  * Each attribute can have a different size.  The sizes are given in the following list:
  *
  *   - @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS             sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES      sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION         sizeof(char*)
  *   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS    sizeof(RTSize[2])
  *   - @ref RT_CONTEXT_ATTRIBUTE_MAX_CONCURRENT_LAUNCHES     sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL sizeof(int)
  *
  * @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS sets the number of host CPU threads OptiX
  * can use for various tasks.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES is a hint about scene usage.  By
  * default OptiX produces device kernels that are optimized for the current scene.  Such
  * kernels generally run faster, but must be recompiled after some types of scene
  * changes, causing delays.  Setting PREFER_FAST_RECOMPILES to 1 will leave out some
  * scene-specific optimizations, producing kernels that generally run slower but are less
  * sensitive to changes in the scene.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS sets whether or not OptiX will
  * automatically inline user functions, which is the default behavior.  Please see the
  * Programming Guide for more information about the benefits and limitations of disabling
  * automatic inlining.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION sets the location where the OptiX disk
  * cache will be created.  The location must be provided as a \a NULL-terminated
  * string. OptiX will attempt to create the directory if it does not exist.  An exception
  * will be thrown if OptiX is unable to create the cache database file at the specified
  * location for any reason (e.g., the path is invalid or the directory is not writable).
  * The location of the disk cache can be overridden with the environment variable \a
  * OPTIX_CACHE_PATH. This environment variable takes precedence over the RTcontext
  * attribute. The default location depends on the operating system:
  *
  *   - Windows: %LOCALAPPDATA%\\NVIDIA\\OptixCache
  *   - Linux:   /var/tmp/OptixCache_\<username\> (or /tmp/OptixCache_\<username\> if the first
  *              choice is not usable), the underscore and username suffix are omitted if the
  *              username cannot be obtained
  *   - MacOS X: /Library/Application Support/NVIDIA/OptixCache
  *
  * @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS sets the low and high watermarks
  * for disk cache garbage collection.  The limits must be passed in as a two-element
  * array of \a RTsize values, with the low limit as the first element.  OptiX will throw
  * an exception if either limit is non-zero and the high limit is not greater than the
  * low limit.  Setting either limit to zero will disable garbage collection.  Garbage
  * collection is triggered whenever the cache data size exceeds the high watermark and
  * proceeds until the size reaches the low watermark.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_MAX_CONCURRENT_LAUNCHES sets the maximum number of allowed
  * concurrent asynchronous launches per device. The actual number of launches can be less than
  * the set limit, and actual GPU scheduling may affect concurrency. This limit affects only
  * asynchronous launches. Valid values are from 1 to the maximum number of CUDA streams
  * supported by a device. Default value is 2.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL sets whether or not OptiX should prefer
  * to use a watertight traversal method or not. The default behaviour is preferring to use
  * watertight traversal. Note that OptiX might still choose to decide otherwise though.
  * Please see the Programming Guide for more information about the different traversal methods.
  *
  * @param[in]   context   The context object to be modified
  * @param[in]   attrib    Attribute to set
  * @param[in]   size      Size of the attribute being set
  * @param[in]   p         Pointer to where the value of the attribute will be copied from.  This must point to at least \a size bytes of memory
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, or if \a p
  * is \a NULL
  *
  * <B>History</B>
  *
  * @ref rtContextSetAttribute was introduced in OptiX 2.5.
  *
  * <B>See also</B>
  * @ref rtContextGetAttribute
  *
  */
  RTresult RTAPI rtContextSetAttribute(RTcontext context, RTcontextattribute attrib, RTsize size, const void* p);

  /**
  * @brief Returns an attribute specific to an OptiX context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetAttribute returns in \a p the value of the per context attribute
  * specified by \a attrib.
  *
  * Each attribute can have a different size.  The sizes are given in the following list:
  *
  *   - @ref RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT        sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS          sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY         sizeof(RTsize)
  *   - @ref RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY  sizeof(RTsize)
  *   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED       sizeof(int)
  *   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION      sizeof(char**)
  *   - @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS sizeof(RTSize[2])
  *   - @ref RT_CONTEXT_ATTRIBUTE_MAX_CONCURRENT_LAUNCHES  sizeof(int)
  *
  * @ref RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT queries the maximum number of textures
  * handled by OptiX. For OptiX versions below 2.5 this value depends on the number of
  * textures supported by CUDA.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS queries the number of host CPU threads OptiX
  * can use for various tasks.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY queries the amount of host memory allocated
  * by OptiX.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY queries the amount of free device
  * memory.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED queries whether or not the OptiX disk
  * cache is enabled.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION queries the file path of the OptiX
  * disk cache.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS queries the low and high watermark values
  * for the OptiX disk cache.
  *
  * @ref RT_CONTEXT_ATTRIBUTE_MAX_CONCURRENT_LAUNCHES queries the number of concurrent asynchronous
  * launches allowed per device.
  *
  * Some attributes are used to get per device information.  In contrast to @ref
  * rtDeviceGetAttribute, these attributes are determined by the context and are therefore
  * queried through the context.  This is done by adding the attribute with the OptiX
  * device ordinal number when querying the attribute.  The following are per device attributes.
  *
  *   @ref RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY
  *
  * @param[in]   context   The context object to be queried
  * @param[in]   attrib    Attribute to query
  * @param[in]   size      Size of the attribute being queried.  Parameter \a p must have at least this much memory allocated
  * @param[out]  p         Return pointer where the value of the attribute will be copied into.  This must point to at least \a size bytes of memory
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE - Can be returned if \a size does not match the proper size of the attribute, if \a p is
  * \a NULL, or if \a attribute+ordinal does not correspond to an OptiX device
  *
  * <B>History</B>
  *
  * @ref rtContextGetAttribute was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtContextGetDeviceCount,
  * @ref rtContextSetAttribute,
  * @ref rtDeviceGetAttribute
  *
  */
  RTresult RTAPI rtContextGetAttribute(RTcontext context, RTcontextattribute attrib, RTsize size, void* p);

  /**
  * @brief Specify a list of hardware devices to be used by the
  * kernel
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetDevices specifies a list of hardware devices to be used during
  * execution of the subsequent trace kernels. Note that the device numbers are
  * OptiX device ordinals, which may not be the same as CUDA device ordinals.
  * Use @ref rtDeviceGetAttribute with @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL to query the CUDA device
  * corresponding to a particular OptiX device.
  *
  * @param[in]   context   The context to which the hardware list is applied
  * @param[in]   count     The number of devices in the list
  * @param[in]   devices   The list of devices
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_NO_DEVICE
  * - @ref RT_ERROR_INVALID_DEVICE
  *
  * <B>History</B>
  *
  * @ref rtContextSetDevices was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetDevices,
  * @ref rtContextGetDeviceCount
  *
  */
  RTresult RTAPI rtContextSetDevices(RTcontext context, unsigned int count, const int* devices);

  /**
  * @brief Retrieve a list of hardware devices being used by the
  * kernel
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetDevices retrieves a list of hardware devices used by the context.
  * Note that the device numbers are  OptiX device ordinals, which may not be the same as CUDA device ordinals.
  * Use @ref rtDeviceGetAttribute with @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL to query the CUDA device
  * corresponding to a particular OptiX device.
  *
  * @param[in]   context   The context to which the hardware list is applied
  * @param[out]  devices   Return parameter for the list of devices.  The memory must be able to hold entries
  * numbering least the number of devices as returned by @ref rtContextGetDeviceCount
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetDevices was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtContextSetDevices,
  * @ref rtContextGetDeviceCount
  *
  */
  RTresult RTAPI rtContextGetDevices(RTcontext context, int* devices);

  /**
  * @brief Query the number of devices currently being used
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetDeviceCount - Query the number of devices currently being used.
  *
  * @param[in]   context   The context containing the devices
  * @param[out]  count     Return parameter for the device count
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetDeviceCount was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtContextSetDevices,
  * @ref rtContextGetDevices
  *
  */
  RTresult RTAPI rtContextGetDeviceCount(RTcontext context, unsigned int* count);

  /**
  * @brief Set the stack size for a given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetStackSize sets the stack size for the given context to
  * \a bytes bytes. Not supported with the RTX execution strategy.
  * With RTX execution strategy @ref rtContextSetMaxTraceDepth and @ref rtContextSetMaxCallableProgramDepth
  * should be used to control stack size.
  * Returns @ref RT_ERROR_INVALID_VALUE if context is not valid.
  *
  * @param[in]   context  The context node to be modified
  * @param[in]   bytes    The desired stack size in bytes
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetStackSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetStackSize
  *
  */
  RTresult RTAPI rtContextSetStackSize(RTcontext context, RTsize bytes);

  /**
  * @brief Query the stack size for this context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetStackSize passes back the stack size associated with this context in
  * \a bytes.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
  *
  * @param[in]   context The context node to be queried
  * @param[out]  bytes   Return parameter to store the size of the stack
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetStackSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextSetStackSize
  *
  */
  RTresult RTAPI rtContextGetStackSize(RTcontext context, RTsize* bytes);

  /**
  * @brief Set maximum callable program call depth for a given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetMaxCallableProgramDepth sets the maximum call depth of a chain of callable programs
  * for the given context to \a maxDepth. This value is only used for stack size computation.
  * Only supported for RTX execution mode. Default value is 5.
  * Returns @ref RT_ERROR_INVALID_VALUE if context is not valid.
  *
  * @param[in]   context            The context node to be modified
  * @param[in]   maxDepth           The desired maximum depth
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetMaxCallableProgramDepth was introduced in OptiX 6.0
  *
  * <B>See also</B>
  * @ref rtContextGetMaxCallableProgramDepth
  *
  */
  RTresult RTAPI rtContextSetMaxCallableProgramDepth( RTcontext context, unsigned int maxDepth );

  /**
  * @brief Query the maximum call depth for callable programs
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetMaxCallableProgramDepth passes back the maximum callable program call depth
  * associated with this context in \a maxDepth.
  * Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
  *
  * @param[in]   context            The context node to be queried
  * @param[out]  maxDepth           Return parameter to store the maximum callable program depth
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetMaxCallableProgramDepth was introduced in OptiX 6.0
  *
  * <B>See also</B>
  * @ref rtContextSetMaxCallableProgramDepth
  *
  */
  RTresult RTAPI rtContextGetMaxCallableProgramDepth( RTcontext context, unsigned int* maxDepth );

  /**
  * @brief Set the maximum trace depth for a given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetMaxTraceDepth sets the maximum trace depth for the given context to
  * \a maxDepth. Only supported for RTX execution mode. Default value is 5. Maximum trace depth is 31.
  * Returns @ref RT_ERROR_INVALID_VALUE if context is not valid.
  *
  * @param[in]   context            The context node to be modified
  * @param[in]   maxDepth           The desired maximum depth
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetMaxTraceDepth was introduced in OptiX 6.0
  *
  * <B>See also</B>
  * @ref rtContextGetMaxTraceDepth
  *
  */
  RTresult RTAPI rtContextSetMaxTraceDepth( RTcontext context, unsigned int maxDepth );

  /**
  * @brief Query the maximum trace depth for this context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetMaxTraceDepth passes back the maximum trace depth associated with this context in
  * \a maxDepth.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
  *
  * @param[in]   context            The context node to be queried
  * @param[out]  maxDepth           Return parameter to store the maximum trace depth
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetMaxTraceDepth was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtContextSetMaxTraceDepth
  *
  */
  RTresult RTAPI rtContextGetMaxTraceDepth( RTcontext context, unsigned int* maxDepth );

  /**
  * Deprecated in OptiX 6.0. Calling this function has no effect.
  */
  RTresult RTAPI rtContextSetTimeoutCallback(RTcontext context, RTtimeoutcallback callback, double minPollingSeconds);

  /**
  * @brief Set usage report callback function
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetUsageReportCallback sets an application-side callback
  * function \a callback and a verbosity level \a verbosity.
  *
  * @ref RTusagereportcallback is defined as
  * \a void (*RTusagereportcallback)(int, const char*, const char*, void*).
  *
  * The provided callback will be invoked with the message's verbosity level as
  * the first parameter.  The second parameter is a descriptive tag string and
  * the third parameter is the message itself.  The fourth parameter is a pointer
  * to user-defined data, which may be NULL.  The descriptive tag will give a
  * terse message category description (eg, 'SCENE STAT').  The messages will
  * be unstructured and subject to change with subsequent releases.  The
  * verbosity argument specifies the granularity of these messages.
  *
  * \a verbosity of 0 disables reporting.  \a callback is ignored in this case.
  *
  * \a verbosity of 1 enables error messages and important warnings.  This
  * verbosity level can be expected to be efficient and have no significant
  * overhead.
  *
  * \a verbosity of 2 additionally enables minor warnings, performance
  * recommendations, and scene statistics at startup or recompilation
  * granularity.  This level may have a performance cost.
  *
  * \a verbosity of 3 additionally enables informational messages and per-launch
  * statistics and messages.
  *
  * A NULL \a callback when verbosity is non-zero or a \a verbosity outside of
  * [0, 3] will result in @ref RT_ERROR_INVALID_VALUE return code.
  *
  * Only one report callback function can be specified at any time.
  *
  * @param[in]   context               The context node to be modified
  * @param[in]   callback              The function to be called
  * @param[in]   verbosity             The verbosity of report messages
  * @param[in]   cbdata                Pointer to user-defined data that will be sent to the callback.  Can be NULL.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetUsageReportCallback was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  *
  */
  RTresult RTAPI rtContextSetUsageReportCallback(RTcontext context, RTusagereportcallback callback, int verbosity, void* cbdata);

  /**
  * @brief Set the number of entry points for a given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetEntryPointCount sets the number of entry points associated with
  * the given context to \a count.
  *
  * @param[in]   context The context to be modified
  * @param[in]   count   The number of entry points to use
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetEntryPointCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetEntryPointCount
  *
  */
  RTresult RTAPI rtContextSetEntryPointCount(RTcontext context, unsigned int count);

  /**
  * @brief Query the number of entry points for this
  * context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetEntryPointCount passes back the number of entry points associated
  * with this context in \a count.  Returns @ref RT_ERROR_INVALID_VALUE if
  * passed a \a NULL pointer.
  *
  * @param[in]   context The context node to be queried
  * @param[out]  count   Return parameter for passing back the entry point count
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetEntryPointCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextSetEntryPointCount
  *
  */
  RTresult RTAPI rtContextGetEntryPointCount(RTcontext context, unsigned int* count);

  /**
  * @brief Specifies the ray generation program for
  * a given context entry point
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetRayGenerationProgram sets \a context's ray generation program at
  * entry point \a entryPointIndex. @ref RT_ERROR_INVALID_VALUE is returned if \a
  * entryPointIndex is outside of the range [\a 0, @ref rtContextGetEntryPointCount
  * \a -1].
  *
  * @param[in]   context             The context node to which the exception program will be added
  * @param[in]   entryPointIndex     The entry point the program will be associated with
  * @param[in]   program             The ray generation program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtContextSetRayGenerationProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetEntryPointCount,
  * @ref rtContextGetRayGenerationProgram
  *
  */
  RTresult RTAPI rtContextSetRayGenerationProgram(RTcontext context, unsigned int entryPointIndex, RTprogram program);

  /**
  * @brief Queries the ray generation program
  * associated with the given context and entry point
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetRayGenerationProgram passes back the ray generation program
  * associated with the given context and entry point.  This program is set via @ref
  * rtContextSetRayGenerationProgram.  Returns @ref RT_ERROR_INVALID_VALUE if given an
  * invalid entry point index or \a NULL pointer.
  *
  * @param[in]   context             The context node associated with the ray generation program
  * @param[in]   entryPointIndex     The entry point index for the desired ray generation program
  * @param[out]  program             Return parameter to store the ray generation program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetRayGenerationProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextSetRayGenerationProgram
  *
  */
  RTresult RTAPI rtContextGetRayGenerationProgram(RTcontext context, unsigned int entryPointIndex, RTprogram* program);

  /**
  * @brief Specifies the exception program for a given context entry point
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetExceptionProgram sets \a context's exception program at entry point
  * \a entryPointIndex. @ref RT_ERROR_INVALID_VALUE is returned if \a entryPointIndex
  * is outside of the range [\a 0, @ref rtContextGetEntryPointCount \a -1].
  *
  * @param[in]   context             The context node to which the exception program will be added
  * @param[in]   entryPointIndex     The entry point the program will be associated with
  * @param[in]   program             The exception program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtContextSetExceptionProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetEntryPointCount,
  * @ref rtContextGetExceptionProgram
  * @ref rtContextSetExceptionEnabled,
  * @ref rtContextGetExceptionEnabled,
  * @ref rtGetExceptionCode,
  * @ref rtThrow,
  * @ref rtPrintExceptionDetails
  *
  */
  RTresult RTAPI rtContextSetExceptionProgram(RTcontext context, unsigned int entryPointIndex, RTprogram program);

  /**
  * @brief Queries the exception program associated with
  * the given context and entry point
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetExceptionProgram passes back the exception program associated with
  * the given context and entry point.  This program is set via @ref
  * rtContextSetExceptionProgram.  Returns @ref RT_ERROR_INVALID_VALUE if given an invalid
  * entry point index or \a NULL pointer.
  *
  * @param[in]   context             The context node associated with the exception program
  * @param[in]   entryPointIndex     The entry point index for the desired exception program
  * @param[out]  program             Return parameter to store the exception program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetExceptionProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextSetExceptionProgram,
  * @ref rtContextSetEntryPointCount,
  * @ref rtContextSetExceptionEnabled,
  * @ref rtContextGetExceptionEnabled,
  * @ref rtGetExceptionCode,
  * @ref rtThrow,
  * @ref rtPrintExceptionDetails
  *
  */
  RTresult RTAPI rtContextGetExceptionProgram(RTcontext context, unsigned int entryPointIndex, RTprogram* program);

  /**
  * @brief Enable or disable an exception
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetExceptionEnabled is used to enable or disable specific exceptions.
  * If an exception is enabled, the exception condition is checked for at runtime, and the
  * exception program is invoked if the condition is met. The exception program can query
  * the type of the caught exception by calling @ref rtGetExceptionCode.
  * \a exception may take one of the following values:
  *
  *   - @ref RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS
  *   - @ref RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS
  *   - @ref RT_EXCEPTION_TRACE_DEPTH_EXCEEDED
  *   - @ref RT_EXCEPTION_TEXTURE_ID_INVALID
  *   - @ref RT_EXCEPTION_BUFFER_ID_INVALID
  *   - @ref RT_EXCEPTION_INDEX_OUT_OF_BOUNDS
  *   - @ref RT_EXCEPTION_STACK_OVERFLOW
  *   - @ref RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS
  *   - @ref RT_EXCEPTION_INVALID_RAY
  *   - @ref RT_EXCEPTION_INTERNAL_ERROR
  *   - @ref RT_EXCEPTION_USER
  *   - @ref RT_EXCEPTION_ALL
  *
  *
  * @ref RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS verifies that accesses to the ray payload are
  * within valid bounds. This exception is only supported with the RTX execution strategy.
  *
  * @ref RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS verifies that the exception code passed
  * to @ref rtThrow is within the valid range from RT_EXCEPTION_USER to RT_EXCEPTION_USER_MAX.
  *
  * @ref RT_EXCEPTION_TRACE_DEPTH_EXCEEDED verifies that the depth of the @ref rtTrace
  * tree does not exceed the configured trace depth (see @ref rtContextSetMaxTraceDepth). This
  * exception is only supported with the RTX execution strategy.
  *
  * @ref RT_EXCEPTION_TEXTURE_ID_INVALID verifies that every access of a texture id is
  * valid, including use of RT_TEXTURE_ID_NULL and IDs out of bounds.
  *
  * @ref RT_EXCEPTION_BUFFER_ID_INVALID verifies that every access of a buffer id is
  * valid, including use of RT_BUFFER_ID_NULL and IDs out of bounds.
  *
  * @ref RT_EXCEPTION_INDEX_OUT_OF_BOUNDS checks that @ref rtIntersectChild and @ref
  * rtReportIntersection are called with a valid index.
  *
  * @ref RT_EXCEPTION_STACK_OVERFLOW checks the runtime stack against overflow. The most common
  * cause for an overflow is a too small trace depth (see @ref rtContextSetMaxTraceDepth). In rare
  * cases, stack overflows might not be detected unless @ref RT_EXCEPTION_TRACE_DEPTH_EXCEEDED is
  * enabled as well.
  *
  * @ref RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS checks every read and write access to
  * @ref rtBuffer objects to be within valid bounds. This exception is supported with the RTX
  * execution strategy only.
  *
  * @ref RT_EXCEPTION_INVALID_RAY checks the each ray's origin and direction values
  * against \a NaNs and \a infinity values.
  *
  * @ref RT_EXCEPTION_INTERNAL_ERROR indicates an unexpected internal error in the
  * runtime.
  *
  * @ref RT_EXCEPTION_USER is used to enable or disable all user-defined exceptions. See
  * @ref rtThrow for more information.
  *
  * @ref RT_EXCEPTION_ALL is a placeholder value which can be used to enable or disable
  * all possible exceptions with a single call to @ref rtContextSetExceptionEnabled.
  *
  * By default, @ref RT_EXCEPTION_STACK_OVERFLOW is enabled and all other exceptions are
  * disabled.
  *
  * @param[in]   context     The context for which the exception is to be enabled or disabled
  * @param[in]   exception   The exception which is to be enabled or disabled
  * @param[in]   enabled     Nonzero to enable the exception, \a 0 to disable the exception
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetExceptionEnabled was introduced in OptiX 1.1.
  *
  * <B>See also</B>
  * @ref rtContextGetExceptionEnabled,
  * @ref rtContextSetExceptionProgram,
  * @ref rtContextGetExceptionProgram,
  * @ref rtGetExceptionCode,
  * @ref rtThrow,
  * @ref rtPrintExceptionDetails,
  * @ref RTexception
  *
  */
  RTresult RTAPI rtContextSetExceptionEnabled(RTcontext context, RTexception exception, int enabled);

  /**
  * @brief Query whether a specified exception is enabled
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetExceptionEnabled passes back \a 1 in \a *enabled if the given exception is
  * enabled, \a 0 otherwise. \a exception specifies the type of exception to be queried. For a list
  * of available types, see @ref rtContextSetExceptionEnabled. If \a exception
  * is @ref RT_EXCEPTION_ALL, \a enabled is set to \a 1 only if all possible
  * exceptions are enabled.
  *
  * @param[in]   context     The context to be queried
  * @param[in]   exception   The exception of which to query the state
  * @param[out]  enabled     Return parameter to store whether the exception is enabled
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetExceptionEnabled was introduced in OptiX 1.1.
  *
  * <B>See also</B>
  * @ref rtContextSetExceptionEnabled,
  * @ref rtContextSetExceptionProgram,
  * @ref rtContextGetExceptionProgram,
  * @ref rtGetExceptionCode,
  * @ref rtThrow,
  * @ref rtPrintExceptionDetails,
  * @ref RTexception
  *
  */
  RTresult RTAPI rtContextGetExceptionEnabled(RTcontext context, RTexception exception, int* enabled);

  /**
  * @brief Sets the number of ray types for a given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetRayTypeCount Sets the number of ray types associated with the given
  * context.
  *
  * @param[in]   context         The context node
  * @param[in]   rayTypeCount    The number of ray types to be used
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetRayTypeCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetRayTypeCount
  *
  */
  RTresult RTAPI rtContextSetRayTypeCount(RTcontext context, unsigned int rayTypeCount);

  /**
  * @brief Query the number of ray types associated with this
  * context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetRayTypeCount passes back the number of entry points associated with
  * this context in \a rayTypeCount.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a
  * NULL pointer.
  *
  * @param[in]   context         The context node to be queried
  * @param[out]  rayTypeCount    Return parameter to store the number of ray types
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetRayTypeCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextSetRayTypeCount
  *
  */
  RTresult RTAPI rtContextGetRayTypeCount(RTcontext context, unsigned int* rayTypeCount);

  /**
  * @brief Specifies the miss program for a given context ray type
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetMissProgram sets \a context's miss program associated with ray type
  * \a rayTypeIndex. @ref RT_ERROR_INVALID_VALUE is returned if \a rayTypeIndex
  * is outside of the range [\a 0, @ref rtContextGetRayTypeCount \a -1].
  *
  * @param[in]   context          The context node to which the miss program will be added
  * @param[in]   rayTypeIndex     The ray type the program will be associated with
  * @param[in]   program          The miss program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtContextSetMissProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetRayTypeCount,
  * @ref rtContextGetMissProgram
  *
  */
  RTresult RTAPI rtContextSetMissProgram(RTcontext context, unsigned int rayTypeIndex, RTprogram program);

  /**
  * @brief Queries the miss program associated with the given
  * context and ray type
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetMissProgram passes back the miss program associated with the
  * given context and ray type.  This program is set via @ref rtContextSetMissProgram.
  * Returns @ref RT_ERROR_INVALID_VALUE if given an invalid ray type index or a \a NULL pointer.
  *
  * @param[in]   context          The context node associated with the miss program
  * @param[in]   rayTypeIndex     The ray type index for the desired miss program
  * @param[out]  program          Return parameter to store the miss program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetMissProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextSetMissProgram,
  * @ref rtContextGetRayTypeCount
  *
  */
  RTresult RTAPI rtContextGetMissProgram(RTcontext context, unsigned int rayTypeIndex, RTprogram* program);

  /**
  * @brief Gets an RTtexturesampler corresponding to the texture id
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetTextureSamplerFromId returns a handle to the texture sampler in \a *sampler
  * corresponding to the \a samplerId supplied.  If \a samplerId does not map to a valid
  * texture handle, \a *sampler is \a NULL or if \a context is invalid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   context     The context the sampler should be originated from
  * @param[in]   samplerId   The ID of the sampler to query
  * @param[out]  sampler     The return handle for the sampler object corresponding to the samplerId
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetTextureSamplerFromId was introduced in OptiX 3.5.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetId
  *
  */
  RTresult RTAPI rtContextGetTextureSamplerFromId(RTcontext context, int samplerId, RTtexturesampler* sampler);

  /**
  * Deprecated in OptiX 4.0. Calling this function has no effect. The kernel is automatically compiled at launch if needed.
  *
  */
  RTresult RTAPI rtContextCompile(RTcontext context);

  /**
  * @brief Executes the computation kernel for a given context
  *
  * @ingroup rtContextLaunch
  *
  * <B>Description</B>
  *
  * @ref rtContextLaunch "rtContextLaunch" functions execute the computation kernel associated with the
  * given context.  If the context has not yet been compiled, or if the context has been
  * modified since the last compile, @ref rtContextLaunch "rtContextLaunch" will recompile the kernel
  * internally.  Acceleration structures of the context which are marked dirty will be
  * updated and their dirty flags will be cleared.  Similarly, validation will occur if
  * necessary.  The ray generation program specified by \a entryPointIndex will be
  * invoked once for every element (pixel or voxel) of the computation grid specified by
  * \a width, \a height, and \a depth.
  *
  * For 3D launches, the product of \a width and \a depth must be smaller than 4294967296 (2^32).
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_INVALID_SOURCE
  * - @ref RT_ERROR_LAUNCH_FAILED
  *
  * <B>History</B>
  *
  * @ref rtContextLaunch "rtContextLaunch" was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextGetRunningState,
  * @ref rtContextValidate
  *
  */
  /**
  * @ingroup rtContextLaunch
  * @param[in]   context                                    The context to be executed
  * @param[in]   entryPointIndex                            The initial entry point into kernel
  * @param[in]   width                                      Width of the computation grid
  */
  RTresult RTAPI rtContextLaunch1D(RTcontext context, unsigned int entryPointIndex, RTsize width);
  /**
  * @ingroup rtContextLaunch
  * @param[in]   context                                    The context to be executed
  * @param[in]   entryPointIndex                            The initial entry point into kernel
  * @param[in]   width                                      Width of the computation grid
  * @param[in]   height                                     Height of the computation grid
  */
  RTresult RTAPI rtContextLaunch2D(RTcontext context, unsigned int entryPointIndex, RTsize width, RTsize height);
  /**
  * @ingroup rtContextLaunch
  * @param[in]   context                                    The context to be executed
  * @param[in]   entryPointIndex                            The initial entry point into kernel
  * @param[in]   width                                      Width of the computation grid
  * @param[in]   height                                     Height of the computation grid
  * @param[in]   depth                                      Depth of the computation grid
  */
  RTresult RTAPI rtContextLaunch3D(RTcontext context, unsigned int entryPointIndex, RTsize width, RTsize height, RTsize depth);

  /**
  * @brief Query whether the given context is currently
  * running
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * This function is currently unimplemented and it is provided as a placeholder for a future implementation.
  *
  * @param[in]   context   The context node to be queried
  * @param[out]  running   Return parameter to store the running state
  *
  * <B>Return values</B>
  *
  * Since unimplemented, this function will always throw an assertion failure.
  *
  * <B>History</B>
  *
  * @ref rtContextGetRunningState was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextLaunch1D,
  * @ref rtContextLaunch2D,
  * @ref rtContextLaunch3D
  *
  */
  RTresult RTAPI rtContextGetRunningState(RTcontext context, int* running);

  /**
  * @brief Executes a Progressive Launch for a given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * Starts the (potentially parallel) generation of subframes for progressive rendering. If
  * \a maxSubframes is zero, there is no limit on the number of subframes generated. The
  * generated subframes are automatically composited into a single result and streamed to
  * the client at regular intervals, where they can be read by mapping an associated stream
  * buffer. An application can therefore initiate a progressive launch, and then repeatedly
  * map and display the contents of the stream buffer in order to visualize the progressive
  * refinement of the image.
  *
  * The call is nonblocking. A polling approach should be used to decide when to map and
  * display the stream buffer contents (see @ref rtBufferGetProgressiveUpdateReady). If a
  * progressive launch is already in progress at the time of the call and its parameters
  * match the initial launch, the call has no effect. Otherwise, the accumulated result will be
  * reset and a new progressive launch will be started.
  *
  * If any other OptiX function is called while a progressive launch is in progress, it will
  * cause the launch to stop generating new subframes (however, subframes that have
  * already been generated and are currently in flight may still arrive at the client). The only
  * exceptions to this rule are the operations to map a stream buffer, issuing another
  * progressive launch with unchanged parameters, and polling for an update. Those
  * exceptions do not cause the progressive launch to stop generating subframes.
  *
  * There is no guarantee that the call actually produces any subframes, especially if
  * @ref rtContextLaunchProgressive2D and other OptiX commands are called in short
  * succession. For example, during an animation, @ref rtVariableSet calls may be tightly
  * interleaved with progressive launches, and when rendering remotely the server may decide to skip some of the
  * launches in order to avoid a large backlog in the command pipeline.
  *
  * @param[in]   context                The context in which the launch is to be executed
  * @param[in]   entryIndex             The initial entry point into kernel
  * @param[in]   width                  Width of the computation grid
  * @param[in]   height                 Height of the computation grid
  * @param[in]   maxSubframes           The maximum number of subframes to be generated. Set to zero to generate an unlimited number of subframes
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_LAUNCH_FAILED
  *
  * <B>History</B>
  *
  * @ref rtContextLaunchProgressive2D was introduced in OptiX 3.8.
  *
  * <B>See also</B>
  * @ref rtContextStopProgressive
  * @ref rtBufferGetProgressiveUpdateReady
  *
  */
  RTresult RTAPI rtContextLaunchProgressive2D(RTcontext context, unsigned int entryIndex, RTsize width, RTsize height, unsigned int maxSubframes);

  /**
  * @brief Stops a Progressive Launch
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * If a progressive launch is currently in progress, calling @ref rtContextStopProgressive
  * terminates it. Otherwise, the call has no effect. If a launch is stopped using this function,
  * no further subframes will arrive at the client, even if they have already been generated
  * by the server and are currently in flight.
  *
  * This call should only be used if the application must guarantee that frames generated by
  * previous progressive launches won't be accessed. Do not call @ref rtContextStopProgressive in
  * the main rendering loop if the goal is only to change OptiX state (e.g. rtVariable values).
  * The call is unnecessary in that case and will degrade performance.
  *
  * @param[in]   context                The context associated with the progressive launch
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  *
  * <B>History</B>
  *
  * @ref rtContextStopProgressive was introduced in OptiX 3.8.
  *
  * <B>See also</B>
  * @ref rtContextLaunchProgressive2D
  *
  */
  RTresult RTAPI rtContextStopProgressive(RTcontext context);

  /**
  * @brief Enable or disable text printing from programs
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetPrintEnabled is used to control whether text printing in programs
  * through @ref rtPrintf is currently enabled for this context.
  *
  * @param[in]   context   The context for which printing is to be enabled or disabled
  * @param[in]   enabled   Setting this parameter to a nonzero value enables printing, \a 0 disables printing
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetPrintEnabled was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtPrintf,
  * @ref rtContextGetPrintEnabled,
  * @ref rtContextSetPrintBufferSize,
  * @ref rtContextGetPrintBufferSize,
  * @ref rtContextSetPrintLaunchIndex,
  * @ref rtContextGetPrintLaunchIndex
  *
  */
  RTresult RTAPI rtContextSetPrintEnabled(RTcontext context, int enabled);

  /**
  * @brief Query whether text printing from programs
  * is enabled
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetPrintEnabled passes back \a 1 if text printing from programs through
  * @ref rtPrintf is currently enabled for this context; \a 0 otherwise.  Returns @ref
  * RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
  *
  * @param[in]   context   The context to be queried
  * @param[out]  enabled   Return parameter to store whether printing is enabled
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetPrintEnabled was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtPrintf,
  * @ref rtContextSetPrintEnabled,
  * @ref rtContextSetPrintBufferSize,
  * @ref rtContextGetPrintBufferSize,
  * @ref rtContextSetPrintLaunchIndex,
  * @ref rtContextGetPrintLaunchIndex
  *
  */
  RTresult RTAPI rtContextGetPrintEnabled(RTcontext context, int* enabled);

  /**
  * @brief Set the size of the print buffer
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetPrintBufferSize is used to set the buffer size available to hold
  * data generated by @ref rtPrintf.
  * Returns @ref RT_ERROR_INVALID_VALUE if it is called after the first invocation of rtContextLaunch.
  *
  *
  * @param[in]   context             The context for which to set the print buffer size
  * @param[in]   bufferSizeBytes     The print buffer size in bytes
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetPrintBufferSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtPrintf,
  * @ref rtContextSetPrintEnabled,
  * @ref rtContextGetPrintEnabled,
  * @ref rtContextGetPrintBufferSize,
  * @ref rtContextSetPrintLaunchIndex,
  * @ref rtContextGetPrintLaunchIndex
  *
  */
  RTresult RTAPI rtContextSetPrintBufferSize(RTcontext context, RTsize bufferSizeBytes);

  /**
  * @brief Get the current size of the print buffer
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetPrintBufferSize is used to query the buffer size available to hold
  * data generated by @ref rtPrintf. Returns @ref RT_ERROR_INVALID_VALUE if passed a \a
  * NULL pointer.
  *
  * @param[in]   context             The context from which to query the print buffer size
  * @param[out]  bufferSizeBytes     The returned print buffer size in bytes
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetPrintBufferSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtPrintf,
  * @ref rtContextSetPrintEnabled,
  * @ref rtContextGetPrintEnabled,
  * @ref rtContextSetPrintBufferSize,
  * @ref rtContextSetPrintLaunchIndex,
  * @ref rtContextGetPrintLaunchIndex
  *
  */
  RTresult RTAPI rtContextGetPrintBufferSize(RTcontext context, RTsize* bufferSizeBytes);

  /**
  * @brief Sets the active launch index to limit text output
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextSetPrintLaunchIndex is used to control for which launch indices @ref
  * rtPrintf generates output. The initial value of (x,y,z) is (\a -1,\a -1,\a -1), which
  * generates output for all indices.
  *
  * @param[in]   context   The context for which to set the print launch index
  * @param[in]   x         The launch index in the x dimension to which to limit the output of @ref rtPrintf invocations.
  * If set to \a -1, output is generated for all launch indices in the x dimension
  * @param[in]   y         The launch index in the y dimension to which to limit the output of @ref rtPrintf invocations.
  * If set to \a -1, output is generated for all launch indices in the y dimension
  * @param[in]   z         The launch index in the z dimension to which to limit the output of @ref rtPrintf invocations.
  * If set to \a -1, output is generated for all launch indices in the z dimension
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextSetPrintLaunchIndex was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtPrintf,
  * @ref rtContextGetPrintEnabled,
  * @ref rtContextSetPrintEnabled,
  * @ref rtContextSetPrintBufferSize,
  * @ref rtContextGetPrintBufferSize,
  * @ref rtContextGetPrintLaunchIndex
  *
  */
  RTresult RTAPI rtContextSetPrintLaunchIndex(RTcontext context, int x, int y, int z);

  /**
  * @brief Gets the active print launch index
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetPrintLaunchIndex is used to query for which launch indices @ref
  * rtPrintf generates output. The initial value of (x,y,z) is (\a -1,\a -1,\a -1), which
  * generates output for all indices.
  *
  * @param[in]   context   The context from which to query the print launch index
  * @param[out]  x         Returns the launch index in the x dimension to which the output of @ref rtPrintf invocations
  * is limited. Will not be written to if a \a NULL pointer is passed
  * @param[out]  y         Returns the launch index in the y dimension to which the output of @ref rtPrintf invocations
  * is limited. Will not be written to if a \a NULL pointer is passed
  * @param[out]  z         Returns the launch index in the z dimension to which the output of @ref rtPrintf invocations
  * is limited. Will not be written to if a \a NULL pointer is passed
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetPrintLaunchIndex was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtPrintf,
  * @ref rtContextGetPrintEnabled,
  * @ref rtContextSetPrintEnabled,
  * @ref rtContextSetPrintBufferSize,
  * @ref rtContextGetPrintBufferSize,
  * @ref rtContextSetPrintLaunchIndex
  *
  */
  RTresult RTAPI rtContextGetPrintLaunchIndex(RTcontext context, int* x, int* y, int* z);

  /**
  * @brief Declares a new named variable associated with this
  * context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextDeclareVariable - Declares a new variable named \a name and associated
  * with this context.  Only a single variable of a given name can exist for a given
  * context and any attempt to create multiple variables with the same name will cause a
  * failure with a return value of @ref RT_ERROR_VARIABLE_REDECLARED.  Returns @ref
  * RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.  Return @ref
  * RT_ERROR_ILLEGAL_SYMBOL if \a name is not syntactically valid.
  *
  * @param[in]   context   The context node to which the variable will be attached
  * @param[in]   name      The name that identifies the variable to be queried
  * @param[out]  v         Pointer to variable handle used to return the new object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_VARIABLE_REDECLARED
  *
  * <B>History</B>
  *
  * @ref rtContextDeclareVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryDeclareVariable,
  * @ref rtGeometryInstanceDeclareVariable,
  * @ref rtMaterialDeclareVariable,
  * @ref rtProgramDeclareVariable,
  * @ref rtSelectorDeclareVariable,
  * @ref rtContextGetVariable,
  * @ref rtContextGetVariableCount,
  * @ref rtContextQueryVariable,
  * @ref rtContextRemoveVariable
  *
  */
  RTresult RTAPI rtContextDeclareVariable(RTcontext context, const char* name, RTvariable* v);

  /**
  * @brief Returns a named variable associated with this context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextQueryVariable queries a variable identified by the string \a name
  * from \a context and stores the result in \a *v. A variable must
  * be declared with @ref rtContextDeclareVariable before it can be queried, otherwise \a *v will be set to \a NULL.
  * @ref RT_ERROR_INVALID_VALUE will be returned if \a name or \a v is \a NULL.
  *
  * @param[in]   context   The context node to query a variable from
  * @param[in]   name      The name that identifies the variable to be queried
  * @param[out]  v         Return value to store the queried variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextQueryVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryQueryVariable,
  * @ref rtGeometryInstanceQueryVariable,
  * @ref rtMaterialQueryVariable,
  * @ref rtProgramQueryVariable,
  * @ref rtSelectorQueryVariable,
  * @ref rtContextDeclareVariable,
  * @ref rtContextGetVariableCount,
  * @ref rtContextGetVariable,
  * @ref rtContextRemoveVariable
  *
  */
  RTresult RTAPI rtContextQueryVariable(RTcontext context, const char* name, RTvariable* v);

  /**
  * @brief Removes a variable from the given context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextRemoveVariable removes variable \a v from \a context if present.
  * Returns @ref RT_ERROR_VARIABLE_NOT_FOUND if the variable is not attached to this
  * context. Returns @ref RT_ERROR_INVALID_VALUE if passed an invalid variable.
  *
  * @param[in]   context   The context node from which to remove a variable
  * @param[in]   v         The variable to be removed
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtContextRemoveVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryRemoveVariable,
  * @ref rtGeometryInstanceRemoveVariable,
  * @ref rtMaterialRemoveVariable,
  * @ref rtProgramRemoveVariable,
  * @ref rtSelectorRemoveVariable,
  * @ref rtContextDeclareVariable,
  * @ref rtContextGetVariable,
  * @ref rtContextGetVariableCount,
  * @ref rtContextQueryVariable,
  *
  */
  RTresult RTAPI rtContextRemoveVariable(RTcontext context, RTvariable v);

  /**
  * @brief Returns the number of variables associated
  * with this context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetVariableCount returns the number of variables that are currently
  * attached to \a context.  Returns @ref RT_ERROR_INVALID_VALUE if passed a \a NULL pointer.
  *
  * @param[in]   context   The context to be queried for number of attached variables
  * @param[out]  count     Return parameter to store the number of variables
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetVariableCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetVariableCount,
  * @ref rtGeometryInstanceGetVariableCount,
  * @ref rtMaterialGetVariableCount,
  * @ref rtProgramGetVariableCount,
  * @ref rtSelectorGetVariable,
  * @ref rtContextDeclareVariable,
  * @ref rtContextGetVariable,
  * @ref rtContextQueryVariable,
  * @ref rtContextRemoveVariable
  *
  */
  RTresult RTAPI rtContextGetVariableCount(RTcontext context, unsigned int* count);

  /**
  * @brief Queries an indexed variable associated with this context
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetVariable queries the variable at position \a index in the
  * variable array from \a context and stores the result in the parameter \a v.
  * A variable must be declared first with @ref rtContextDeclareVariable and
  * \a index must be in the range [\a 0, @ref rtContextGetVariableCount \a -1].
  *
  * @param[in]   context   The context node to be queried for an indexed variable
  * @param[in]   index     The index that identifies the variable to be queried
  * @param[out]  v         Return value to store the queried variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetVariable,
  * @ref rtGeometryInstanceGetVariable,
  * @ref rtMaterialGetVariable,
  * @ref rtProgramGetVariable,
  * @ref rtSelectorGetVariable,
  * @ref rtContextDeclareVariable,
  * @ref rtContextGetVariableCount,
  * @ref rtContextQueryVariable,
  * @ref rtContextRemoveVariable
  *
  */
  RTresult RTAPI rtContextGetVariable(RTcontext context, unsigned int index, RTvariable* v);

/************************************
 **
 **    Program object
 **
 ***********************************/

  /**
  * @brief Creates a new program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramCreateFromPTXString allocates and returns a handle to a new program
  * object.  The program is created from PTX code held in the \a NULL-terminated string \a
  * ptx from function \a programName.
  *
  * @param[in]   context        The context to create the program in
  * @param[in]   ptx            The string containing the PTX code
  * @param[in]   programName    The name of the PTX function to create the program from
  * @param[in]   program        Handle to the program to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_INVALID_SOURCE
  *
  * <B>History</B>
  *
  * @ref rtProgramCreateFromPTXString was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref RT_PROGRAM,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXFiles,
  * @ref rtProgramCreateFromPTXStrings,
  * @ref rtProgramDestroy
  *
  */
  RTresult RTAPI rtProgramCreateFromPTXString(RTcontext context, const char* ptx, const char* programName, RTprogram* program);

  /**
  * @brief Creates a new program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramCreateFromPTXStrings allocates and returns a handle to a new program
  * object.  The program is created by linking PTX code held in one or more \a NULL-terminated strings.
  * C-style linking rules apply: global functions and variables are visible across input strings and must
  * be defined uniquely.  There must be a visible function for \a programName.
  *
  * @param[in]   context        The context to create the program in
  * @param[in]   n              Number of ptx strings
  * @param[in]   ptxStrings     Array of strings containing PTX code
  * @param[in]   programName    The name of the PTX function to create the program from
  * @param[in]   program        Handle to the program to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_INVALID_SOURCE
  *
  * <B>History</B>
  *
  * <B>See also</B>
  * @ref RT_PROGRAM,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXFiles,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtProgramDestroy
  *
  */
  RTresult RTAPI rtProgramCreateFromPTXStrings(RTcontext context, unsigned int n, const char** ptxStrings, const char* programName, RTprogram* program);

  /**
  * @brief Creates a new program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramCreateFromPTXFile allocates and returns a handle to a new program object.
  * The program is created from PTX code held in \a filename from function \a programName.
  *
  * @param[in]   context        The context to create the program in
  * @param[in]   filename       Path to the file containing the PTX code
  * @param[in]   programName    The name of the PTX function to create the program from
  * @param[in]   program        Handle to the program to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_INVALID_SOURCE
  * - @ref RT_ERROR_FILE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtProgramCreateFromPTXFile was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref RT_PROGRAM,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtProgramDestroy
  *
  */
  RTresult RTAPI rtProgramCreateFromPTXFile(RTcontext context, const char* filename, const char* programName, RTprogram* program);

  /**
  * @brief Creates a new program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramCreateFromPTXFiles allocates and returns a handle to a new program object.
  * The program is created by linking PTX code held in one or more files.
  * C-style linking rules apply: global functions and variables are visible across input files and must
  * be defined uniquely.  There must be a visible function for \a programName.
  *
  * @param[in]   context        The context to create the program in
  * @param[in]   n              Number of filenames
  * @param[in]   filenames      Array of one or more paths to files containing PTX code
  * @param[in]   programName    The name of the PTX function to create the program from
  * @param[in]   program        Handle to the program to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_INVALID_SOURCE
  * - @ref RT_ERROR_FILE_NOT_FOUND
  *
  * <B>History</B>
  *
  * <B>See also</B>
  * @ref RT_PROGRAM,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtProgramCreateFromPTXStrings,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromProgram,
  * @ref rtProgramDestroy
  *
  */
  RTresult RTAPI rtProgramCreateFromPTXFiles(RTcontext context, unsigned int n, const char** filenames, const char* programName, RTprogram* program);

    /**
  * @brief Creates a new program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramCreateFromProgram allocates and returns a handle to a new program object.
  * The program code is taken from another program, but none of the other attributes are taken.
  *
  * @param[in]   context        The context to create the program in
  * @param[in]   program_in     The program whose program code to use.
  * @param[in]   program_out    Handle to the program to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * <B>See also</B>
  * @ref RT_PROGRAM,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtProgramCreateFromPTXStrings,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramDestroy
  *
  */
  RTresult RTAPI rtProgramCreateFromProgram(RTcontext context, RTprogram program_in, RTprogram* program_out);


  /**
  * @brief Destroys a program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramDestroy removes \a program from its context and deletes it.
  * \a program should be a value returned by \a rtProgramCreate*.
  * Associated variables declared via @ref rtProgramDeclareVariable are destroyed.
  * After the call, \a program is no longer a valid handle.
  *
  * @param[in]   program   Handle of the program to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtProgramDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString
  *
  */
  RTresult RTAPI rtProgramDestroy(RTprogram program);

  /**
  * @brief Validates the state of a program
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramValidate checks \a program for completeness.  If \a program or any of
  * the objects attached to \a program are not valid, returns @ref
  * RT_ERROR_INVALID_CONTEXT.
  *
  * @param[in]   program   The program to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtProgramValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString
  *
  */
  RTresult RTAPI rtProgramValidate(RTprogram program);

  /**
  * @brief Gets the context object that created a program
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramGetContext returns a handle to the context object that was used to
  * create \a program. Returns @ref RT_ERROR_INVALID_VALUE if \a context is \a NULL.
  *
  * @param[in]   program   The program to be queried for its context object
  * @param[out]  context   The return handle for the requested context object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtProgramGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtProgramGetContext(RTprogram program, RTcontext* context);

  /**
  * @brief Declares a new named variable associated with a program
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramDeclareVariable declares a new variable, \a name, and associates it with
  * the program.  A variable can only be declared with the same name once on the program.
  * Any attempt to declare multiple variables with the same name will cause the call to
  * fail and return @ref RT_ERROR_VARIABLE_REDECLARED.  If \a name or\a v is \a NULL
  * returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   program   The program the declared variable will be attached to
  * @param[in]   name      The name of the variable to be created
  * @param[out]  v         Return handle to the variable to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_REDECLARED
  * - @ref RT_ERROR_ILLEGAL_SYMBOL
  *
  * <B>History</B>
  *
  * @ref rtProgramDeclareVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramRemoveVariable,
  * @ref rtProgramGetVariable,
  * @ref rtProgramGetVariableCount,
  * @ref rtProgramQueryVariable
  *
  */
  RTresult RTAPI rtProgramDeclareVariable(RTprogram program, const char* name, RTvariable* v);

  /**
  * @brief Returns a handle to the named variable attached to a program
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramQueryVariable returns a handle to a variable object, in \a *v, attached
  * to \a program referenced by the \a NULL-terminated string \a name.  If \a name is not
  * the name of a variable attached to \a program, \a *v will be \a NULL after the call.
  *
  * @param[in]   program   The program to be queried for the named variable
  * @param[in]   name      The name of the program to be queried for
  * @param[out]  v         The return handle to the variable object
  * @param  program   Handle to the program to be created
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtProgramQueryVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramDeclareVariable,
  * @ref rtProgramRemoveVariable,
  * @ref rtProgramGetVariable,
  * @ref rtProgramGetVariableCount
  *
  */
  RTresult RTAPI rtProgramQueryVariable(RTprogram program, const char* name, RTvariable* v);

  /**
  * @brief Removes the named variable from a program
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramRemoveVariable removes variable \a v from the \a program object.  Once a
  * variable has been removed from this program, another variable with the same name as
  * the removed variable may be declared.
  *
  * @param[in]   program   The program to remove the variable from
  * @param[in]   v         The variable to remove
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtProgramRemoveVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramDeclareVariable,
  * @ref rtProgramGetVariable,
  * @ref rtProgramGetVariableCount,
  * @ref rtProgramQueryVariable
  *
  */
  RTresult RTAPI rtProgramRemoveVariable(RTprogram program, RTvariable v);

  /**
  * @brief Returns the number of variables attached to a program
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramGetVariableCount returns, in \a *count, the number of variable objects that
  * have been attached to \a program.
  *
  * @param[in]   program   The program to be queried for its variable count
  * @param[out]  count     The return handle for the number of variables attached to this program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtProgramGetVariableCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramDeclareVariable,
  * @ref rtProgramRemoveVariable,
  * @ref rtProgramGetVariable,
  * @ref rtProgramQueryVariable
  *
  */
  RTresult RTAPI rtProgramGetVariableCount(RTprogram program, unsigned int* count);

  /**
  * @brief Returns a handle to a variable attached to a program by index
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramGetVariable returns a handle to a variable in \a *v attached to \a
  * program with @ref rtProgramDeclareVariable by \a index.  \a index must be between
  * 0 and one less than the value returned by @ref rtProgramGetVariableCount.  The order
  * in which variables are enumerated is not constant and may change as variables are
  * attached and removed from the program object.
  *
  * @param[in]   program   The program to be queried for the indexed variable object
  * @param[in]   index     The index of the variable to return
  * @param[out]  v         Return handle to the variable object specified by the index
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtProgramGetVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtProgramDeclareVariable,
  * @ref rtProgramRemoveVariable,
  * @ref rtProgramGetVariableCount,
  * @ref rtProgramQueryVariable
  *
  */
  RTresult RTAPI rtProgramGetVariable(RTprogram program, unsigned int index, RTvariable* v);


  /**
  * @brief Returns the ID for the Program object
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramGetId returns an ID for the provided program.  The returned ID is used
  * to reference \a program from device code.  If \a programId is \a NULL or the \a
  * program is not a valid \a RTprogram, returns @ref RT_ERROR_INVALID_VALUE.
  * @ref RT_PROGRAM_ID_NULL can be used as a sentinel for a non-existent program, since
  * this value will never be returned as a valid program id.
  *
  * @param[in]   program      The program to be queried for its id
  * @param[out]  programId    The returned ID of the program.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtProgramGetId was introduced in OptiX 3.6.
  *
  * <B>See also</B>
  * @ref rtContextGetProgramFromId
  *
  */
  RTresult RTAPI rtProgramGetId(RTprogram program, int* programId);

  /**
  * @brief Sets the program ids that may potentially be called at a call site
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtProgramCallsiteSetPotentialCallees specifies the program IDs of potential
  * callees at the call site in the \a program identified by \a name to the list
  * provided in \a ids. If \a program is bit a valid \a RTprogram or the \a program
  * does not contain a call site with the identifier \a name or \a ids contains
  * invalid program ids, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in] program        The program that includes the call site.
  * @param[in] name           The string identifier for the call site to modify.
  * @param[in] ids            The program IDs of the programs that may potentially be called at the call site
  * @param[in] numIds         The size of the array passed in for \a ids.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtProgramCallsiteSetPotentialCallees was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtProgramGetId
  *
  */
  RTresult RTAPI rtProgramCallsiteSetPotentialCallees( RTprogram program, const char* name, const int* ids, int numIds );

  /**
  * @brief Gets an RTprogram corresponding to the program id
  *
  * @ingroup Program
  *
  * <B>Description</B>
  *
  * @ref rtContextGetProgramFromId returns a handle to the program in \a *program
  * corresponding to the \a programId supplied.  If \a programId is not a valid
  * program handle, \a *program is set to \a NULL. Returns @ref RT_ERROR_INVALID_VALUE
  * if \a context is invalid or \a programId is not a valid program handle.
  *
  * @param[in]   context     The context the program should be originated from
  * @param[in]   programId   The ID of the program to query
  * @param[out]  program     The return handle for the program object corresponding to the programId
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetProgramFromId was introduced in OptiX 3.6.
  *
  * <B>See also</B>
  * @ref rtProgramGetId
  *
  */
  RTresult RTAPI rtContextGetProgramFromId(RTcontext context, int programId, RTprogram* program);

/************************************
 **
 **    Group object
 **
 ***********************************/

  /**
  * @brief Creates a new group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupCreate creates a new group within a context. \a context
  * specifies the target context, and should be a value returned by
  * @ref rtContextCreate.  Sets \a *group to the handle of a newly created group
  * within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a group is \a NULL.
  *
  * @param[in]   context   Specifies a context within which to create a new group
  * @param[out]  group     Returns a newly created group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupDestroy,
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtGroupCreate(RTcontext context, RTgroup* group);

  /**
  * @brief Destroys a group node
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupDestroy removes \a group from its context and deletes it.
  * \a group should be a value returned by @ref rtGroupCreate.
  * No child graph nodes are destroyed.
  * After the call, \a group is no longer a valid handle.
  *
  * @param[in]   group   Handle of the group node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupCreate
  *
  */
  RTresult RTAPI rtGroupDestroy(RTgroup group);

  /**
  * @brief Verifies the state of the group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupValidate checks \a group for completeness. If \a group or
  * any of the objects attached to \a group are not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   group   Specifies the group to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupCreate
  *
  */
  RTresult RTAPI rtGroupValidate(RTgroup group);

  /**
  * @brief Returns the context associated with a group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupGetContext queries a group for its associated context.
  * \a group specifies the group to query, and must be a value returned by
  * @ref rtGroupCreate. Sets \a *context to the context
  * associated with \a group.
  *
  * @param[in]   group     Specifies the group to query
  * @param[out]  context   Returns the context associated with the group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextCreate,
  * @ref rtGroupCreate
  *
  */
  RTresult RTAPI rtGroupGetContext(RTgroup group, RTcontext* context);

  /**
  * @brief Set the acceleration structure for a group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupSetAcceleration attaches an acceleration structure to a group. The acceleration
  * structure must have been previously created using @ref rtAccelerationCreate. Every group is
  * required to have an acceleration structure assigned in order to pass validation. The acceleration
  * structure will be built over the children of the group. For example, if an acceleration structure
  * is attached to a group that has a selector, a geometry group, and a transform child,
  * the acceleration structure will be built over the bounding volumes of these three objects.
  *
  * Note that it is legal to attach a single RTacceleration object to multiple groups, as long as
  * the underlying bounds of the children are the same. For example, if another group has three
  * children which are known to have the same bounding volumes as the ones in the example
  * above, the two groups can share an acceleration structure, thus saving build time. This is
  * true even if the details of the children, such as the actual type of a node or its geometry
  * content, differ from the first set of group children. All that is required is for a child
  * node at a given index to have the same bounds as the other group's child node at the same index.
  *
  * Sharing an acceleration structure this way corresponds to attaching an acceleration structure
  * to multiple geometry groups at lower graph levels using @ref rtGeometryGroupSetAcceleration.
  *
  * @param[in]   group          The group handle
  * @param[in]   acceleration   The acceleration structure to attach to the group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupSetAcceleration was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupGetAcceleration,
  * @ref rtAccelerationCreate,
  * @ref rtGeometryGroupSetAcceleration
  *
  */
  RTresult RTAPI rtGroupSetAcceleration(RTgroup group, RTacceleration acceleration);

  /**
   * @brief Sets the visibility mask for a group.
   *
   * @ingroup GroupNode
   *
   * <B>Description</B>
   * Geometry is intersected by rays if the ray's @ref RTvisibilitymask shares at
   * least one bit with the group's mask. This mechanism allows for a number of
   * user-defined visibility groups that can be excluded from certain types of rays
   * as needed.
   * Note that the visibility mask is not checked for the root node of a trace call.
   * (It is assumed to be visible otherwise trace should not be called).
   * Note that the @pre mask is currently limited to 8 bits.
   *
   * @param[in] group   The group handle
   * @param[in] mask    A set of bits for which rays will intersect the group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGroupSetVisibilityMask was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryGroupSetVisibilityMask,
   * @ref rtGroupGetVisibilityMask,
   * @ref rtTrace
   */
  RTresult RTAPI rtGroupSetVisibilityMask( RTgroup group, RTvisibilitymask mask );

  /**
   * @brief Retrieves the visibility mask of a group.
   *
   * @ingroup GroupNode
   *
   * <B>Description</B>
   * See @ref rtGroupSetVisibilityMask for details.
   *
   * @param[in] group   The group handle
   * @param[out] mask   A set of bits for which rays will intersect the group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGroupGetVisibilityMask was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryGroupGetVisibilityMask,
   * @ref rtGroupSetVisibilityMask,
   * @ref rtTrace
   */
  RTresult RTAPI rtGroupGetVisibilityMask( RTgroup group, RTvisibilitymask* mask );

  /**
  * @brief Returns the acceleration structure attached to a group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupGetAcceleration returns the acceleration structure attached to a group using @ref rtGroupSetAcceleration.
  * If no acceleration structure has previously been set, \a *acceleration is set to \a NULL.
  *
  * @param[in]   group          The group handle
  * @param[out]  acceleration   The returned acceleration structure object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupGetAcceleration was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupSetAcceleration,
  * @ref rtAccelerationCreate
  *
  */
  RTresult RTAPI rtGroupGetAcceleration(RTgroup group, RTacceleration* acceleration);

  /**
  * @brief Sets the number of child nodes to be attached to the group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupSetChildCount specifies the number of child slots in this group. Potentially existing links to children
  * at indices greater than \a count-1 are removed. If the call increases the number of slots, the newly
  * created slots are empty and need to be filled using @ref rtGroupSetChild before validation.
  *
  * @param[in]   group   The parent group handle
  * @param[in]   count   Number of child slots to allocate for the group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupSetChildCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupGetChild,
  * @ref rtGroupGetChildCount,
  * @ref rtGroupGetChildType,
  * @ref rtGroupSetChild
  *
  */
  RTresult RTAPI rtGroupSetChildCount(RTgroup group, unsigned int count);

  /**
  * @brief Returns the number of child slots for a group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupGetChildCount returns the number of child slots allocated using @ref
  * rtGroupSetChildCount.  This includes empty slots which may not yet have actual children assigned
  * by @ref rtGroupSetChild.  Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  *
  * @param[in]   group   The parent group handle
  * @param[out]  count   Returned number of child slots
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupGetChildCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupSetChild,
  * @ref rtGroupGetChild,
  * @ref rtGroupSetChildCount,
  * @ref rtGroupGetChildType
  *
  */
  RTresult RTAPI rtGroupGetChildCount(RTgroup group, unsigned int* count);

  /**
  * @brief Attaches a child node to a group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * Attaches a new child node \a child to the parent node
  * \a group. \a index specifies the number of the slot where the child
  * node gets attached. A sufficient number of slots must be allocated
  * using @ref rtGroupSetChildCount.
  * Legal child node types are @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, and
  * @ref RTtransform.
  *
  * @param[in]   group   The parent group handle
  * @param[in]   index   The index in the parent's child slot array
  * @param[in]   child   The child node to be attached. Can be of type {@ref RTgroup, @ref RTselector,
  * @ref RTgeometrygroup, @ref RTtransform}
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupSetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupSetChildCount,
  * @ref rtGroupGetChildCount,
  * @ref rtGroupGetChild,
  * @ref rtGroupGetChildType
  *
  */
  RTresult RTAPI rtGroupSetChild(RTgroup group, unsigned int index, RTobject child);

  /**
  * @brief Returns a child node of a group
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupGetChild returns the child object at slot \a index of the parent \a group.
  * If no child has been assigned to the given slot, \a *child is set to \a NULL.
  * Returns @ref RT_ERROR_INVALID_VALUE if given an invalid child index or \a NULL pointer.
  *
  * @param[in]   group   The parent group handle
  * @param[in]   index   The index of the child slot to query
  * @param[out]  child   The returned child object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupGetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupSetChild,
  * @ref rtGroupSetChildCount,
  * @ref rtGroupGetChildCount,
  * @ref rtGroupGetChildType
  *
  */
  RTresult RTAPI rtGroupGetChild(RTgroup group, unsigned int index, RTobject* child);

  /**
  * @brief Get the type of a group child
  *
  * @ingroup GroupNode
  *
  * <B>Description</B>
  *
  * @ref rtGroupGetChildType returns the type of the group child at slot \a index.
  * If no child is associated with the given index, \a *type is set to
  * @ref RT_OBJECTTYPE_UNKNOWN and @ref RT_ERROR_INVALID_VALUE is returned.
  * Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  *
  * @param[in]   group   The parent group handle
  * @param[in]   index   The index of the child slot to query
  * @param[out]  type    The returned child type
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGroupGetChildType was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGroupSetChild,
  * @ref rtGroupGetChild,
  * @ref rtGroupSetChildCount,
  * @ref rtGroupGetChildCount
  *
  */
  RTresult RTAPI rtGroupGetChildType(RTgroup group, unsigned int index, RTobjecttype* type);

/************************************
 **
 **    Selector object
 **
 ***********************************/

  /**
  * @brief Creates a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * Creates a new Selector node within \a context. After calling
  * @ref rtSelectorCreate the new node is in an invalid state.  For the node
  * to be valid, a visit program must be assigned using
  * @ref rtSelectorSetVisitProgram. Furthermore, a number of (zero or
  * more) children can be attached by using @ref rtSelectorSetChildCount and
  * @ref rtSelectorSetChild. Sets \a *selector to the handle of a newly
  * created selector within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a selector is \a NULL.
  *
  * @param[in]   context    Specifies the rendering context of the Selector node
  * @param[out]  selector   New Selector node handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorDestroy,
  * @ref rtSelectorValidate,
  * @ref rtSelectorGetContext,
  * @ref rtSelectorSetVisitProgram,
  * @ref rtSelectorSetChildCount,
  * @ref rtSelectorSetChild
  *
  */
  RTresult RTAPI rtSelectorCreate(RTcontext context, RTselector* selector);

  /**
  * @brief Destroys a selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorDestroy removes \a selector from its context and deletes it.  \a selector should
  * be a value returned by @ref rtSelectorCreate.  Associated variables declared via @ref
  * rtSelectorDeclareVariable are destroyed, but no child graph nodes are destroyed.  After the
  * call, \a selector is no longer a valid handle.
  *
  * @param[in]   selector   Handle of the selector node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorCreate,
  * @ref rtSelectorValidate,
  * @ref rtSelectorGetContext
  *
  */
  RTresult RTAPI rtSelectorDestroy(RTselector selector);

  /**
  * @brief Checks a Selector node for internal consistency
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorValidate recursively checks consistency of the Selector
  * node \a selector and its children, i.e., it tries to validate the
  * whole model sub-tree with \a selector as root. For a Selector node to
  * be valid, it must be assigned a visit program, and the number of its
  * children must match the number specified by
  * @ref rtSelectorSetChildCount.
  *
  * @param[in]   selector   Selector root node of a model sub-tree to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorCreate,
  * @ref rtSelectorDestroy,
  * @ref rtSelectorGetContext,
  * @ref rtSelectorSetVisitProgram,
  * @ref rtSelectorSetChildCount,
  * @ref rtSelectorSetChild
  *
  */
  RTresult RTAPI rtSelectorValidate(RTselector selector);

  /**
  * @brief Returns the context of a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorGetContext returns in \a context the rendering context
  * in which the Selector node \a selector has been created.
  *
  * @param[in]   selector   Selector node handle
  * @param[out]  context    The context, \a selector belongs to
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorCreate,
  * @ref rtSelectorDestroy,
  * @ref rtSelectorValidate
  *
  */
  RTresult RTAPI rtSelectorGetContext(RTselector selector, RTcontext* context);

  /**
  * @brief Assigns a visit program to a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorSetVisitProgram specifies a visit program that is
  * executed when the Selector node \a selector gets visited by a ray
  * during traversal of the model graph. A visit program steers how
  * traversal of the Selectors's children is performed.  It usually
  * chooses only a single child to continue traversal, but is also allowed
  * to process zero or multiple children. Programs can be created from PTX
  * files using @ref rtProgramCreateFromPTXFile.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   program    Program handle associated with a visit program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtSelectorSetVisitProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorGetVisitProgram,
  * @ref rtProgramCreateFromPTXFile
  *
  */
  RTresult RTAPI rtSelectorSetVisitProgram(RTselector selector, RTprogram program);

  /**
  * @brief Returns the currently assigned visit program
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorGetVisitProgram returns in \a program a handle of the
  * visit program curently bound to \a selector.
  *
  * @param[in]   selector   Selector node handle
  * @param[out]  program    Current visit progam assigned to \a selector
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetVisitProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorSetVisitProgram
  *
  */
  RTresult RTAPI rtSelectorGetVisitProgram(RTselector selector, RTprogram* program);

  /**
  * @brief Specifies the number of child nodes to be
  * attached to a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorSetChildCount allocates a number of children slots,
  * i.e., it pre-defines the exact number of child nodes the parent
  * Selector node \a selector will have.  Child nodes have to be attached
  * to the Selector node using @ref rtSelectorSetChild. Empty slots will
  * cause a validation error.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   count      Number of child nodes to be attached to \a selector
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorSetChildCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorValidate,
  * @ref rtSelectorGetChildCount,
  * @ref rtSelectorSetChild,
  * @ref rtSelectorGetChild,
  * @ref rtSelectorGetChildType
  *
  */
  RTresult RTAPI rtSelectorSetChildCount(RTselector selector, unsigned int count);

  /**
  * @brief Returns the number of child node slots of
  * a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorGetChildCount returns in \a count the number of child
  * node slots that have been previously reserved for the Selector node
  * \a selector by @ref rtSelectorSetChildCount. The value of \a count
  * does not reflect the actual number of child nodes that have so far
  * been attached to the Selector node using @ref rtSelectorSetChild.
  *
  * @param[in]   selector   Selector node handle
  * @param[out]  count      Number of child node slots reserved for \a selector
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetChildCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorSetChildCount,
  * @ref rtSelectorSetChild,
  * @ref rtSelectorGetChild,
  * @ref rtSelectorGetChildType
  *
  */
  RTresult RTAPI rtSelectorGetChildCount(RTselector selector, unsigned int* count);

  /**
  * @brief Attaches a child node to a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * Attaches a new child node \a child to the parent node
  * \a selector. \a index specifies the number of the slot where the child
  * node gets attached.  The index value must be lower than the number
  * previously set by @ref rtSelectorSetChildCount, thus it must be in
  * the range from \a 0 to @ref rtSelectorGetChildCount \a -1.  Legal child
  * node types are @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, and
  * @ref RTtransform.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   index      Index of the parent slot the node \a child gets attached to
  * @param[in]   child      Child node to be attached. Can be {@ref RTgroup, @ref RTselector,
  * @ref RTgeometrygroup, @ref RTtransform}
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorSetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorSetChildCount,
  * @ref rtSelectorGetChildCount,
  * @ref rtSelectorGetChild,
  * @ref rtSelectorGetChildType
  *
  */
  RTresult RTAPI rtSelectorSetChild(RTselector selector, unsigned int index, RTobject child);

  /**
  * @brief Returns a child node that is attached to a
  * Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorGetChild returns in \a child a handle of the child node
  * currently attached to \a selector at slot \a index. The index value
  * must be lower than the number previously set by
  * @ref rtSelectorSetChildCount, thus it must be in the range from \a 0
  * to @ref rtSelectorGetChildCount \a - 1. The returned pointer is of generic
  * type @ref RTobject and needs to be cast to the actual child type, which
  * can be @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, or
  * @ref RTtransform. The actual type of \a child can be queried using
  * @ref rtSelectorGetChildType;
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   index      Child node index
  * @param[out]  child      Child node handle. Can be {@ref RTgroup, @ref RTselector,
  * @ref RTgeometrygroup, @ref RTtransform}
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorSetChildCount,
  * @ref rtSelectorGetChildCount,
  * @ref rtSelectorSetChild,
  * @ref rtSelectorGetChildType
  *
  */
  RTresult RTAPI rtSelectorGetChild(RTselector selector, unsigned int index, RTobject* child);

  /**
  * @brief Returns type information about a Selector
  * child node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorGetChildType queries the type of the child node
  * attached to \a selector at slot \a index.
  * If no child is associated with the given index, \a *type is set to
  * @ref RT_OBJECTTYPE_UNKNOWN and @ref RT_ERROR_INVALID_VALUE is returned.
  * Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  * The returned type is one of:
  *
  *   @ref RT_OBJECTTYPE_GROUP
  *   @ref RT_OBJECTTYPE_GEOMETRY_GROUP
  *   @ref RT_OBJECTTYPE_TRANSFORM
  *   @ref RT_OBJECTTYPE_SELECTOR
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   index      Child node index
  * @param[out]  type       Type of the child node
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetChildType was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorSetChildCount,
  * @ref rtSelectorGetChildCount,
  * @ref rtSelectorSetChild,
  * @ref rtSelectorGetChild
  *
  */
  RTresult RTAPI rtSelectorGetChildType(RTselector selector, unsigned int index, RTobjecttype* type);

  /**
  * @brief Declares a variable associated with a
  * Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * Declares a new variable identified by \a name, and associates it with
  * the Selector node \a selector. The new variable handle is returned in
  * \a v. After declaration, a variable does not have a type until its
  * value is set by an \a rtVariableSet{...} function. Once a variable
  * type has been set, it cannot be changed, i.e., only
  * \a rtVariableSet{...} functions of the same type can be used to
  * change the value of the variable.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   name       Variable identifier
  * @param[out]  v          New variable handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_REDECLARED
  * - @ref RT_ERROR_ILLEGAL_SYMBOL
  *
  * <B>History</B>
  *
  * @ref rtSelectorDeclareVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorQueryVariable,
  * @ref rtSelectorRemoveVariable,
  * @ref rtSelectorGetVariableCount,
  * @ref rtSelectorGetVariable,
  * @ref rtVariableSet{...}
  *
  */
  RTresult RTAPI rtSelectorDeclareVariable(RTselector selector, const char* name, RTvariable* v);

  /**
  * @brief Returns a variable associated with a
  * Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * Returns in \a v a handle to the variable identified by \a name, which
  * is associated with the Selector node \a selector. The current value of
  * a variable can be retrieved from its handle by using an appropriate
  * \a rtVariableGet{...} function matching the variable's type.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   name       Variable identifier
  * @param[out]  v          Variable handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorQueryVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorDeclareVariable,
  * @ref rtSelectorRemoveVariable,
  * @ref rtSelectorGetVariableCount,
  * @ref rtSelectorGetVariable,
  * \a rtVariableGet{...}
  *
  */
  RTresult RTAPI rtSelectorQueryVariable(RTselector selector, const char* name, RTvariable* v);

  /**
  * @brief Removes a variable from a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorRemoveVariable removes the variable \a v from the
  * Selector node \a selector and deletes it. The handle \a v must be
  * considered invalid afterwards.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   v          Variable handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtSelectorRemoveVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorDeclareVariable,
  * @ref rtSelectorQueryVariable,
  * @ref rtSelectorGetVariableCount,
  * @ref rtSelectorGetVariable
  *
  */
  RTresult RTAPI rtSelectorRemoveVariable(RTselector selector, RTvariable v);

  /**
  * @brief Returns the number of variables
  * attached to a Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * @ref rtSelectorGetVariableCount returns in \a count the number of
  * variables that are currently attached to the Selector node
  * \a selector.
  *
  * @param[in]   selector   Selector node handle
  * @param[out]  count      Number of variables associated with \a selector
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetVariableCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorDeclareVariable,
  * @ref rtSelectorQueryVariable,
  * @ref rtSelectorRemoveVariable,
  * @ref rtSelectorGetVariable
  *
  */
  RTresult RTAPI rtSelectorGetVariableCount(RTselector selector, unsigned int* count);

  /**
  * @brief Returns a variable associated with a
  * Selector node
  *
  * @ingroup SelectorNode
  *
  * <B>Description</B>
  *
  * Returns in \a v a handle to the variable located at position \a index
  * in the Selectors's variable array. \a index is a sequential number
  * depending on the order of variable declarations. The index must be
  * in the range from \a 0 to @ref rtSelectorGetVariableCount \a - 1.  The
  * current value of a variable can be retrieved from its handle by using
  * an appropriate \a rtVariableGet{...} function matching the
  * variable's type.
  *
  * @param[in]   selector   Selector node handle
  * @param[in]   index      Variable index
  * @param[out]  v          Variable handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtSelectorGetVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtSelectorDeclareVariable,
  * @ref rtSelectorQueryVariable,
  * @ref rtSelectorRemoveVariable,
  * @ref rtSelectorGetVariableCount,
  * \a rtVariableGet{...}
  *
  */
  RTresult RTAPI rtSelectorGetVariable(RTselector selector, unsigned int index, RTvariable* v);

/************************************
 **
 **    Transform object
 **
 ***********************************/

  /**
  * @brief Creates a new Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * Creates a new Transform node within the given context. For the node to be functional, a child
  * node must be attached using @ref rtTransformSetChild.  A transformation matrix can be associated
  * with the transform node with @ref rtTransformSetMatrix. Sets \a *transform to the handle of a
  * newly created transform within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a transform
  * is \a NULL.
  *
  * @param[in]   context    Specifies the rendering context of the Transform node
  * @param[out]  transform  New Transform node handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformDestroy,
  * @ref rtTransformValidate,
  * @ref rtTransformGetContext,
  * @ref rtTransformSetMatrix,
  * @ref rtTransformGetMatrix,
  * @ref rtTransformSetChild,
  * @ref rtTransformGetChild,
  * @ref rtTransformGetChildType
  *
  */
  RTresult RTAPI rtTransformCreate(RTcontext context, RTtransform* transform);

  /**
  * @brief Destroys a transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformDestroy removes \a transform from its context and deletes it.
  * \a transform should be a value returned by @ref rtTransformCreate.
  * No child graph nodes are destroyed.
  * After the call, \a transform is no longer a valid handle.
  *
  * @param[in]   transform   Handle of the transform node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformCreate,
  * @ref rtTransformValidate,
  * @ref rtTransformGetContext
  *
  */
  RTresult RTAPI rtTransformDestroy(RTtransform transform);

  /**
  * @brief Checks a Transform node for internal
  * consistency
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformValidate recursively checks consistency of the
  * Transform node \a transform and its child, i.e., it tries to validate
  * the whole model sub-tree with \a transform as root. For a Transform
  * node to be valid, it must have a child node attached. It is, however,
  * not required to explicitly set a transformation matrix. Without a specified
  * transformation matrix, the identity matrix is applied.
  *
  * @param[in]   transform   Transform root node of a model sub-tree to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformCreate,
  * @ref rtTransformDestroy,
  * @ref rtTransformGetContext,
  * @ref rtTransformSetMatrix,
  * @ref rtTransformSetChild
  *
  */
  RTresult RTAPI rtTransformValidate(RTtransform transform);

  /**
  * @brief Returns the context of a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformGetContext queries a transform node for its associated context.  \a transform
  * specifies the transform node to query, and should be a value returned by @ref
  * rtTransformCreate. Sets \a *context to the context associated with \a transform.
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  context     The context associated with \a transform
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformCreate,
  * @ref rtTransformDestroy,
  * @ref rtTransformValidate
  *
  */
  RTresult RTAPI rtTransformGetContext(RTtransform transform, RTcontext* context);

  /**
  * @brief Associates an affine transformation matrix
  * with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformSetMatrix associates a 4x4 matrix with the Transform
  * node \a transform. The provided transformation matrix results in a
  * corresponding affine transformation of all geometry contained in the
  * sub-tree with \a transform as root. At least one of the pointers
  * \a matrix and \a inverseMatrix must be non-\a NULL. If exactly one
  * pointer is valid, the other matrix will be computed. If both are
  * valid, the matrices will be used as-is. If \a transpose is \a 0,
  * source matrices are expected to be in row-major format, i.e., matrix
  * rows are contiguously laid out in memory:
  *
  *   float matrix[4*4] = { a11,  a12,  a13,  a14,
  *                         a21,  a22,  a23,  a24,
  *                         a31,  a32,  a33,  a34,
  *                         a41,  a42,  a43,  a44 };
  *
  * Here, the translational elements \a a14, \a a24, and \a a34 are at the
  * 4th, 8th, and 12th position the matrix array.  If the supplied
  * matrices are in column-major format, a non-0 \a transpose flag
  * can be used to trigger an automatic transpose of the input matrices.
  *
  * Calling this function clears any motion keys previously set for the Transform.
  *
  * @param[in]   transform        Transform node handle
  * @param[in]   transpose        Flag indicating whether \a matrix and \a inverseMatrix should be
  * transposed
  * @param[in]   matrix           Affine matrix (4x4 float array)
  * @param[in]   inverseMatrix    Inverted form of \a matrix
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformSetMatrix was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformGetMatrix
  *
  */
  RTresult RTAPI rtTransformSetMatrix(RTtransform transform, int transpose, const float* matrix, const float* inverseMatrix);

  /**
  * @brief Returns the affine matrix and its inverse associated with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformGetMatrix returns in \a matrix the affine matrix that
  * is currently used to perform a transformation of the geometry
  * contained in the sub-tree with \a transform as root. The corresponding
  * inverse matrix will be returned in \a inverseMatrix. One or both
  * pointers are allowed to be \a NULL. If \a transpose is \a 0, matrices
  * are returned in row-major format, i.e., matrix rows are contiguously
  * laid out in memory. If \a transpose is non-zero, matrices are returned
  * in column-major format. If non-\a NULL, matrix pointers must point to a
  * float array of at least 16 elements.
  *
  * @param[in]   transform        Transform node handle
  * @param[in]   transpose        Flag indicating whether \a matrix and \a inverseMatrix should be
  * transposed
  * @param[out]  matrix           Affine matrix (4x4 float array)
  * @param[out]  inverseMatrix    Inverted form of \a matrix
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformGetMatrix was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMatrix
  *
  */
  RTresult RTAPI rtTransformGetMatrix(RTtransform transform, int transpose, float* matrix, float* inverseMatrix);

  /**
  * @brief Sets the motion time range for a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * Sets the inclusive motion time range [timeBegin, timeEnd] for \a transform, where timeBegin <= timeEnd.
  * The default time range is [0.0, 1.0].  Has no effect unless @ref rtTransformSetMotionKeys
  * is also called, in which case the left endpoint of the time range, \a timeBegin, is associated with
  * the first motion key, and the right endpoint, \a timeEnd, with the last motion key.  The keys uniformly
  * divide the time range.
  *
  * @param[in]   transform   Transform node handle
  * @param[in]   timeBegin   Beginning time value of range
  * @param[in]   timeEnd     Ending time value of range
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformSetMotionRange was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformGetMotionRange,
  * @ref rtTransformSetMotionBorderMode,
  * @ref rtTransformSetMotionKeys,
  *
  */
  RTresult RTAPI rtTransformSetMotionRange( RTtransform transform, float timeBegin, float timeEnd );

  /**
  * @brief Returns the motion time range associated with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformGetMotionRange returns the motion time range set for the Transform.
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  timeBegin   Beginning time value of range
  * @param[out]  timeEnd     Ending time value of range
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformGetMotionRange was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMotionRange,
  * @ref rtTransformGetMotionBorderMode,
  * @ref rtTransformGetMotionKeyCount,
  * @ref rtTransformGetMotionKeyType,
  * @ref rtTransformGetMotionKeys,
  *
  */
  RTresult RTAPI rtTransformGetMotionRange( RTtransform transform, float* timeBegin, float* timeEnd );

  /**
  * @brief Sets the motion border modes of a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformSetMotionBorderMode sets the behavior of \a transform
  * outside its motion time range. The \a beginMode and \a endMode arguments
  * correspond to timeBegin and timeEnd set with @ref rtTransformSetMotionRange.
  * The arguments are independent, and each has one of the following values:
  *
  * - @ref RT_MOTIONBORDERMODE_CLAMP :
  *   The transform and the scene under it still exist at times less than timeBegin
  *   or greater than timeEnd, with the transform clamped to its values at timeBegin
  *   or timeEnd, respectively.
  *
  * - @ref RT_MOTIONBORDERMODE_VANISH :
  *   The transform and the scene under it vanish for times less than timeBegin
  *   or greater than timeEnd.
  *
  * @param[in]   transform   Transform node handle
  * @param[in]   beginMode   Motion border mode at motion range begin
  * @param[in]   endMode     Motion border mode at motion range end
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformSetMotionBorderMode was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformGetMotionBorderMode,
  * @ref rtTransformSetMotionRange,
  * @ref rtTransformSetMotionKeys,
  *
  */
  RTresult RTAPI rtTransformSetMotionBorderMode( RTtransform transform, RTmotionbordermode beginMode, RTmotionbordermode endMode );

  /**
  * @brief Returns the motion border modes of a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformGetMotionBorderMode returns the motion border modes
  * for the time range associated with \a transform.
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  beginMode   Motion border mode at motion time range begin
  * @param[out]  endMode     Motion border mode at motion time range end
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformGetMotionBorderMode was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMotionBorderMode,
  * @ref rtTransformGetMotionRange,
  * @ref rtTransformGetMotionKeyCount,
  * @ref rtTransformGetMotionKeyType,
  * @ref rtTransformGetMotionKeys,
  *
  */
  RTresult RTAPI rtTransformGetMotionBorderMode( RTtransform transform, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );

  /**
  * @brief Sets the motion keys associated with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformSetMotionKeys sets a series of key values defining how
  * \a transform varies with time.  The float values in \a keys are one of the
  * following types:
  *
  * - @ref RT_MOTIONKEYTYPE_MATRIX_FLOAT12
  *   Each key is a 12-float 3x4 matrix in row major order (3 rows, 4 columns).
  *   The length of \a keys is 12*n.
  *
  * - @ref RT_MOTIONKEYTYPE_SRT_FLOAT16
  *   Each key is a packed 16-float array in this order:
  *     [sx, a, b, pvx, sy, c, pvy, sz, pvz, qx, qy, qz, qw, tx, ty, tz]
  *   The length of \a keys is 16*n.
  *
  *   These are packed components of a scale/shear S, a quaternion R, and a translation T.
  *
  *   S = [ sx  a  b  pvx ]
  *       [  * sy  c  pvy ]
  *       [  *  * sz  pvz ]
  *
  *   R = [ qx, qy, qz, qw ]
  *     where qw = cos(theta/2) and [qx, qy, qz] = sin(theta/2)*normalized_axis.
  *
  *   T = [ tx, ty, tz ]
  *
  * Removing motion keys:
  *
  * Passing a single key with \a n == 1, or calling @ref rtTransformSetMatrix, removes any
  * motion data from \a transform, and sets its matrix to values derived from the single key.
  *
  * @param[in]   transform   Transform node handle
  * @param[in]   n           Number of motion keys >= 1
  * @param[in]   type        Type of motion keys
  * @param[in]   keys        \a n Motion keys associated with this Transform
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformSetMotionKeys was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformGetMotionKeyCount,
  * @ref rtTransformGetMotionKeyType,
  * @ref rtTransformGetMotionKeys,
  * @ref rtTransformSetMotionBorderMode,
  * @ref rtTransformSetMotionRange,
  *
  */
  RTresult RTAPI rtTransformSetMotionKeys( RTtransform transform, unsigned int n, RTmotionkeytype type, const float* keys );

  /**
  * @brief Returns the motion key type associated with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformGetMotionKeyType returns the key type from the most recent
  * call to @ref rtTransformSetMotionKeys, or @ref RT_MOTIONKEYTYPE_NONE if no
  * keys have been set.
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  type        Motion key type associated with this Transform
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformGetMotionKeyType was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMotionKeys,
  * @ref rtTransformGetMotionBorderMode,
  * @ref rtTransformGetMotionRange,
  * @ref rtTransformGetMotionKeyCount,
  * @ref rtTransformGetMotionKeys
  *
  */
  RTresult RTAPI rtTransformGetMotionKeyType( RTtransform transform, RTmotionkeytype* type );

  /**
  * @brief Returns the number of motion keys associated with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformGetMotionKeyCount returns in \a n the number of motion keys associated
  * with \a transform using @ref rtTransformSetMotionKeys.  Note that the default value
  * is 1, not 0, for a transform without motion.
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  n           Number of motion steps n >= 1
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformGetMotionKeyCount was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMotionKeys,
  * @ref rtTransformGetMotionBorderMode,
  * @ref rtTransformGetMotionRange,
  * @ref rtTransformGetMotionKeyType
  * @ref rtTransformGetMotionKeys
  *
  */
  RTresult RTAPI rtTransformGetMotionKeyCount( RTtransform transform, unsigned int* n );

  /**
  * @brief Returns the motion keys associated with a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  * @ref rtTransformGetMotionKeys returns in \a keys packed float values for
  * all motion keys.  The \a keys array must be large enough to hold all the keys,
  * based on the key type returned by @ref rtTransformGetMotionKeyType and the
  * number of keys returned by @ref rtTransformGetMotionKeyCount.  A single key
  * consists of either 12 floats (type RT_MOTIONKEYTYPE_MATRIX_FLOAT12) or
  * 16 floats (type RT_MOTIONKEYTYPE_SRT_FLOAT16).
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  keys        Motion keys associated with this Transform
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTransformGetMotionKeys was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMotionKeys,
  * @ref rtTransformGetMotionBorderMode,
  * @ref rtTransformGetMotionRange,
  * @ref rtTransformGetMotionKeyCount,
  * @ref rtTransformGetMotionKeyType
  *
  */
  RTresult RTAPI rtTransformGetMotionKeys( RTtransform transform, float* keys );

  /**
  * @brief Attaches a child node to a Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * Attaches a child node \a child to the parent node \a transform. Legal
  * child node types are @ref RTgroup, @ref RTselector, @ref RTgeometrygroup,
  * and @ref RTtransform. A transform node must have exactly one child.  If
  * a transformation matrix has been attached to \a transform with
  * @ref rtTransformSetMatrix, it is effective on the model sub-tree with
  * \a child as root node.
  *
  * @param[in]   transform   Transform node handle
  * @param[in]   child       Child node to be attached. Can be {@ref RTgroup, @ref RTselector,
  * @ref RTgeometrygroup, @ref RTtransform}
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformSetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetMatrix,
  * @ref rtTransformGetChild,
  * @ref rtTransformGetChildType
  *
  */
  RTresult RTAPI rtTransformSetChild(RTtransform transform, RTobject child);

  /**
  * @brief Returns the child node that is attached to a
  * Transform node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformGetChild returns in \a child a handle of the child
  * node currently attached to \a transform. The returned pointer is of
  * generic type @ref RTobject and needs to be cast to the actual child
  * type, which can be @ref RTgroup, @ref RTselector, @ref RTgeometrygroup, or
  * @ref RTtransform. The actual type of \a child can be queried using
  * @ref rtTransformGetChildType.
  * Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  child       Child node handle. Can be {@ref RTgroup, @ref RTselector,
  * @ref RTgeometrygroup, @ref RTtransform}
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformGetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetChild,
  * @ref rtTransformGetChildType
  *
  */
  RTresult RTAPI rtTransformGetChild(RTtransform transform, RTobject* child);

  /**
  * @brief Returns type information about a
  * Transform child node
  *
  * @ingroup TransformNode
  *
  * <B>Description</B>
  *
  * @ref rtTransformGetChildType queries the type of the child node
  * attached to \a transform. If no child is attached, \a *type is set to
  * @ref RT_OBJECTTYPE_UNKNOWN and @ref RT_ERROR_INVALID_VALUE is returned.
  * Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  * The returned type is one of:
  *
  *  - @ref RT_OBJECTTYPE_GROUP
  *  - @ref RT_OBJECTTYPE_GEOMETRY_GROUP
  *  - @ref RT_OBJECTTYPE_TRANSFORM
  *  - @ref RT_OBJECTTYPE_SELECTOR
  *
  * @param[in]   transform   Transform node handle
  * @param[out]  type        Type of the child node
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTransformGetChildType was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTransformSetChild,
  * @ref rtTransformGetChild
  *
  */
  RTresult RTAPI rtTransformGetChildType(RTtransform transform, RTobjecttype* type);

/************************************
 **
 **    GeometryGroup object
 **
 ***********************************/

  /**
  * @brief Creates a new geometry group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupCreate creates a new geometry group within a context. \a context
  * specifies the target context, and should be a value returned by @ref rtContextCreate.
  * Sets \a *geometrygroup to the handle of a newly created geometry group within \a context.
  * Returns @ref RT_ERROR_INVALID_VALUE if \a geometrygroup is \a NULL.
  *
  * @param[in]   context         Specifies a context within which to create a new geometry group
  * @param[out]  geometrygroup   Returns a newly created geometry group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupDestroy,
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtGeometryGroupCreate(RTcontext context, RTgeometrygroup* geometrygroup);

  /**
  * @brief Destroys a geometry group node
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupDestroy removes \a geometrygroup from its context and deletes it.
  * \a geometrygroup should be a value returned by @ref rtGeometryGroupCreate.
  * No child graph nodes are destroyed.
  * After the call, \a geometrygroup is no longer a valid handle.
  *
  * @param[in]   geometrygroup   Handle of the geometry group node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupCreate
  *
  */
  RTresult RTAPI rtGeometryGroupDestroy(RTgeometrygroup geometrygroup);

  /**
  * @brief Validates the state of the geometry group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupValidate checks \a geometrygroup for completeness. If \a geometrygroup or
  * any of the objects attached to \a geometrygroup are not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   geometrygroup   Specifies the geometry group to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupCreate
  *
  */
  RTresult RTAPI rtGeometryGroupValidate(RTgeometrygroup geometrygroup);

  /**
  * @brief Returns the context associated with a geometry group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupGetContext queries a geometry group for its associated context.
  * \a geometrygroup specifies the geometry group to query, and must be a value returned by
  * @ref rtGeometryGroupCreate. Sets \a *context to the context
  * associated with \a geometrygroup.
  *
  * @param[in]   geometrygroup   Specifies the geometry group to query
  * @param[out]  context         Returns the context associated with the geometry group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextCreate,
  * @ref rtGeometryGroupCreate
  *
  */
  RTresult RTAPI rtGeometryGroupGetContext(RTgeometrygroup geometrygroup, RTcontext* context);

  /**
  * @brief Set the acceleration structure for a group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupSetAcceleration attaches an acceleration structure to a geometry group. The
  * acceleration structure must have been previously created using @ref rtAccelerationCreate. Every
  * geometry group is required to have an acceleration structure assigned in order to pass
  * validation. The acceleration structure will be built over the primitives contained in all
  * children of the geometry group. This enables a single acceleration structure to be built over
  * primitives of multiple geometry instances.  Note that it is legal to attach a single
  * RTacceleration object to multiple geometry groups, as long as the underlying geometry of all
  * children is the same. This corresponds to attaching an acceleration structure to multiple groups
  * at higher graph levels using @ref rtGroupSetAcceleration.
  *
  * @param[in]   geometrygroup   The geometry group handle
  * @param[in]   acceleration    The acceleration structure to attach to the geometry group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupSetAcceleration was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupGetAcceleration,
  * @ref rtAccelerationCreate,
  * @ref rtGroupSetAcceleration
  *
  */
  RTresult RTAPI rtGeometryGroupSetAcceleration(RTgeometrygroup geometrygroup, RTacceleration acceleration);

  /**
  * @brief Returns the acceleration structure attached to a geometry group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupGetAcceleration returns the acceleration structure attached to a geometry
  * group using @ref rtGeometryGroupSetAcceleration.  If no acceleration structure has previously
  * been set, \a *acceleration is set to \a NULL.
  *
  * @param[in]   geometrygroup   The geometry group handle
  * @param[out]  acceleration    The returned acceleration structure object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupGetAcceleration was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupSetAcceleration,
  * @ref rtAccelerationCreate
  *
  */
  RTresult RTAPI rtGeometryGroupGetAcceleration(RTgeometrygroup geometrygroup, RTacceleration* acceleration);

  /**
   * @brief Sets instance flags for a geometry group.
   *
   * @ingroup GeometryGroup
   *
   * <B>Description</B>
   *
   * This function controls the @ref RTinstanceflags of the given geometry group.
   * Note that flags are only considered when tracing against an RTgroup with this GeometryGroup
   * as a child (potentially with Transforms).
   * Tracing directly against the GeometryGroup will ignore the flags.
   * The flags override the @ref RTgeometryflags of the underlying geometry where appropriate.
   *
   * @param[in] group   The group handle
   * @param[in] flags   Instance flags for the given geometry group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometryGroupSetFlags was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryTrianglesSetFlagsPerMaterial,
   * @ref rtGeometrySetFlags,
   * @ref rtGeometryGroupGetFlags,
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometryGroupSetFlags( RTgeometrygroup group, RTinstanceflags flags );

  /**
   * @brief Gets instance flags of a geometry group.
   *
   * @ingroup GeometryGroup
   *
   * <B>Description</B>
   *
   * See @ref rtGeometryGroupSetFlags for details.
   *
   * @param[in] group   The group handle
   * @param[out] flags  Instance flags for the given geometry group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometryGroupGetFlags was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryGroupSetFlags,
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometryGroupGetFlags( RTgeometrygroup group, RTinstanceflags* flags );

  /**
   * @brief Sets the visibility mask of a geometry group.
   *
   * @ingroup GeometryGroup
   *
   * <B>Description</B>
   * Geometry is intersected by rays if the ray's @ref RTvisibilitymask shares at
   * least one bit with the group's mask. This mechanism allows for a number of
   * user-defined visibility groups that can be excluded from certain types of rays
   * as needed.
   * Note that the visibility mask is not checked for the root node of a trace call.
   * (It is assumed to be visible otherwise trace should not be called).
   * Note that the @pre mask is currently limited to 8 bits.
   *
   * @param[in] group   The group handle
   * @param[in] mask    A set of bits for which rays will intersect the group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometryGroupSetVisibilityMask was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGroupSetVisibilityMask
   * @ref rtGeometryGroupGetVisibilityMask,
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometryGroupSetVisibilityMask( RTgeometrygroup group, RTvisibilitymask mask );

  /**
   * @brief Gets the visibility mask of a geometry group.
   *
   * @ingroup GeometryGroup
   *
   * <B>Description</B>
   * See @ref rtGeometryGroupSetVisibilityMask for details/
   *
   * @param[in] group   The group handle
   * @param[out] mask   A set of bits for which rays will intersect the group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometryGroupGetVisibilityMask was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGroupGetVisibilityMask
   * @ref rtGeometryGroupSetVisibilityMask,
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometryGroupGetVisibilityMask( RTgeometrygroup group, RTvisibilitymask* mask );

  /**
  * @brief Sets the number of child nodes to be attached to the group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupSetChildCount specifies the number of child slots in this geometry
  * group. Potentially existing links to children at indices greater than \a count-1 are removed. If
  * the call increases the number of slots, the newly created slots are empty and need to be filled
  * using @ref rtGeometryGroupSetChild before validation.
  *
  * @param[in]   geometrygroup   The parent geometry group handle
  * @param[in]   count           Number of child slots to allocate for the geometry group
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupSetChildCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupGetChild,
  * @ref rtGeometryGroupGetChildCount
  * @ref rtGeometryGroupSetChild
  *
  */
  RTresult RTAPI rtGeometryGroupSetChildCount(RTgeometrygroup geometrygroup, unsigned int count);

  /**
  * @brief Returns the number of child slots for a group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupGetChildCount returns the number of child slots allocated using @ref
  * rtGeometryGroupSetChildCount.  This includes empty slots which may not yet have actual children
  * assigned by @ref rtGeometryGroupSetChild.
  *
  * @param[in]   geometrygroup   The parent geometry group handle
  * @param[out]  count           Returned number of child slots
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupGetChildCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupSetChild,
  * @ref rtGeometryGroupGetChild,
  * @ref rtGeometryGroupSetChildCount
  *
  */
  RTresult RTAPI rtGeometryGroupGetChildCount(RTgeometrygroup geometrygroup, unsigned int* count);

  /**
  * @brief Attaches a child node to a geometry group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupSetChild attaches a new child node \a geometryinstance to the parent node
  * \a geometrygroup. \a index specifies the number of the slot where the child
  * node gets attached.  The index value must be lower than the number
  * previously set by @ref rtGeometryGroupSetChildCount.
  *
  * @param[in]   geometrygroup      The parent geometry group handle
  * @param[in]   index              The index in the parent's child slot array
  * @param[in]   geometryinstance   The child node to be attached
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupSetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupSetChildCount,
  * @ref rtGeometryGroupGetChildCount,
  * @ref rtGeometryGroupGetChild
  *
  */
  RTresult RTAPI rtGeometryGroupSetChild(RTgeometrygroup geometrygroup, unsigned int index, RTgeometryinstance geometryinstance);

  /**
  * @brief Returns a child node of a geometry group
  *
  * @ingroup GeometryGroup
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGroupGetChild returns the child geometry instance at slot \a index of the parent
  * \a geometrygroup.  If no child has been assigned to the given slot, \a *geometryinstance is set
  * to \a NULL.  Returns @ref RT_ERROR_INVALID_VALUE if given an invalid child index or \a NULL
  * pointer.
  *
  * @param[in]   geometrygroup      The parent geometry group handle
  * @param[in]   index              The index of the child slot to query
  * @param[out]  geometryinstance   The returned child geometry instance
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGroupGetChild was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGroupSetChild,
  * @ref rtGeometryGroupSetChildCount,
  * @ref rtGeometryGroupGetChildCount,
  *
  */
  RTresult RTAPI rtGeometryGroupGetChild(RTgeometrygroup geometrygroup, unsigned int index, RTgeometryinstance* geometryinstance);

/************************************
 **
 **    Acceleration object
 **
 ***********************************/

  /**
  * @brief Creates a new acceleration structure
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationCreate creates a new ray tracing acceleration structure within a context.  An
  * acceleration structure is used by attaching it to a group or geometry group by calling @ref
  * rtGroupSetAcceleration or @ref rtGeometryGroupSetAcceleration. Note that an acceleration
  * structure can be shared by attaching it to multiple groups or geometry groups if the underlying
  * geometric structures are the same, see @ref rtGroupSetAcceleration and @ref
  * rtGeometryGroupSetAcceleration for more details. A newly created acceleration structure is
  * initially in dirty state.  Sets \a *acceleration to the handle of a newly created acceleration
  * structure within \a context.  Returns @ref RT_ERROR_INVALID_VALUE if \a acceleration is \a NULL.
  *
  * @param[in]   context        Specifies a context within which to create a new acceleration structure
  * @param[out]  acceleration   Returns the newly created acceleration structure
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationDestroy,
  * @ref rtContextCreate,
  * @ref rtAccelerationMarkDirty,
  * @ref rtAccelerationIsDirty,
  * @ref rtGroupSetAcceleration,
  * @ref rtGeometryGroupSetAcceleration
  *
  */
  RTresult RTAPI rtAccelerationCreate(RTcontext context, RTacceleration* acceleration);

  /**
  * @brief Destroys an acceleration structure object
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationDestroy removes \a acceleration from its context and deletes it.
  * \a acceleration should be a value returned by @ref rtAccelerationCreate.
  * After the call, \a acceleration is no longer a valid handle.
  *
  * @param[in]   acceleration   Handle of the acceleration structure to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationCreate
  *
  */
  RTresult RTAPI rtAccelerationDestroy(RTacceleration acceleration);

  /**
  * @brief Validates the state of an acceleration structure
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationValidate checks \a acceleration for completeness. If \a acceleration is
  * not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   acceleration   The acceleration structure handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationCreate
  *
  */
  RTresult RTAPI rtAccelerationValidate(RTacceleration acceleration);

  /**
  * @brief Returns the context associated with an acceleration structure
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationGetContext queries an acceleration structure for its associated context.
  * The context handle is returned in \a *context.
  *
  * @param[in]   acceleration   The acceleration structure handle
  * @param[out]  context        Returns the context associated with the acceleration structure
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationCreate
  *
  */
  RTresult RTAPI rtAccelerationGetContext(RTacceleration acceleration, RTcontext* context);


  /**
  * @brief Specifies the builder to be used for an acceleration structure
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationSetBuilder specifies the method used to construct the ray tracing
  * acceleration structure represented by \a acceleration. A builder must be set for the
  * acceleration structure to pass validation.  The current builder can be changed at any time,
  * including after a call to @ref rtContextLaunch "rtContextLaunch".  In this case, data previously
  * computed for the acceleration structure is invalidated and the acceleration will be marked
  * dirty.
  *
  * \a builder can take one of the following values:
  *
  * - "NoAccel": Specifies that no acceleration structure is explicitly built. Traversal linearly loops through the
  * list of primitives to intersect. This can be useful e.g. for higher level groups with only few children, where managing a more complex structure introduces unnecessary overhead.
  *
  * - "Bvh": A standard bounding volume hierarchy, useful for most types of graph levels and geometry. Medium build speed, good ray tracing performance.
  *
  * - "Sbvh": A high quality BVH variant for maximum ray tracing performance. Slower build speed and slightly higher memory footprint than "Bvh".
  *
  * - "Trbvh": High quality similar to Sbvh but with fast build performance. The Trbvh builder uses about 2.5 times the size of the final BVH for scratch space. A CPU-based Trbvh builder that does not have the memory constraints is available. OptiX includes an optional automatic fallback to the CPU version when out of GPU memory. Please refer to the Programming Guide for more details.  Supports motion blur.
  *
  * - "MedianBvh": Deprecated in OptiX 4.0. This builder is now internally remapped to Trbvh.
  *
  * - "Lbvh": Deprecated in OptiX 4.0. This builder is now internally remapped to Trbvh.
  *
  * - "TriangleKdTree": Deprecated in OptiX 4.0. This builder is now internally remapped to Trbvh.
  *
  * @param[in]   acceleration   The acceleration structure handle
  * @param[in]   builder        String value specifying the builder type
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationSetBuilder was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationGetBuilder,
  * @ref rtAccelerationSetProperty
  *
  */
  RTresult RTAPI rtAccelerationSetBuilder(RTacceleration acceleration, const char* builder);

  /**
  * @brief Query the current builder from an acceleration structure
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationGetBuilder returns the name of the builder currently
  * used in the acceleration structure \a acceleration. If no builder has
  * been set for \a acceleration, an empty string is returned.
  * \a stringReturn will be set to point to the returned string. The
  * memory \a stringReturn points to will be valid until the next API
  * call that returns a string.
  *
  * @param[in]   acceleration    The acceleration structure handle
  * @param[out]  stringReturn    Return string buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationGetBuilder was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationSetBuilder
  *
  */
  RTresult RTAPI rtAccelerationGetBuilder(RTacceleration acceleration, const char** stringReturn);

  /**
  * Deprecated in OptiX 4.0. Setting a traverser is no longer necessary and will be ignored.
  *
  */
  RTresult RTAPI rtAccelerationSetTraverser(RTacceleration acceleration, const char* traverser);

  /**
  * Deprecated in OptiX 4.0.
  *
  */
  RTresult RTAPI rtAccelerationGetTraverser(RTacceleration acceleration, const char** stringReturn);

  /**
  * @brief Sets an acceleration structure property
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationSetProperty sets a named property value for an
  * acceleration structure. Properties can be used to fine tune the way an
  * acceleration structure is built, in order to achieve faster build
  * times or better ray tracing performance.  Properties are evaluated and
  * applied by the acceleration structure during build time, and
  * different builders recognize different properties. Setting a property
  * will never fail as long as \a acceleration is a valid
  * handle. Properties that are not recognized by an acceleration
  * structure will be ignored.
  *
  * The following is a list of the properties used by the individual builders:
  *
  * - "refit":
  * Available in: Trbvh, Bvh
  * If set to "1", the builder will only readjust the node bounds of the bounding
  * volume hierarchy instead of constructing it from scratch. Refit is only
  * effective if there is an initial BVH already in place, and the underlying
  * geometry has undergone relatively modest deformation.  In this case, the
  * builder delivers a very fast BVH update without sacrificing too much ray
  * tracing performance.  The default is "0".
  *
  * - "vertex_buffer_name":
  * Available in: Trbvh, Sbvh
  * The name of the buffer variable holding triangle vertex data.  Each vertex
  * consists of 3 floats.  The default is "vertex_buffer".
  *
  * - "vertex_buffer_stride":
  * Available in: Trbvh, Sbvh
  * The offset between two vertices in the vertex buffer, given in bytes.  The
  * default value is "0", which assumes the vertices are tightly packed.
  *
  * - "index_buffer_name":
  * Available in: Trbvh, Sbvh
  * The name of the buffer variable holding vertex index data. The entries in
  * this buffer are indices of type int, where each index refers to one entry in
  * the vertex buffer. A sequence of three indices represents one triangle. If no
  * index buffer is given, the vertices in the vertex buffer are assumed to be a
  * list of triangles, i.e. every 3 vertices in a row form a triangle.  The
  * default is "index_buffer".
  *
  * - "index_buffer_stride":
  * Available in: Trbvh, Sbvh
  * The offset between two indices in the index buffer, given in bytes.  The
  * default value is "0", which assumes the indices are tightly packed.
  *
  * - "chunk_size":
  * Available in: Trbvh
  * Number of bytes to be used for a partitioned acceleration structure build. If
  * no chunk size is set, or set to "0", the chunk size is chosen automatically.
  * If set to "-1", the chunk size is unlimited. The minimum chunk size is 64MB.
  * Please note that specifying a small chunk size reduces the peak-memory
  * footprint of the Trbvh but can result in slower rendering performance.
  *
  * - " motion_steps"
  * Available in: Trbvh
  * Number of motion steps to build into an acceleration structure that contains
  * motion geometry or motion transforms. Ignored for acceleration structures
  * built over static nodes. Gives a tradeoff between device memory
  * and time: if the input geometry or transforms have many motion steps,
  * then increasing the motion steps in the acceleration structure may result in
  * faster traversal, at the cost of linear increase in memory usage.
  * Default 2, and clamped >=1.
  *
  * @param[in]   acceleration   The acceleration structure handle
  * @param[in]   name           String value specifying the name of the property
  * @param[in]   value          String value specifying the value of the property
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationSetProperty was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationGetProperty,
  * @ref rtAccelerationSetBuilder,
  *
  */
  RTresult RTAPI rtAccelerationSetProperty(RTacceleration acceleration, const char* name, const char* value);

  /**
  * @brief Queries an acceleration structure property
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationGetProperty returns the value of the acceleration
  * structure property \a name.  See @ref rtAccelerationSetProperty for a
  * list of supported properties.  If the property name is not found, an
  * empty string is returned.  \a stringReturn will be set to point to
  * the returned string. The memory \a stringReturn points to will be
  * valid until the next API call that returns a string.
  *
  * @param[in]   acceleration    The acceleration structure handle
  * @param[in]   name            The name of the property to be queried
  * @param[out]  stringReturn    Return string buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationGetProperty was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationSetProperty,
  * @ref rtAccelerationSetBuilder,
  *
  */
  RTresult RTAPI rtAccelerationGetProperty(RTacceleration acceleration, const char* name, const char** stringReturn);

  /**
  * Deprecated in OptiX 4.0. Should not be called.
  *
  */
  RTresult RTAPI rtAccelerationGetDataSize(RTacceleration acceleration, RTsize* size);

  /**
  * Deprecated in OptiX 4.0. Should not be called.
  *
  */
  RTresult RTAPI rtAccelerationGetData(RTacceleration acceleration, void* data);

  /**
  * Deprecated in OptiX 4.0. Should not be called.
  *
  */
  RTresult RTAPI rtAccelerationSetData(RTacceleration acceleration, const void* data, RTsize size);

  /**
  * @brief Marks an acceleration structure as dirty
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationMarkDirty sets the dirty flag for \a acceleration.
  *
  * Any acceleration structure which is marked dirty will be rebuilt on a call to one of the @ref
  * rtContextLaunch "rtContextLaunch" functions, and its dirty flag will be reset.
  *
  * An acceleration structure which is not marked dirty will never be rebuilt, even if associated
  * groups, geometry, properties, or any other values have changed.
  *
  * Initially after creation, acceleration structures are marked dirty.
  *
  * @param[in]   acceleration   The acceleration structure handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationMarkDirty was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationIsDirty,
  * @ref rtContextLaunch
  *
  */
  RTresult RTAPI rtAccelerationMarkDirty(RTacceleration acceleration);

  /**
  * @brief Returns the dirty flag of an acceleration structure
  *
  * @ingroup AccelerationStructure
  *
  * <B>Description</B>
  *
  * @ref rtAccelerationIsDirty returns whether the acceleration structure is currently marked dirty.
  * If the flag is set, a nonzero value will be returned in \a *dirty. Otherwise, zero is returned.
  *
  * Any acceleration structure which is marked dirty will be rebuilt on a call to one of the @ref
  * rtContextLaunch "rtContextLaunch" functions, and its dirty flag will be reset.
  *
  * An acceleration structure which is not marked dirty will never be rebuilt, even if associated
  * groups, geometry, properties, or any other values have changed.
  *
  * Initially after creation, acceleration structures are marked dirty.
  *
  * @param[in]   acceleration   The acceleration structure handle
  * @param[out]  dirty          Returned dirty flag
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtAccelerationIsDirty was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtAccelerationMarkDirty,
  * @ref rtContextLaunch
  *
  */
  RTresult RTAPI rtAccelerationIsDirty(RTacceleration acceleration, int* dirty);

/************************************
 **
 **    GeometryInstance object
 **
 ***********************************/

  /**
  * @brief Creates a new geometry instance node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceCreate creates a new geometry instance node within a context. \a context
  * specifies the target context, and should be a value returned by @ref rtContextCreate.
  * Sets \a *geometryinstance to the handle of a newly created geometry instance within \a context.
  * Returns @ref RT_ERROR_INVALID_VALUE if \a geometryinstance is \a NULL.
  *
  * @param[in]   context            Specifies the rendering context of the GeometryInstance node
  * @param[out]  geometryinstance   New GeometryInstance node handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceDestroy,
  * @ref rtGeometryInstanceDestroy,
  * @ref rtGeometryInstanceGetContext
  *
  */
  RTresult RTAPI rtGeometryInstanceCreate(RTcontext context, RTgeometryinstance* geometryinstance);

  /**
  * @brief Destroys a geometry instance node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceDestroy removes \a geometryinstance from its context and deletes it.  \a
  * geometryinstance should be a value returned by @ref rtGeometryInstanceCreate.  Associated
  * variables declared via @ref rtGeometryInstanceDeclareVariable are destroyed, but no child graph
  * nodes are destroyed.  After the call, \a geometryinstance is no longer a valid handle.
  *
  * @param[in]   geometryinstance   Handle of the geometry instance node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceCreate
  *
  */
  RTresult RTAPI rtGeometryInstanceDestroy(RTgeometryinstance geometryinstance);

  /**
  * @brief Checks a GeometryInstance node for internal consistency
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceValidate checks \a geometryinstance for completeness. If \a geomertryinstance or
  * any of the objects attached to \a geometry are not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   geometryinstance   GeometryInstance node of a model sub-tree to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceCreate
  *
  */
  RTresult RTAPI rtGeometryInstanceValidate(RTgeometryinstance geometryinstance);

  /**
  * @brief Returns the context associated with a geometry instance node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetContext queries a geometry instance node for its associated context.
  * \a geometryinstance specifies the geometry node to query, and should be a value returned by
  * @ref rtGeometryInstanceCreate. Sets \a *context to the context
  * associated with \a geometryinstance.
  *
  * @param[in]   geometryinstance   Specifies the geometry instance
  * @param[out]  context            Handle for queried context
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetContext
  *
  */
  RTresult RTAPI rtGeometryInstanceGetContext(RTgeometryinstance geometryinstance, RTcontext* context);

  /**
  * @brief Attaches a Geometry node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceSetGeometry attaches a Geometry node to a GeometryInstance.
  * Only one GeometryTriangles or Geometry node can be attached to a GeometryInstance at a time.
  * However, it is possible at any time to attach a different GeometryTriangles or Geometry via
  * rtGeometryInstanceSetGeometryTriangles or rtGeometryInstanceSetGeometry respectively.
  *
  * @param[in]   geometryinstance   GeometryInstance node handle to attach \a geometry to
  * @param[in]   geometry           Geometry handle to attach to \a geometryinstance
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceSetGeometry was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetGeometry
  * @ref rtGeometryInstanceGetGeometryTriangles
  * @ref rtGeometryInstanceSetGeometryTriangles
  *
  */
  RTresult RTAPI rtGeometryInstanceSetGeometry(RTgeometryinstance geometryinstance, RTgeometry geometry);

  /**
  * @brief Returns the attached Geometry node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetGeometry sets \a geometry to the handle of the attached Geometry node.
  * Only one GeometryTriangles or Geometry node can be attached to a GeometryInstance at a time.
  *
  * @param[in]   geometryinstance   GeometryInstance node handle to query geometry
  * @param[out]  geometry           Handle to attached Geometry node
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetGeometry was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceCreate,
  * @ref rtGeometryInstanceDestroy,
  * @ref rtGeometryInstanceValidate,
  * @ref rtGeometryInstanceSetGeometry
  * @ref rtGeometryInstanceSetGeometryTriangles
  * @ref rtGeometryInstanceGetGeometryTriangles
  *
  */
  RTresult RTAPI rtGeometryInstanceGetGeometry(RTgeometryinstance geometryinstance, RTgeometry* geometry);

  /**
  * @brief Attaches a Geometry node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceSetGeometryTriangles attaches a GeometryTriangles node to a GeometryInstance.
  * Only one GeometryTriangles or Geometry node can be attached to a GeometryInstance at a time.
  * However, it is possible at any time to attach a different GeometryTriangles or Geometry via
  * rtGeometryInstanceSetGeometryTriangles or rtGeometryInstanceSetGeometry respectively.
  *
  * @param[in]   geometryinstance   GeometryInstance node handle to attach \a geometrytriangles to
  * @param[in]   geometrytriangles  GeometryTriangles handle to attach to \a geometryinstance
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceSetGeometryTriangles was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetGeometryTriangles
  * @ref rtGeometryInstanceSetGeometry
  * @ref rtGeometryInstanceGetGeometry
  *
  */
  RTresult RTAPI rtGeometryInstanceSetGeometryTriangles(RTgeometryinstance geometryinstance, RTgeometrytriangles geometrytriangles);

  /**
  * @brief Returns the attached Geometry node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetGeometryTriangles sets \a geometrytriangles to the handle of the attached GeometryTriangles node.
  * If no GeometryTriangles node is attached or a Geometry node is attached, @ref RT_ERROR_INVALID_VALUE is returned, else @ref RT_SUCCESS.
  *
  * @param[in]   geometryinstance   GeometryInstance node handle to query geometrytriangles
  * @param[out]  geometrytriangles  Handle to attached GeometryTriangles node
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetGeometryTriangles was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceCreate,
  * @ref rtGeometryInstanceDestroy,
  * @ref rtGeometryInstanceValidate,
  * @ref rtGeometryInstanceSetGeometryTriangles
  * @ref rtGeometryInstanceSetGeometry
  * @ref rtGeometryInstanceGetGeometry
  *
  */
  RTresult RTAPI rtGeometryInstanceGetGeometryTriangles(RTgeometryinstance geometryinstance, RTgeometrytriangles* geometrytriangles);

  /**
  * @brief Sets the number of materials
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceSetMaterialCount sets the number of materials \a count that will be
  * attached to \a geometryinstance. The number of attached materials can be changed at any
  * time.  Increasing the number of materials will not modify already assigned materials.
  * Decreasing the number of materials will not modify the remaining already assigned
  * materials.
  *
  * @param[in]   geometryinstance   GeometryInstance node to set number of materials
  * @param[in]   count              Number of materials to be set
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceSetMaterialCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetMaterialCount
  *
  */
  RTresult RTAPI rtGeometryInstanceSetMaterialCount(RTgeometryinstance geometryinstance, unsigned int count);

  /**
  * @brief Returns the number of attached materials
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetMaterialCount returns for \a geometryinstance the number of attached
  * Material nodes \a count. The number of materials can be set with @ref
  * rtGeometryInstanceSetMaterialCount.
  *
  * @param[in]   geometryinstance   GeometryInstance node to query from the number of materials
  * @param[out]  count              Number of attached materials
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetMaterialCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceSetMaterialCount
  *
  */
  RTresult RTAPI rtGeometryInstanceGetMaterialCount(RTgeometryinstance geometryinstance, unsigned int* count);

  /**
  * @brief Sets a material
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceSetMaterial attaches \a material to \a geometryinstance at position \a index
  * in its internal Material node list.  \a index must be in the range \a 0 to @ref
  * rtGeometryInstanceGetMaterialCount \a - 1.
  *
  * @param[in]   geometryinstance   GeometryInstance node for which to set a material
  * @param[in]   index              Index into the material list
  * @param[in]   material           Material handle to attach to \a geometryinstance
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceSetMaterial was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetMaterialCount,
  * @ref rtGeometryInstanceSetMaterialCount
  *
  */
  RTresult RTAPI rtGeometryInstanceSetMaterial(RTgeometryinstance geometryinstance, unsigned int index, RTmaterial material);

  /**
  * @brief Returns a material handle
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetMaterial returns handle \a material for the Material node at position
  * \a index in the material list of \a geometryinstance. Returns @ref RT_ERROR_INVALID_VALUE if \a
  * index is invalid.
  *
  * @param[in]   geometryinstance   GeometryInstance node handle to query material
  * @param[in]   index              Index of material
  * @param[out]  material           Handle to material
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetMaterial was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetMaterialCount,
  * @ref rtGeometryInstanceSetMaterial
  *
  */
  RTresult RTAPI rtGeometryInstanceGetMaterial(RTgeometryinstance geometryinstance, unsigned int index, RTmaterial* material);

  /**
  * @brief Declares a new named variable associated with a geometry node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceDeclareVariable declares a new variable associated with a geometry
  * instance node. \a geometryinstance specifies the target geometry node, and should be a value
  * returned by @ref rtGeometryInstanceCreate. \a name specifies the name of the variable, and
  * should be a \a NULL-terminated string. If there is currently no variable associated with \a
  * geometryinstance named \a name, a new variable named \a name will be created and associated with
  * \a geometryinstance.  After the call, \a *v will be set to the handle of the newly-created
  * variable.  Otherwise, \a *v will be set to \a NULL. After declaration, the variable can be
  * queried with @ref rtGeometryInstanceQueryVariable or @ref rtGeometryInstanceGetVariable. A
  * declared variable does not have a type until its value is set with one of the @ref rtVariableSet
  * functions. Once a variable is set, its type cannot be changed anymore.
  *
  * @param[in]   geometryinstance   Specifies the associated GeometryInstance node
  * @param[in]   name               The name that identifies the variable
  * @param[out]  v                  Returns a handle to a newly declared variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceDeclareVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref Variables,
  * @ref rtGeometryInstanceQueryVariable,
  * @ref rtGeometryInstanceGetVariable,
  * @ref rtGeometryInstanceRemoveVariable
  *
  */
  RTresult RTAPI rtGeometryInstanceDeclareVariable(RTgeometryinstance geometryinstance, const char* name, RTvariable* v);

  /**
  * @brief Returns a handle to a named variable of a geometry node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceQueryVariable queries the handle of a geometry instance node's named
  * variable.  \a geometryinstance specifies the target geometry instance node, as returned by
  * @ref rtGeometryInstanceCreate. \a name specifies the name of the variable, and should be a \a
  * \a NULL -terminated string. If \a name is the name of a variable attached to \a geometryinstance,
  * returns a handle to that variable in \a *v, otherwise \a NULL.  Geometry instance variables have
  * to be declared with @ref rtGeometryInstanceDeclareVariable before they can be queried.
  *
  * @param[in]   geometryinstance   The GeometryInstance node to query from a variable
  * @param[in]   name               The name that identifies the variable to be queried
  * @param[out]  v                  Returns the named variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceQueryVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceDeclareVariable,
  * @ref rtGeometryInstanceRemoveVariable,
  * @ref rtGeometryInstanceGetVariableCount,
  * @ref rtGeometryInstanceGetVariable
  *
  */
  RTresult RTAPI rtGeometryInstanceQueryVariable(RTgeometryinstance geometryinstance, const char* name, RTvariable* v);

  /**
  * @brief Removes a named variable from a geometry instance node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceRemoveVariable removes a named variable from a geometry instance. The
  * target geometry instance is specified by \a geometryinstance, which should be a value returned
  * by @ref rtGeometryInstanceCreate. The variable to be removed is specified by \a v, which should
  * be a value returned by @ref rtGeometryInstanceDeclareVariable. Once a variable has been removed
  * from this geometry instance, another variable with the same name as the removed variable may be
  * declared.
  *
  * @param[in]   geometryinstance   The GeometryInstance node from which to remove a variable
  * @param[in]   v                  The variable to be removed
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceRemoveVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextRemoveVariable,
  * @ref rtGeometryInstanceDeclareVariable
  *
  */
  RTresult RTAPI rtGeometryInstanceRemoveVariable(RTgeometryinstance geometryinstance, RTvariable v);

  /**
  * @brief Returns the number of attached variables
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetVariableCount queries the number of variables attached to a geometry instance.
  * \a geometryinstance specifies the geometry instance, and should be a value returned by @ref rtGeometryInstanceCreate.
  * After the call, the number of variables attached to \a geometryinstance is returned to \a *count.
  *
  * @param[in]   geometryinstance   The GeometryInstance node to query from the number of attached variables
  * @param[out]  count              Returns the number of attached variables
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetVariableCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryInstanceGetVariableCount,
  * @ref rtGeometryInstanceDeclareVariable,
  * @ref rtGeometryInstanceRemoveVariable
  *
  */
  RTresult RTAPI rtGeometryInstanceGetVariableCount(RTgeometryinstance geometryinstance, unsigned int* count);

  /**
  * @brief Returns a handle to an indexed variable of a geometry instance node
  *
  * @ingroup GeometryInstance
  *
  * <B>Description</B>
  *
  * @ref rtGeometryInstanceGetVariable queries the handle of a geometry instance's indexed variable.
  * \a geometryinstance specifies the target geometry instance and should be a value returned by
  * @ref rtGeometryInstanceCreate. \a index specifies the index of the variable, and should be a
  * value less than @ref rtGeometryInstanceGetVariableCount. If \a index is the index of a variable
  * attached to \a geometryinstance, returns a handle to that variable in \a *v, and \a NULL
  * otherwise. \a *v must be declared first with @ref rtGeometryInstanceDeclareVariable before it
  * can be queried.
  *
  * @param[in]   geometryinstance   The GeometryInstance node from which to query a variable
  * @param[in]   index              The index that identifies the variable to be queried
  * @param[out]  v                  Returns handle to indexed variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtGeometryInstanceGetVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryDeclareVariable,
  * @ref rtGeometryGetVariableCount,
  * @ref rtGeometryRemoveVariable,
  * @ref rtGeometryQueryVariable
  *
  */
  RTresult RTAPI rtGeometryInstanceGetVariable(RTgeometryinstance geometryinstance, unsigned int index, RTvariable* v);

/************************************
 **
 **    Geometry object
 **
 ***********************************/

  /**
  * @brief Creates a new geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryCreate creates a new geometry node within a context. \a context
  * specifies the target context, and should be a value returned by @ref rtContextCreate.
  * Sets \a *geometry to the handle of a newly created geometry within \a context.
  * Returns @ref RT_ERROR_INVALID_VALUE if \a geometry is \a NULL.
  *
  * @param[in]   context    Specifies the rendering context of the Geometry node
  * @param[out]  geometry   New Geometry node handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryDestroy,
  * @ref rtGeometrySetBoundingBoxProgram,
  * @ref rtGeometrySetIntersectionProgram
  *
  */
  RTresult RTAPI rtGeometryCreate(RTcontext context, RTgeometry* geometry);

  /**
  * @brief Destroys a geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryDestroy removes \a geometry from its context and deletes it.  \a geometry should
  * be a value returned by @ref rtGeometryCreate.  Associated variables declared via
  * @ref rtGeometryDeclareVariable are destroyed, but no child graph nodes are destroyed.  After the
  * call, \a geometry is no longer a valid handle.
  *
  * @param[in]   geometry   Handle of the geometry node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryCreate,
  * @ref rtGeometrySetPrimitiveCount,
  * @ref rtGeometryGetPrimitiveCount
  *
  */
  RTresult RTAPI rtGeometryDestroy(RTgeometry geometry);

  /**
  * @brief Validates the geometry nodes integrity
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryValidate checks \a geometry for completeness. If \a geometry or any of the
  * objects attached to \a geometry are not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   geometry   The geometry node to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextValidate
  *
  */
  RTresult RTAPI rtGeometryValidate(RTgeometry geometry);

  /**
  * @brief Returns the context associated with a geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetContext queries a geometry node for its associated context.  \a geometry
  * specifies the geometry node to query, and should be a value returned by @ref
  * rtGeometryCreate. Sets \a *context to the context associated with \a geometry.
  *
  * @param[in]   geometry   Specifies the geometry to query
  * @param[out]  context    The context associated with \a geometry
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryCreate
  *
  */
  RTresult RTAPI rtGeometryGetContext(RTgeometry geometry, RTcontext* context);

  /**
  * @brief Sets the number of primitives
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometrySetPrimitiveCount sets the number of primitives \a primitiveCount in \a geometry.
  *
  * @param[in]   geometry         The geometry node for which to set the number of primitives
  * @param[in]   primitiveCount   The number of primitives
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetPrimitiveCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetPrimitiveCount
  *
  */
  RTresult RTAPI rtGeometrySetPrimitiveCount(RTgeometry geometry, unsigned int primitiveCount);

  /**
  * @brief Returns the number of primitives
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetPrimitiveCount returns for \a geometry the number of set primitives. The
  * number of primitvies can be set with @ref rtGeometryGetPrimitiveCount.
  *
  * @param[in]   geometry         Geometry node to query from the number of primitives
  * @param[out]  primitiveCount   Number of primitives
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetPrimitiveCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometrySetPrimitiveCount
  *
  */
  RTresult RTAPI rtGeometryGetPrimitiveCount(RTgeometry geometry, unsigned int* primitiveCount);

  /**
  * @brief Sets the primitive index offset
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometrySetPrimitiveIndexOffset sets the primitive index offset
  * \a indexOffset in \a geometry.  In the past, a @ref Geometry object's primitive
  * index range always started at zero (i.e., a Geometry with \a N primitives would
  * have a primitive index range of [0,N-1]).  The index offset is used to allow
  * @ref Geometry objects to have primitive index ranges starting at non-zero
  * positions (i.e., a Geometry with \a N primitives and an index offset of \a M
  * would have a primitive index range of [M,M+N-1]).  This feature enables the
  * sharing of vertex index buffers between multiple @ref Geometry objects.
  *
  * @param[in]   geometry       The geometry node for which to set the primitive index offset
  * @param[in]   indexOffset    The primitive index offset
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetPrimitiveIndexOffset was introduced in OptiX 3.5.
  *
  * <B>See also</B>
  * @ref rtGeometryGetPrimitiveIndexOffset
  *
  */
  RTresult RTAPI rtGeometrySetPrimitiveIndexOffset(RTgeometry geometry, unsigned int indexOffset);

  /**
  * @brief Returns the current primitive index offset
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetPrimitiveIndexOffset returns for \a geometry the primitive index offset. The
  * primitive index offset can be set with @ref rtGeometrySetPrimitiveIndexOffset.
  *
  * @param[in]   geometry       Geometry node to query for the primitive index offset
  * @param[out]  indexOffset    Primitive index offset
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetPrimitiveIndexOffset was introduced in OptiX 3.5.
  *
  * <B>See also</B>
  * @ref rtGeometrySetPrimitiveIndexOffset
  *
  */
  RTresult RTAPI rtGeometryGetPrimitiveIndexOffset(RTgeometry geometry, unsigned int* indexOffset);

  /**
  * @brief Sets the motion time range for a Geometry node.
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  * Sets the inclusive motion time range [timeBegin, timeEnd] for \a geometry,
  * where timeBegin <= timeEnd.  The default time range is [0.0, 1.0].  The
  * time range has no effect unless @ref rtGeometrySetMotionSteps is
  * called, in which case the time steps uniformly divide the time range.  See
  * @ref rtGeometrySetMotionSteps for additional requirements on the bounds
  * program.
  *
  * @param[in]   geometry    Geometry node handle
  * @param[out]  timeBegin   Beginning time value of range
  * @param[out]  timeEnd     Ending time value of range
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetMotionRange was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetMotionRange
  * @ref rtGeometrySetMotionBorderMode
  * @ref rtGeometrySetMotionSteps
  *
  */
  RTresult RTAPI rtGeometrySetMotionRange( RTgeometry geometry, float timeBegin, float timeEnd );

  /**
  * @brief Returns the motion time range associated with a Geometry node.
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  * @ref rtGeometryGetMotionRange returns the motion time range associated with
  * \a geometry from a previous call to @ref rtGeometrySetMotionRange, or the
  * default values of [0.0, 1.0].
  *
  *
  * @param[in]   geometry    Geometry node handle
  * @param[out]  timeBegin   Beginning time value of range
  * @param[out]  timeEnd     Ending time value of range
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetMotionRange was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtGeometrySetMotionRange
  * @ref rtGeometryGetMotionBorderMode
  * @ref rtGeometryGetMotionSteps
  *
  */
  RTresult RTAPI rtGeometryGetMotionRange( RTgeometry geometry, float* timeBegin, float* timeEnd );

  /**
  * @brief Sets the motion border modes of a Geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  * @ref rtGeometrySetMotionBorderMode sets the behavior of \a geometry
  * outside its motion time range. Options are @ref RT_MOTIONBORDERMODE_CLAMP
  * or @ref RT_MOTIONBORDERMODE_VANISH.  See @ref rtTransformSetMotionBorderMode
  * for details.
  *
  * @param[in]   geometry    Geometry node handle
  * @param[in]   beginMode   Motion border mode at motion range begin
  * @param[in]   endMode     Motion border mode at motion range end
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetMotionBorderMode was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetMotionBorderMode
  * @ref rtGeometrySetMotionRange
  * @ref rtGeometrySetMotionSteps
  *
  */
  RTresult RTAPI rtGeometrySetMotionBorderMode( RTgeometry geometry, RTmotionbordermode beginMode, RTmotionbordermode endMode );

  /**
  * @brief Returns the motion border modes of a Geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  * @ref rtGeometryGetMotionBorderMode returns the motion border modes
  * for the time range associated with \a geometry.
  *
  * @param[in]   geometry    Geometry node handle
  * @param[out]  beginMode   Motion border mode at motion range begin
  * @param[out]  endMode     Motion border mode at motion range end
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetMotionBorderMode was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtGeometrySetMotionBorderMode
  * @ref rtGeometryGetMotionRange
  * @ref rtGeometryGetMotionSteps
  *
  */
  RTresult RTAPI rtGeometryGetMotionBorderMode( RTgeometry geometry, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );

  /**
  * @brief Specifies the number of motion steps associated with a Geometry
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  * @ref rtGeometrySetMotionSteps sets the number of motion steps associated
  * with \a geometry.  If the value of \a n is greater than 1, then \a geometry
  * must have an associated bounding box program that takes both a primitive index
  * and a motion index as arguments, and computes an aabb at the motion index.
  * See @ref rtGeometrySetBoundingBoxProgram.
  *
  * Note that all Geometry has at least one 1 motion step (the default), and
  * Geometry that linearly moves has 2 motion steps.
  *
  * @param[in]   geometry    Geometry node handle
  * @param[in]   n           Number of motion steps >= 1
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetMotionSteps was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetMotionSteps
  * @ref rtGeometrySetMotionBorderMode
  * @ref rtGeometrySetMotionRange
  *
  */
  RTresult RTAPI rtGeometrySetMotionSteps( RTgeometry geometry, unsigned int n );

  /**
  * @brief Returns the number of motion steps associated with a Geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  * @ref rtGeometryGetMotionSteps returns in \a n the number of motion steps
  * associated with \a geometry.  Note that the default value is 1, not 0,
  * for geometry without motion.
  *
  * @param[in]   geometry    Geometry node handle
  * @param[out]  n           Number of motion steps n >= 1
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetMotionSteps was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetMotionSteps
  * @ref rtGeometrySetMotionBorderMode
  * @ref rtGeometrySetMotionRange
  *
  */
  RTresult RTAPI rtGeometryGetMotionSteps( RTgeometry geometry, unsigned int* n );

  /**
  * @brief Sets the bounding box program
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometrySetBoundingBoxProgram sets for \a geometry the \a program that computes an axis aligned bounding box
  * for each attached primitive to \a geometry. RTprogram's can be either generated with @ref rtProgramCreateFromPTXFile or
  * @ref rtProgramCreateFromPTXString. A bounding box program is mandatory for every geometry node.
  *
  * If \a geometry has more than one motion step, set using @ref rtGeometrySetMotionSteps, then the bounding
  * box program must compute a bounding box per primitive and per motion step.
  *
  * @param[in]   geometry   The geometry node for which to set the bounding box program
  * @param[in]   program    Handle to the bounding box program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetBoundingBoxProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetBoundingBoxProgram,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString
  *
  */
  RTresult RTAPI rtGeometrySetBoundingBoxProgram(RTgeometry geometry, RTprogram program);

  /**
  * @brief Returns the attached bounding box program
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetBoundingBoxProgram returns the handle \a program for
  * the attached bounding box program of \a geometry.
  *
  * @param[in]   geometry   Geometry node handle from which to query program
  * @param[out]  program    Handle to attached bounding box program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetBoundingBoxProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometrySetBoundingBoxProgram
  *
  */
  RTresult RTAPI rtGeometryGetBoundingBoxProgram(RTgeometry geometry, RTprogram* program);

  /**
  * @brief Sets the intersection program
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometrySetIntersectionProgram sets for \a geometry the \a program that performs ray primitive intersections.
  * RTprogram's can be either generated with @ref rtProgramCreateFromPTXFile or @ref rtProgramCreateFromPTXString. An intersection
  * program is mandatory for every geometry node.
  *
  * @param[in]   geometry   The geometry node for which to set the intersection program
  * @param[in]   program    A handle to the ray primitive intersection program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtGeometrySetIntersectionProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetIntersectionProgram,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString
  *
  */
  RTresult RTAPI rtGeometrySetIntersectionProgram(RTgeometry geometry, RTprogram program);

  /**
  * @brief Returns the attached intersection program
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetIntersectionProgram returns in \a program a handle of the attached intersection program.
  *
  * @param[in]   geometry   Geometry node handle to query program
  * @param[out]  program    Handle to attached intersection program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetIntersectionProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometrySetIntersectionProgram,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString
  *
  */
  RTresult RTAPI rtGeometryGetIntersectionProgram(RTgeometry geometry, RTprogram* program);

  /**
   * @brief Sets geometry flags
   *
   * @ingroup Geometry
   *
   * <B>Description</B>
   *
   * See @ref rtGeometryTrianglesSetFlagsPerMaterial for a description of the behavior of the
   * various flags.
   *
   * @param[in] geometry        The group handle
   * @param[out] flags          Flags for the given geometry group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometrySetFlags was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryTrianglesSetFlagsPerMaterial,
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometrySetFlags( RTgeometry geometry, RTgeometryflags flags );

  /**
   * @brief Retrieves geometry flags
   *
   * @ingroup Geometry
   *
   * <B>Description</B>
   *
   * See @ref rtGeometrySetFlags for details.
   *
   * @param[in] geometry        The group handle
   * @param[out] flags          Flags for the given geometry group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometryGetFlags was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryTrianglesSetFlagsPerMaterial,
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometryGetFlags( RTgeometry geometry, RTgeometryflags* flags );

  /**
  * Deprecated in OptiX 4.0. Calling this function has no effect.
  *
  */
  RTresult RTAPI rtGeometryMarkDirty(RTgeometry geometry);

  /**
  * Deprecated in OptiX 4.0. Calling this function has no effect.
  *
  */
  RTresult RTAPI rtGeometryIsDirty(RTgeometry geometry, int* dirty);

  /**
  * @brief Declares a new named variable associated with a geometry instance
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryDeclareVariable declares a new variable associated with a geometry node. \a
  * geometry specifies the target geometry node, and should be a value returned by @ref
  * rtGeometryCreate. \a name specifies the name of the variable, and should be a \a NULL-terminated
  * string. If there is currently no variable associated with \a geometry named \a name, a new
  * variable named \a name will be created and associated with \a geometry.  Returns the handle of
  * the newly-created variable in \a *v or \a NULL otherwise.  After declaration, the variable can
  * be queried with @ref rtGeometryQueryVariable or @ref rtGeometryGetVariable. A declared variable
  * does not have a type until its value is set with one of the @ref rtVariableSet functions. Once a
  * variable is set, its type cannot be changed anymore.
  *
  * @param[in]   geometry   Specifies the associated Geometry node
  * @param[in]   name       The name that identifies the variable
  * @param[out]  v          Returns a handle to a newly declared variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_REDECLARED
  * - @ref RT_ERROR_ILLEGAL_SYMBOL
  *
  * <B>History</B>
  *
  * @ref rtGeometryDeclareVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref Variables,
  * @ref rtGeometryQueryVariable,
  * @ref rtGeometryGetVariable,
  * @ref rtGeometryRemoveVariable
  *
  */
  RTresult RTAPI rtGeometryDeclareVariable(RTgeometry geometry, const char* name, RTvariable* v);

  /**
  * @brief Returns a handle to a named variable of a geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryQueryVariable queries the handle of a geometry node's named variable.
  * \a geometry specifies the target geometry node and should be a value returned
  * by @ref rtGeometryCreate. \a name specifies the name of the variable, and should
  * be a \a NULL-terminated string. If \a name is the name of a variable attached to
  * \a geometry, returns a handle to that variable in \a *v or \a NULL otherwise. Geometry
  * variables must be declared with @ref rtGeometryDeclareVariable before they can be queried.
  *
  * @param[in]   geometry   The geometry node to query from a variable
  * @param[in]   name       The name that identifies the variable to be queried
  * @param[out]  v          Returns the named variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtGeometryQueryVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryDeclareVariable,
  * @ref rtGeometryRemoveVariable,
  * @ref rtGeometryGetVariableCount,
  * @ref rtGeometryGetVariable
  *
  */
  RTresult RTAPI rtGeometryQueryVariable(RTgeometry geometry, const char* name, RTvariable* v);

  /**
  * @brief Removes a named variable from a geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryRemoveVariable removes a named variable from a geometry node. The
  * target geometry is specified by \a geometry, which should be a value
  * returned by @ref rtGeometryCreate. The variable to remove is specified by
  * \a v, which should be a value returned by @ref rtGeometryDeclareVariable.
  * Once a variable has been removed from this geometry node, another variable with the
  * same name as the removed variable may be declared.
  *
  * @param[in]   geometry   The geometry node from which to remove a variable
  * @param[in]   v          The variable to be removed
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtGeometryRemoveVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextRemoveVariable
  *
  */
  RTresult RTAPI rtGeometryRemoveVariable(RTgeometry geometry, RTvariable v);

  /**
  * @brief Returns the number of attached variables
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetVariableCount queries the number of variables attached to a geometry node.
  * \a geometry specifies the geometry node, and should be a value returned by @ref rtGeometryCreate.
  * After the call, the number of variables attached to \a geometry is returned to \a *count.
  *
  * @param[in]   geometry   The Geometry node to query from the number of attached variables
  * @param[out]  count      Returns the number of attached variables
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetVariableCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryGetVariableCount,
  * @ref rtGeometryDeclareVariable,
  * @ref rtGeometryRemoveVariable
  *
  */
  RTresult RTAPI rtGeometryGetVariableCount(RTgeometry geometry, unsigned int* count);

  /**
  * @brief Returns a handle to an indexed variable of a geometry node
  *
  * @ingroup Geometry
  *
  * <B>Description</B>
  *
  * @ref rtGeometryGetVariable queries the handle of a geometry node's indexed variable.
  * \a geometry specifies the target geometry and should be a value returned
  * by @ref rtGeometryCreate. \a index specifies the index of the variable, and
  * should be a value less than @ref rtGeometryGetVariableCount. If \a index is the
  * index of a variable attached to \a geometry, returns its handle in \a *v or \a NULL otherwise.
  * \a *v must be declared first with @ref rtGeometryDeclareVariable before it can be queried.
  *
  * @param[in]   geometry   The geometry node from which to query a variable
  * @param[in]   index      The index that identifies the variable to be queried
  * @param[out]  v          Returns handle to indexed variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtGeometryGetVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtGeometryDeclareVariable,
  * @ref rtGeometryGetVariableCount,
  * @ref rtGeometryRemoveVariable,
  * @ref rtGeometryQueryVariable
  *
  */
  RTresult RTAPI rtGeometryGetVariable(RTgeometry geometry, unsigned int index, RTvariable* v);

/************************************
 **
 **    GeometryTriangles object
 **
 ***********************************/

  /**
  * @brief Creates a new GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesCreate creates a new GeometryTriangles node within a context. \a context
  * specifies the target context, and should be a value returned by @ref rtContextCreate.
  * Sets \a *geometrytriangles to the handle of a newly created GeometryTriangles node within \a context.
  * Returns @ref RT_ERROR_INVALID_VALUE if \a geometrytriangles is \a NULL.
  *
  * @param[in]   context            Specifies the rendering context of the GeometryTriangles node
  * @param[out]  geometrytriangles  New GeometryTriangles node handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesCreate was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesDestroy,
  *
  */
  RTresult RTAPI rtGeometryTrianglesCreate(RTcontext context, RTgeometrytriangles* geometrytriangles);

  /**
  * @brief Destroys a GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesDestroy removes \a geometrytriangles from its context and deletes it.  \a geometrytriangles should
  * be a value returned by @ref rtGeometryTrianglesCreate.  After the call, \a geometrytriangles is no longer a valid handle.
  *
  * @param[in]   geometrytriangles   Handle of the GeometryTriangles node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesDestroy was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesCreate,
  * @ref rtGeometryTrianglesSetPrimitiveCount,
  * @ref rtGeometryTrianglesGetPrimitiveCount
  *
  */
  RTresult RTAPI rtGeometryTrianglesDestroy(RTgeometrytriangles geometrytriangles);

  /**
  * @brief Validates the GeometryTriangles nodes integrity
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesValidate checks \a geometrytriangles for completeness. If \a geometrytriangles or any of the
  * objects attached to \a geometrytriangles are not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   geometrytriangles   The GeometryTriangles node to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesValidate was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtContextValidate
  *
  */
  RTresult RTAPI rtGeometryTrianglesValidate(RTgeometrytriangles geometrytriangles);

  /**
  * @brief Returns the context associated with a GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetContext queries a GeometryTriangles node for its associated context.  \a geometrytriangles
  * specifies the GeometryTriangles node to query, and should be a value returned by @ref
  * rtGeometryTrianglesCreate. Sets \a *context to the context associated with \a geometrytriangles.
  *
  * @param[in]   geometrytriangles   Specifies the GeometryTriangles to query
  * @param[out]  context             The context associated with \a geometrytriangles
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetContext was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesCreate
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetContext(RTgeometrytriangles geometrytriangles, RTcontext* context);

  /**
  * @brief Sets the primitive index offset
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetPrimitiveIndexOffset sets the primitive index offset
  * \a indexOffset in \a geometrytriangles.
  * With an offset of zero, a GeometryTriangles with \a N triangles has a primitive index range of [0,N-1].
  * The index offset is used to allow GeometryTriangles objects to have primitive index ranges starting at non-zero
  * positions (i.e., a GeometryTriangles with \a N triangles and an index offset of \a M
  * has a primitive index range of [M,M+N-1]).
  * Note that this offset only affects the primitive index that is reported in case of an intersection and does not
  * affect the input data that is specified via @ref rtGeometryTrianglesSetVertices or @ref
  * rtGeometryTrianglesSetTriangleIndices.
  * This feature enables the packing of multiple Geometries or GeometryTriangles into a single buffer.
  * While the same effect could be reached via a user variable, it is recommended to specify the offset via
  * @ref rtGeometryTrianglesSetPrimitiveIndexOffset.
  *
  * @param[in]   geometrytriangles  The GeometryTriangles node for which to set the primitive index offset
  * @param[in]   indexOffset        The primitive index offset
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetPrimitiveIndexOffset was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometrySetPrimitiveIndexOffset
  * @ref rtGeometryTrianglesGetPrimitiveIndexOffset
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetPrimitiveIndexOffset(RTgeometrytriangles geometrytriangles, unsigned int indexOffset);

  /**
  * @brief Returns the current primitive index offset
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetPrimitiveIndexOffset returns for \a geometrytriangles the primitive index offset. The
  * primitive index offset can be set with @ref rtGeometryTrianglesSetPrimitiveIndexOffset.
  *
  * @param[in]   geometrytriangles  GeometryTriangles node to query for the primitive index offset
  * @param[out]  indexOffset        Primitive index offset
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetPrimitiveIndexOffset was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetPrimitiveIndexOffset
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetPrimitiveIndexOffset(RTgeometrytriangles geometrytriangles, unsigned int* indexOffset);

  /**
  * @brief Sets a pre-transform matrix
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetPreTransformMatrix can be used to bake a transformation for a mesh.
  * Vertices of triangles are multiplied by the user-specified 3x4 matrix before the acceleration build.
  * Note that the input triangle data stays untouched (set via @ref rtGeometryTrianglesSetVertices).
  * Triangle intersection uses transformed triangles.
  * The 3x4 matrix is expected to be in a row-major data layout, use the transpose option if \a matrix is in a column-major data layout.
  * Use rtGeometryTrianglesSetPreTransformMatrix(geometrytriangles, false, 0); to unset a previously set matrix.
  *
  * @param[in]   geometrytriangles  Geometry node to query from the number of primitives
  * @param[in]   transpose          If the input matrix is column-major and needs to be transposed before usage
  * @param[in]   matrix             The 3x4 matrix that is used to transform the vertices
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetPreTransformMatrix was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetPreTransformMatrix
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetPreTransformMatrix( RTgeometrytriangles geometrytriangles, int transpose, const float* matrix );

  /**
  * @brief Gets a pre-transform matrix
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetPreTransformMatrix returns a previously set 3x4 matrix or the 'identity' matrix (with ones in the main diagonal of the 3x3 submatrix) if no matrix is set.
  *
  * @param[in]   geometrytriangles  Geometry node to query from the number of primitives
  * @param[in]   transpose          Set to true if the output matrix is expected to be column-major rather than row-major
  * @param[out]  matrix             The 3x4 matrix that is used to transform the vertices
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetPreTransformMatrix was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetPreTransformMatrix
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetPreTransformMatrix( RTgeometrytriangles geometrytriangles, int transpose, float* matrix );

  /**
  * @brief Sets the number of triangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetPrimitiveCount sets the number of triangles \a triangleCount in \a geometrytriangles.
  * A triangle geometry is either a triangle soup for which every three vertices stored in the vertex buffer form a triangle,
  * or indexed triangles are used for which three indices reference different vertices.
  * In the latter case, an index buffer must be set (@ref rtGeometryTrianglesSetTriangleIndices).
  * The vertices of the triangles are specified via one of the SetVertices functions.
  *
  * @param[in]   geometrytriangles  GeometryTriangles node for which to set the number of triangles
  * @param[in]   triangleCount      Number of triangles
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetPrimitiveCount was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetPrimitiveCount
  * @ref rtGeometrySetPrimitiveCount
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetPrimitiveCount( RTgeometrytriangles geometrytriangles, unsigned int triangleCount );

  /**
  * @brief Returns the number of triangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetPrimitiveCount returns the number of set triangles for \a geometrytriangles. The
  * number of primitives can be set with @ref rtGeometryTrianglesSetPrimitiveCount.
  *
  * @param[in]   geometrytriangles  GeometryTriangles node to query from the number of primitives
  * @param[out]  triangleCount      Number of triangles
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetPrimitiveCount was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetPrimitiveCount
  * @ref rtGeometryGetPrimitiveCount
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetPrimitiveCount(RTgeometrytriangles geometrytriangles, unsigned int* triangleCount);

  /**
  * @brief Sets the index buffer of indexed triangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetTriangleIndices is used to set the index buffer for indexed triangles.
  * Triplets of indices from buffer \a indexBuffer index vertices to form triangles.
  * If the buffer is set, it is assumed that the geometry is given as indexed triangles.
  * If the index buffer is not set, it is assumed that the geometry is given as a triangle soup.
  * A previously set index buffer can be unset by passing NULL as \a indexBuffer parameter, e.g., rtGeometryTrianglesSetTriangleIndices( geometrytriangles, NULL, 0, 0, RT_FORMAT_UNSIGNED_INT3);
  * Buffer \a indexBuffer is expected to hold 3 times \a triangleCount indices (see @ref rtGeometryTrianglesSetPrimitiveCount).
  * Parameter \a indexBufferByteOffset can be used to specify a byte offset to the first index in buffer \a indexBuffer.
  * Parameter \a triIndicesByteStride sets the stride in bytes between triplets of indices. There mustn't be any spacing between indices within a triplet, spacing is only supported between triplets.
  * Parameter \a triIndicesFormat must be one of the following: RT_FORMAT_UNSIGNED_INT3, RT_FORMAT_UNSIGNED_SHORT3.
  *
  * @param[in]   geometrytriangles               GeometryTriangles node to query for the primitive index offset
  * @param[in]   indexBuffer                     Buffer that holds the indices into the vertex buffer of the triangles
  * @param[in]   indexBufferByteOffset           Offset in bytes to the first index in buffer indexBuffer
  * @param[in]   triIndicesByteStride            Stride in bytes between triplets of indices
  * @param[in]   triIndicesFormat                Format of the triplet of indices to index the vertices of a triangle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetTriangleIndices was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetVertices
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetTriangleIndices(  RTgeometrytriangles geometrytriangles,
                                                         RTbuffer            indexBuffer,
                                                         RTsize              indexBufferByteOffset,
                                                         RTsize              triIndicesByteStride,
                                                         RTformat            triIndicesFormat );

  /**
  * @brief Sets the vertex buffer of a triangle soup
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetVertices interprets the buffer \a vertexBuffer as the vertices of triangles of the GeometryTriangles \a geometrytriangles.
  * The number of vertices is set as \a vertexCount.
  * If an index buffer is set, it is assumed that the geometry is given as indexed triangles.
  * If the index buffer is not set, it is assumed that the geometry is given as a triangle soup and \a vertexCount must be 3 times triangleCount (see @ref rtGeometryTrianglesSetPrimitiveCount).
  * Buffer \a vertexBuffer is expected to hold \a vertexCount vertices.
  * Parameter \a vertexBufferByteOffset can be used to specify a byte offset to the position of the first vertex in buffer \a vertexBuffer.
  * Parameter \a vertexByteStride sets the stride in bytes between vertices.
  * Parameter \a positionFormat must be one of the following: RT_FORMAT_FLOAT3, RT_FORMAT_HALF3, RT_FORMAT_FLOAT2, RT_FORMAT_HALF2.
  * In case of formats RT_FORMAT_FLOAT2 or RT_FORMAT_HALF2 the third component is assumed to be zero, which can be useful for planar geometry.
  * Calling this function overrides any previous call to any of the set(Motion)Vertices functions.
  *
  * @param[in]   geometrytriangles            GeometryTriangles node to query for the primitive index offset
  * @param[in]   vertexCount                  Number of vertices of the geometry
  * @param[in]   vertexBuffer                 Buffer that holds the vertices of the triangles
  * @param[in]   vertexBufferByteOffset       Offset in bytes to the first vertex in buffer vertexBuffer
  * @param[in]   vertexByteStride             Stride in bytes between vertices
  * @param[in]   positionFormat               Format of the position attribute of a vertex
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetVertices was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetTriangleIndices
  * @ref rtGeometryTrianglesSetMotionVertices
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetVertices( RTgeometrytriangles geometrytriangles,
                                                 unsigned int        vertexCount,
                                                 RTbuffer            vertexBuffer,
                                                 RTsize              vertexBufferByteOffset,
                                                 RTsize              vertexByteStride,
                                                 RTformat            positionFormat );

  /**
  * @brief Sets the vertex buffer of motion triangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetMotionVertices interprets the buffer \a vertexBuffer as the vertices of triangles of the GeometryTriangles \a geometrytriangles.
  * The number of triangles for one motion step is set as \a vertexCount.
  * Similar to it's non-motion counterpart, \a vertexCount must be 3 times \a triangleCount if no index buffer is set.
  * The total number of vertices stored in \a vertexBuffer is \a vertexCount times \a motionStepCount (see @ref rtGeometryTrianglesSetMotionSteps).
  * Triangles are linearly interpolated between motion steps.
  * Parameter \a vertexBufferByteOffset can be used to specify a byte offset to the position of the first vertex of the first motion step in buffer \a vertexBuffer.
  * Parameter \a vertexByteStride sets the stride in bytes between vertices within a motion step.
  * Parameter \a vertexMotionStepByteStride sets the stride in bytes between motion steps for a single vertex.
  * The stride parameters allow for two types of layouts of the motion data:
  * a) serialized: vertexByteStride = sizeof(Vertex), vertexMotionStepByteStride = vertexCount * vertexByteStride
  * b) interleaved: vertexMotionStepByteStride = sizeof(Vertex), vertexByteStride = sizeof(Vertex) * motion_steps
  * Vertex N at time step i is at: vertexBuffer[N * vertexByteStride + i * vertexMotionStepByteStride + vertexBufferByteOffset]
  * Parameter \a positionFormat must be one of the following: RT_FORMAT_FLOAT3, RT_FORMAT_HALF3, RT_FORMAT_FLOAT2, RT_FORMAT_HALF2.
  * In case of formats RT_FORMAT_FLOAT2 or RT_FORMAT_HALF2 the third component is assumed to be zero, which can be useful for planar geometry.
  * Calling this function overrides any previous call to any of the set(Motion)Vertices functions.
  *
  * @param[in]   geometrytriangles               GeometryTriangles node to query for the primitive index offset
  * @param[in]   vertexCount                     Number of vertices for one motion step
  * @param[in]   vertexBuffer                    Buffer that holds the vertices of the triangles for all motion steps
  * @param[in]   vertexBufferByteOffset          Offset in bytes to the first vertex of the first motion step in buffer vertexBuffer
  * @param[in]   vertexByteStride                Stride in bytes between vertices, belonging to the same motion step
  * @param[in]   vertexMotionStepByteStride      Stride in bytes between vertices of the same triangle, but neighboring motion step
  * @param[in]   positionFormat                  Format of the position attribute of a vertex
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetMotionVertices was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetVertices
  * @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMotionVertices( RTgeometrytriangles geometrytriangles,
                                                       unsigned int        vertexCount,
                                                       RTbuffer            vertexBuffer,
                                                       RTsize              vertexBufferByteOffset,
                                                       RTsize              vertexByteStride,
                                                       RTsize              vertexMotionStepByteStride,
                                                       RTformat            positionFormat );

  /**
  * @brief Sets the vertex buffer of motion triangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer can be used instead of @ref rtGeometryTrianglesSetMotionVertices if the vertices for the different motion steps are stored in separate buffers.
  * Parameter \a vertexBuffers must point to an array of buffers of minimal size \a motionStepCount (see @ref rtGeometryTrianglesSetMotionSteps).
  * All buffers must, however, share the same byte offset as well as vertex stride and position format.
  * Calling this function overrides any previous call to any of the set(Motion)Vertices functions.
  *
  * @param[in]   geometrytriangles               GeometryTriangles node to query for the primitive index offset
  * @param[in]   vertexCount                     Number of vertices for one motion step
  * @param[in]   vertexBuffers                   Buffers that hold the vertices of the triangles per motion step
  * @param[in]   vertexBufferCount               Number of buffers passed, must match the number of motion steps before a launch call
  * @param[in]   vertexBufferByteOffset          Offset in bytes to the first vertex in every buffer vertexBuffers
  * @param[in]   vertexByteStride                Stride in bytes between vertices, belonging to the same motion step
  * @param[in]   positionFormat                  Format of the position attribute of a vertex
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetMotionVertices was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetVertices
  * @ref rtGeometryTrianglesSetMotionVertices
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMotionVerticesMultiBuffer( RTgeometrytriangles geometrytriangles,
                                                                  unsigned int        vertexCount,
                                                                  RTbuffer*           vertexBuffers,
                                                                  unsigned int        vertexBufferCount,
                                                                  RTsize              vertexBufferByteOffset,
                                                                  RTsize              vertexByteStride,
                                                                  RTformat            positionFormat );

  /**
  * @brief Sets the number of motion steps associated with a GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesSetMotionSteps sets the number of motion steps as specified in \a motionStepCount
  * associated with \a geometrytriangles.  Note that the default value is 1, not 0,
  * for geometry without motion.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[in]   motionStepCount      Number of motion steps, motionStepCount >= 1
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetMotionSteps was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMotionVertices
  * @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer
  * @ref rtGeometryTrianglesGetMotionSteps
  * @ref rtGeometryTrianglesSetMotionBorderMode
  * @ref rtGeometryTrianglesSetMotionRange
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMotionSteps( RTgeometrytriangles geometrytriangles, unsigned int motionStepCount );

  /**
  * @brief Returns the number of motion steps associated with a GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesGetMotionSteps returns in \a motionStepCount the number of motion steps
  * associated with \a geometrytriangles.  Note that the default value is 1, not 0,
  * for geometry without motion.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[out]  motionStepCount      Number of motion steps motionStepCount >= 1
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetMotionSteps was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMotionSteps
  * @ref rtGeometryTrianglesGetMotionBorderMode
  * @ref rtGeometryTrianglesGetMotionRange
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetMotionSteps( RTgeometrytriangles geometrytriangles, unsigned int* motionStepCount );

  /**
  * @brief Sets the motion time range for a GeometryTriangles node.
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * Sets the inclusive motion time range [timeBegin, timeEnd] for \a geometrytriangles,
  * where timeBegin <= timeEnd.  The default time range is [0.0, 1.0].  The
  * time range has no effect unless @ref rtGeometryTrianglesSetMotionVertices or
  * @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer with motionStepCount > 1 is
  * called, in which case the time steps uniformly divide the time range.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[out]  timeBegin            Beginning time value of range
  * @param[out]  timeEnd              Ending time value of range
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetMotionRange was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetMotionRange
  * @ref rtGeometryTrianglesSetMotionBorderMode
  * @ref rtGeometryTrianglesGetMotionSteps
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMotionRange( RTgeometrytriangles geometrytriangles, float timeBegin, float timeEnd );

  /**
  * @brief Returns the motion time range associated with a GeometryTriangles node.
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesGetMotionRange returns the motion time range associated with
  * \a geometrytriangles from a previous call to @ref rtGeometryTrianglesSetMotionRange, or the
  * default values of [0.0, 1.0].
  *
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[out]  timeBegin            Beginning time value of range
  * @param[out]  timeEnd              Ending time value of range
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetMotionRange was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMotionRange
  * @ref rtGeometryTrianglesGetMotionBorderMode
  * @ref rtGeometryTrianglesGetMotionSteps
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetMotionRange( RTgeometrytriangles geometrytriangles, float* timeBegin, float* timeEnd );

  /**
  * @brief Sets the motion border modes of a GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesSetMotionBorderMode sets the behavior of \a geometrytriangles
  * outside its motion time range. Options are @ref RT_MOTIONBORDERMODE_CLAMP
  * or @ref RT_MOTIONBORDERMODE_VANISH.  See @ref rtTransformSetMotionBorderMode
  * for details.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[in]   beginMode            Motion border mode at motion range begin
  * @param[in]   endMode              Motion border mode at motion range end
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetMotionBorderMode was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetMotionBorderMode
  * @ref rtGeometryTrianglesSetMotionRange
  * @ref rtGeometryTrianglesGetMotionSteps
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMotionBorderMode( RTgeometrytriangles geometrytriangles, RTmotionbordermode beginMode, RTmotionbordermode endMode );

  /**
  * @brief Returns the motion border modes of a GeometryTriangles node
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesGetMotionBorderMode returns the motion border modes
  * for the time range associated with \a geometrytriangles.
  *
  * @param[in]   geometrytriangles   GeometryTriangles node handle
  * @param[out]  beginMode           Motion border mode at motion range begin
  * @param[out]  endMode             Motion border mode at motion range end
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetMotionBorderMode was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMotionBorderMode
  * @ref rtGeometryTrianglesGetMotionRange
  * @ref rtGeometryTrianglesGetMotionSteps
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetMotionBorderMode( RTgeometrytriangles geometrytriangles, RTmotionbordermode* beginMode, RTmotionbordermode* endMode );

  /**
  * @brief Sets flags that influence the behavior of traversal
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesSetBuildFlags can be used to set object-specific flags that affect the acceleration-structure-build behavior.
  * If parameter \a buildFlags contains the RT_GEOMETRY_BUILD_FLAG_RELEASE_BUFFERS flag, all buffers (including the vertex, index, and materialIndex buffer) holding
  * information that is evaluated at acceleration-structure-build time will be released after the build.
  * OptiX does not take ownership over the buffers, but simply frees the corresponding device memory.
  * Sharing buffers with other GeometryTriangles nodes is possible if all of them are built within one OptiX launch.
  * Note that it is the users responsibility that the buffers hold data for the next acceleration structure build if the acceleration structure is marked dirty.
  * E.g., if the flag is set, an OptiX launch will cause the acceleration structure build and release the memory afterwards.
  * If the acceleration structure is marked dirty before the next launch (e.g., due to refitting), the user needs to map the buffers before the launch to fill them with data.
  * Further, there are certain configurations with motion when the buffers cannot be released in which case the flag is ignored and the data is not freed.
  * The buffers can only be released if all GeometryTriangles belonging to a GeometryGroup have the same number of motion steps and equal motion begin / end times.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[in]   buildFlags           The flags to set
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetBuildFlags was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetBuildFlags
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetBuildFlags( RTgeometrytriangles geometrytriangles, RTgeometrybuildflags buildFlags );

  /**
  * @brief Sets the number of materials used for the GeometryTriangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesGetMaterialCount returns the number of materials that are used with \a geometrytriangles.
  * By default there is one material slot.

  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[out]  numMaterials         Number of materials used with this GeometryTriangles node
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetMaterialCount was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMaterialCount
  *
  */
  RTresult RTAPI rtGeometryTrianglesGetMaterialCount( RTgeometrytriangles geometrytriangles, unsigned int* numMaterials );

  /**
  * @brief Sets the number of materials used for the GeometryTriangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesSetMaterialCount sets the number of materials that are used with \a geometrytriangles.
  * By default there is one material slot.
  * This number must be equal to the number of materials that is set at the GeometryInstance where \a geometrytriangles is attached to.
  * Multi-material support for GeometryTriangles is limited to a fixed partition of the geometry into sets of triangles.
  * Each triangle set maps to one material slot (within range [0, numMaterials-1]).
  * The mapping is set via @ref rtGeometryTrianglesSetMaterialIndices.
  * The actual materials are set at the GeometryInstance.
  * The geometry can be instanced when attached to multiple GeometryInstances.
  * In that case, the materials attached to each GeometryInstance can differ (effectively causing different materials per instance of the geometry).
  * \a numMaterials must be >=1 and <=2^16.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[in]   numMaterials         Number of materials used with this geometry
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetMaterialCount was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetMaterialCount
  * @ref rtGeometryTrianglesSetMaterialIndices
  * @ref rtGeometryTrianglesSetFlagsPerMaterial
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMaterialCount( RTgeometrytriangles geometrytriangles, unsigned int numMaterials );


  /**
  * @brief Sets the index buffer of indexed triangles
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetMaterialIndices set the material slot per triangle of \a geometrytriangles.
  * Hence, buffer \a materialIndexBuffer must hold triangleCount entries.
  * Every material index must be in range [0, numMaterials-1] (see @ref rtGeometryTrianglesSetMaterialCount).
  * Parameter \a materialIndexBufferByteOffset can be used to specify a byte offset to the first index in buffer \a materialIndexBuffer.
  * Parameter \a materialIndexByteStride sets the stride in bytes between indices.
  * Parameter \a materialIndexFormat must be one of the following: RT_FORMAT_UNSIGNED_INT, RT_FORMAT_UNSIGNED_SHORT, RT_FORMAT_UNSIGNED_BYTE.
  * The buffer is only used if the number of materials as set via @ref rtGeometryTrianglesSetMaterialCount is larger than one.
  *
  * @param[in]   geometrytriangles                   GeometryTriangles node to query for the primitive index offset
  * @param[in]   materialIndexBuffer                 Buffer that holds the indices into the vertex buffer of the triangles
  * @param[in]   materialIndexBufferByteOffset       Offset to first index in buffer indexBuffer
  * @param[in]   materialIndexByteStride             Stride in bytes between triplets of indices
  * @param[in]   materialIndexFormat                 Format of the triplet of indices to index the vertices of a triangle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetMaterialIndices was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMaterialCount
  * @ref rtGeometryTrianglesSetFlagsPerMaterial
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetMaterialIndices( RTgeometrytriangles geometrytriangles,
                                                        RTbuffer            materialIndexBuffer,
                                                        RTsize              materialIndexBufferByteOffset,
                                                        RTsize              materialIndexByteStride,
                                                        RTformat            materialIndexFormat );

  /**
  * @brief Sets geometry-specific flags that influence the behavior of traversal
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  * @ref rtGeometryTrianglesSetFlagsPerMaterial can be used to set geometry-specific flags that may
  * change the behavior of traversal when intersecting the geometry.
  * Note that the flags are evaluated at acceleration structure build time.
  * An acceleration must be marked dirty for changes to the flags to take effect.
  * Setting the flags RT_GEOMETRY_FLAG_NO_SPLITTING and/or RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be dependent on the
  * material that is used for the intersection.
  * Therefore, the flags are set per material slot (with the actual material binding being set on the GeomteryInstance).
  * If the geometry is instanced and different instances apply different materials to the geometry, the per-material geometry-specific flags
  * need to apply to the materials of all instances.
  * Example with two instances with each having two materials, node graph:
  *        G
  *       / \
  *      /   \
  *     T0    T1
  *     |     |
  *    GG0-A-GG1
  *     |     |
  * M0-GI0   GI1-M2
  *    /  \ /  \
  *  M1    GT   M3
  * with: G-Group, GG-GeometryGroup, T-Transform, A-Acceleration, GI-GeometryInstance, M-Material, GT-GeometryTriangles
  * RT_GEOMETRY_FLAG_NO_SPLITTING needs to be set for material index 0, if M0 or M2 require it.
  * RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be set for material index 0, if M0 and M2 allow it.
  * RT_GEOMETRY_FLAG_NO_SPLITTING needs to be set for material index 1, if M1 or M3 require it.
  * RT_GEOMETRY_FLAG_DISABLE_ANYHIT should be set for material index 1, if M1 and M3 allow it.
  *
  * Setting RT_GEOMETRY_FLAG_NO_SPLITTING prevents splitting the primitive during the acceleration structure build.
  * Splitting is done to increase performance, but as a side-effect may result in multiple executions of
  * the any-hit program for a single intersection.
  * To avoid further side effects (e.g., multiple accumulations of a value) that may result of a multiple execution,
  * RT_GEOMETRY_FLAG_NO_SPLITTING needs to be set.
  * RT_GEOMETRY_FLAG_DISABLE_ANYHIT is an optimization due to which the execution of the any-hit program is skipped.
  * If possible, the flag should be set.
  * Note that if no any-hit program is set on a material by the user, a no-op any-hit program will be used.
  * Therefore, this flag still needs to be set to skip the execution of any any-hit program.
  * An automatic determination of whether to set the DISABLE_ANYHIT flag is not possible since the information
  * whether or not to skip the any-hit program depends on the materials that are used, and this information
  * may not be available at acceleration build time.
  * For example, materials can change afterwards (e.g., between frames) without a rebuild of an acceleration.
  * Note that the final decision whether or not to execute the any-hit program at run time also depends on the flags set on
  * the ray as well as the geometry group that this geometry is part of.
  *
  * @param[in]   geometrytriangles    GeometryTriangles node handle
  * @param[in]   materialIndex        The material index for which to set the flags
  * @param[in]   flags                The flags to set.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetFlagsPerMaterial was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesSetMaterialCount
  * @ref rtGeometryTrianglesSetMaterialIndices
  * @ref rtGeometryTrianglesSetBuildFlags
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetFlagsPerMaterial( RTgeometrytriangles geometrytriangles,
                                                         unsigned int        materialIndex,
                                                         RTgeometryflags     flags );

  /**
   * @brief Gets geometry flags for triangles.
   *
   * @ingroup GeometryTriangles
   *
   * <B>Description</B>
   *
   * See @ref rtGeometryTrianglesSetFlagsPerMaterial for details.
   *
   * @param[in] triangles       The triangles handle
   * @param[in] materialIndex   The index of the material for which to retrieve the flags
   * @param[out] flags          Flags for the given geometry group
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RT_SUCCESS
   * - @ref RT_ERROR_INVALID_VALUE
   *
   * <B>History</B>
   *
   * @ref rtGeometryTrianglesGetFlagsPerMaterial was introduced in OptiX 6.0.
   *
   * <B>See also</B>
   * @ref rtGeometryTrianglesSetFlagsPerMaterial,
   * @ref rtGeometryTrianglesSetMaterialIndices
   * @ref rtTrace
   */
  RTresult RTAPI rtGeometryTrianglesGetFlagsPerMaterial( RTgeometrytriangles triangles,
                                                         unsigned int        materialIndex,
                                                         RTgeometryflags*    flags );

  /************************************
 **
 **    Material object
 **
 ***********************************/

  /**
  * @brief Creates a new material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialCreate creates a new material within a context. \a context specifies the target
  * context, as returned by @ref rtContextCreate. Sets \a *material to the handle of a newly
  * created material within \a context. Returns @ref RT_ERROR_INVALID_VALUE if \a material is \a NULL.
  *
  * @param[in]   context    Specifies a context within which to create a new material
  * @param[out]  material   Returns a newly created material
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtMaterialCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialDestroy,
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtMaterialCreate(RTcontext context, RTmaterial* material);

  /**
  * @brief Destroys a material object
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialDestroy removes \a material from its context and deletes it.  \a material should
  * be a value returned by @ref rtMaterialCreate.  Associated variables declared via @ref
  * rtMaterialDeclareVariable are destroyed, but no child graph nodes are destroyed.  After the
  * call, \a material is no longer a valid handle.
  *
  * @param[in]   material   Handle of the material node to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtMaterialDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialDestroy(RTmaterial material);

  /**
  * @brief Verifies the state of a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialValidate checks \a material for completeness. If \a material or
  * any of the objects attached to \a material are not valid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   material   Specifies the material to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtMaterialValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialValidate(RTmaterial material);

  /**
  * @brief Returns the context associated with a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialGetContext queries a material for its associated context.
  * \a material specifies the material to query, and should be a value returned by
  * @ref rtMaterialCreate. If both parameters are valid, \a *context
  * sets to the context associated with \a material. Otherwise, the call
  * has no effect and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   material   Specifies the material to query
  * @param[out]  context    Returns the context associated with the material
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtMaterialGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialGetContext(RTmaterial material, RTcontext* context);

  /**
  * @brief Sets the closest-hit program associated with a (material, ray type) tuple
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialSetClosestHitProgram specifies a closest-hit program to associate
  * with a (material, ray type) tuple. \a material specifies the material of
  * interest and should be a value returned by @ref rtMaterialCreate.
  * \a rayTypeIndex specifies the type of ray to which the program applies and
  * should be a value less than the value returned by @ref rtContextGetRayTypeCount.
  * \a program specifies the target closest-hit program which applies to
  * the tuple (\a material, \a rayTypeIndex) and should be a value returned by
  * either @ref rtProgramCreateFromPTXString or @ref rtProgramCreateFromPTXFile.
  *
  * @param[in]   material         Specifies the material of the (material, ray type) tuple to modify
  * @param[in]   rayTypeIndex     Specifies the ray type of the (material, ray type) tuple to modify
  * @param[in]   program          Specifies the closest-hit program to associate with the (material, ray type) tuple
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtMaterialSetClosestHitProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialGetClosestHitProgram,
  * @ref rtMaterialCreate,
  * @ref rtContextGetRayTypeCount,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtProgramCreateFromPTXFile
  *
  */
  RTresult RTAPI rtMaterialSetClosestHitProgram(RTmaterial material, unsigned int rayTypeIndex, RTprogram program);

  /**
  * @brief Returns the closest-hit program associated with a (material, ray type) tuple
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialGetClosestHitProgram queries the closest-hit program associated
  * with a (material, ray type) tuple. \a material specifies the material of
  * interest and should be a value returned by @ref rtMaterialCreate.
  * \a rayTypeIndex specifies the target ray type and should be a value
  * less than the value returned by @ref rtContextGetRayTypeCount.
  * If all parameters are valid, \a *program sets to the handle of the
  * any-hit program associated with the tuple (\a material, \a rayTypeIndex).
  * Otherwise, the call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   material         Specifies the material of the (material, ray type) tuple to query
  * @param[in]   rayTypeIndex     Specifies the type of ray of the (material, ray type) tuple to query
  * @param[out]  program          Returns the closest-hit program associated with the (material, ray type) tuple
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtMaterialGetClosestHitProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialSetClosestHitProgram,
  * @ref rtMaterialCreate,
  * @ref rtContextGetRayTypeCount
  *
  */
  RTresult RTAPI rtMaterialGetClosestHitProgram(RTmaterial material, unsigned int rayTypeIndex, RTprogram* program);

  /**
  * @brief Sets the any-hit program associated with a (material, ray type) tuple
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialSetAnyHitProgram specifies an any-hit program to associate with a
  * (material, ray type) tuple. \a material specifies the target material and
  * should be a value returned by @ref rtMaterialCreate. \a rayTypeIndex specifies
  * the type of ray to which the program applies and should be a value less than
  * the value returned by @ref rtContextGetRayTypeCount. \a program specifies the
  * target any-hit program which applies to the tuple (\a material,
  * \a rayTypeIndex) and should be a value returned by either
  * @ref rtProgramCreateFromPTXString or @ref rtProgramCreateFromPTXFile.
  *
  * @param[in]   material         Specifies the material of the (material, ray type) tuple to modify
  * @param[in]   rayTypeIndex     Specifies the type of ray of the (material, ray type) tuple to modify
  * @param[in]   program          Specifies the any-hit program to associate with the (material, ray type) tuple
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtMaterialSetAnyHitProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialGetAnyHitProgram,
  * @ref rtMaterialCreate,
  * @ref rtContextGetRayTypeCount,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtProgramCreateFromPTXFile
  *
  */
  RTresult RTAPI rtMaterialSetAnyHitProgram(RTmaterial material, unsigned int rayTypeIndex, RTprogram program);

  /**
  * @brief Returns the any-hit program associated with a (material, ray type) tuple
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialGetAnyHitProgram queries the any-hit program associated
  * with a (material, ray type) tuple. \a material specifies the material of
  * interest and should be a value returned by @ref rtMaterialCreate.
  * \a rayTypeIndex specifies the target ray type and should be a value
  * less than the value returned by @ref rtContextGetRayTypeCount.
  * if all parameters are valid, \a *program sets to the handle of the
  * any-hit program associated with the tuple (\a material, \a rayTypeIndex).
  * Otherwise, the call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   material         Specifies the material of the (material, ray type) tuple to query
  * @param[in]   rayTypeIndex     Specifies the type of ray of the (material, ray type) tuple to query
  * @param[out]  program          Returns the any-hit program associated with the (material, ray type) tuple
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtMaterialGetAnyHitProgram was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialSetAnyHitProgram,
  * @ref rtMaterialCreate,
  * @ref rtContextGetRayTypeCount
  *
  */
  RTresult RTAPI rtMaterialGetAnyHitProgram(RTmaterial material, unsigned int rayTypeIndex, RTprogram* program);

  /**
  * @brief Declares a new named variable to be associated with a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialDeclareVariable declares a new variable to be associated with a material.
  * \a material specifies the target material, and should be a value returned by @ref
  * rtMaterialCreate. \a name specifies the name of the variable, and should be a \a NULL-terminated
  * string. If there is currently no variable associated with \a material named \a name, and \a v is
  * not \a NULL, a new variable named \a name will be created and associated with \a material and \a
  * *v will be set to the handle of the newly-created variable. Otherwise, this call has no effect
  * and returns either @ref RT_ERROR_INVALID_VALUE if either \a name or \a v is \a NULL or @ref
  * RT_ERROR_VARIABLE_REDECLARED if \a name is the name of an existing variable associated with the
  * material.
  *
  * @param[in]   material   Specifies the material to modify
  * @param[in]   name       Specifies the name of the variable
  * @param[out]  v          Returns a handle to a newly declared variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_REDECLARED
  * - @ref RT_ERROR_ILLEGAL_SYMBOL
  *
  * <B>History</B>
  *
  * @ref rtMaterialDeclareVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialGetVariable,
  * @ref rtMaterialQueryVariable,
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialDeclareVariable(RTmaterial material, const char* name, RTvariable* v);

  /**
  * @brief Queries for the existence of a named variable of a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialQueryVariable queries for the existence of a material's named variable. \a
  * material specifies the target material and should be a value returned by @ref rtMaterialCreate.
  * \a name specifies the name of the variable, and should be a \a NULL-terminated
  * string. If \a material is a valid material and \a name is the name of a variable attached to \a
  * material, \a *v is set to a handle to that variable after the call. Otherwise, \a *v is set to
  * \a NULL. If \a material is not a valid material, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   material   Specifies the material to query
  * @param[in]   name       Specifies the name of the variable to query
  * @param[out]  v          Returns a the named variable, if it exists
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtMaterialQueryVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialGetVariable,
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialQueryVariable(RTmaterial material, const char* name, RTvariable* v);

  /**
  * @brief Removes a variable from a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialRemoveVariable removes a variable from a material. The material of
  * interest is specified by \a material, which should be a value returned by
  * @ref rtMaterialCreate. The variable to remove is specified by \a v, which
  * should be a value returned by @ref rtMaterialDeclareVariable. Once a variable
  * has been removed from this material, another variable with the same name as the
  * removed variable may be declared. If \a material does not refer to a valid material,
  * this call has no effect and returns @ref RT_ERROR_INVALID_VALUE. If \a v is not
  * a valid variable or does not belong to \a material, this call has no effect and
  * returns @ref RT_ERROR_INVALID_VALUE or @ref RT_ERROR_VARIABLE_NOT_FOUND, respectively.
  *
  * @param[in]   material   Specifies the material to modify
  * @param[in]   v          Specifies the variable to remove
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtMaterialRemoveVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialDeclareVariable,
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialRemoveVariable(RTmaterial material, RTvariable v);

  /**
  * @brief Returns the number of variables attached to a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialGetVariableCount queries the number of variables attached to a
  * material. \a material specifies the material, and should be a value returned by
  * @ref rtMaterialCreate. After the call, if both parameters are valid, the number
  * of variables attached to \a material is returned to \a *count. Otherwise, the
  * call has no effect and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   material   Specifies the material to query
  * @param[out]  count      Returns the number of variables
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtMaterialGetVariableCount was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialGetVariableCount(RTmaterial material, unsigned int* count);

  /**
  * @brief Returns a handle to an indexed variable of a material
  *
  * @ingroup Material
  *
  * <B>Description</B>
  *
  * @ref rtMaterialGetVariable queries the handle of a material's indexed variable.  \a material
  * specifies the target material and should be a value returned by @ref rtMaterialCreate. \a index
  * specifies the index of the variable, and should be a value less than
  * @ref rtMaterialGetVariableCount. If \a material is a valid material and \a index is the index of a
  * variable attached to \a material, \a *v is set to a handle to that variable. Otherwise, \a *v is
  * set to \a NULL and either @ref RT_ERROR_INVALID_VALUE or @ref RT_ERROR_VARIABLE_NOT_FOUND is
  * returned depending on the validity of \a material, or \a index, respectively.
  *
  * @param[in]   material   Specifies the material to query
  * @param[in]   index      Specifies the index of the variable to query
  * @param[out]  v          Returns the indexed variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_VARIABLE_NOT_FOUND
  *
  * <B>History</B>
  *
  * @ref rtMaterialGetVariable was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtMaterialQueryVariable,
  * @ref rtMaterialGetVariableCount,
  * @ref rtMaterialCreate
  *
  */
  RTresult RTAPI rtMaterialGetVariable(RTmaterial material, unsigned int index, RTvariable* v);

/************************************
 **
 **    TextureSampler object
 **
 ***********************************/

  /**
  * @brief Creates a new texture sampler object
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerCreate allocates a texture sampler object.
  * Sets \a *texturesampler to the handle of a newly created texture sampler within \a context.
  * Returns @ref RT_ERROR_INVALID_VALUE if \a texturesampler is \a NULL.
  *
  * @param[in]   context          The context the texture sampler object will be created in
  * @param[out]  texturesampler   The return handle to the new texture sampler object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerCreate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerDestroy
  *
  */
  RTresult RTAPI rtTextureSamplerCreate(RTcontext context, RTtexturesampler* texturesampler);

  /**
  * @brief Structure describing a block of demand loaded memory.
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref \RTmemoryblock describes a one-, two- or three-dimensional block of bytes in memory
  * for a \a mipLevel that are interpreted as elements of \a format.
  *
  * The region is defined by the elements beginning at (x, y, z) and extending to
  * (x + width - 1, y + height - 1, z + depth - 1).  The element size must be taken into account
  * when computing addresses into the memory block based on the size of elements.  There is no
  * padding between elements within a row, e.g. along the x direction.
  *
  * The starting address of the block is given by \a baseAddress and data is stored at addresses
  * increasing from \a baseAddress.  One-dimensional blocks ignore the \a rowPitch and
  * \a planePitch members and are described entirely by the \a baseAddress of the block.  Two
  * dimensional blocks have contiguous bytes in every row, starting with \a baseAddress, but
  * may have gaps between subsequent rows along the height dimension.  The \a rowPitch describes
  * the offset in bytes between subsequent rows within the two-dimensional block.  Similarly,
  * the \a planePitch describes the offset in bytes between subsequent planes within the depth
  * dimension.
  *
  * <B>History</B>
  *
  * @ref RTmemoryblock was introduced in OptiX 6.1
  *
  * <B>See also</B>
  * @ref RTbuffercallback
  * @ref RTtexturesamplercallback
  */
  typedef struct {
    RTformat format;
    void* baseAddress;
    unsigned int mipLevel;
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    unsigned int rowPitch;
    unsigned int planePitch;
  } RTmemoryblock;

  /**
  * @brief Destroys a texture sampler object
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerDestroy removes \a texturesampler from its context and deletes it.
  * \a texturesampler should be a value returned by @ref rtTextureSamplerCreate.
  * After the call, \a texturesampler is no longer a valid handle.
  * Any API object that referenced \a texturesampler will have its reference invalidated.
  *
  * @param[in]   texturesampler   Handle of the texture sampler to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerCreate
  *
  */
  RTresult RTAPI rtTextureSamplerDestroy(RTtexturesampler texturesampler);

  /**
  * @brief Validates the state of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerValidate checks \a texturesampler for completeness.  If \a texturesampler does not have buffers
  * attached to all of its MIP levels and array slices or if the filtering modes are incompatible with the current
  * MIP level and array slice configuration then returns @ref RT_ERROR_INVALID_CONTEXT.
  *
  * @param[in]   texturesampler   The texture sampler to be validated
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextValidate
  *
  */
  RTresult RTAPI rtTextureSamplerValidate(RTtexturesampler texturesampler);

  /**
  * @brief Gets the context object that created this texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetContext returns a handle to the context object that was used to create
  * \a texturesampler.  If \a context is \a NULL, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried for its context
  * @param[out]  context          The return handle for the context object of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtTextureSamplerGetContext(RTtexturesampler texturesampler, RTcontext* context);

  /**
  * Deprecated in OptiX 3.9. Use @ref rtBufferSetMipLevelCount instead.
  *
  */
  RTresult RTAPI rtTextureSamplerSetMipLevelCount(RTtexturesampler texturesampler, unsigned int mipLevelCount);

  /**
  * Deprecated in OptiX 3.9. Use @ref rtBufferGetMipLevelCount instead.
  *
  */
  RTresult RTAPI rtTextureSamplerGetMipLevelCount(RTtexturesampler texturesampler, unsigned int* mipLevelCount);

  /**
  * Deprecated in OptiX 3.9. Use texture samplers with layered buffers instead. See @ref rtBufferCreate.
  *
  */
  RTresult RTAPI rtTextureSamplerSetArraySize(RTtexturesampler texturesampler, unsigned int textureCount);

  /**
  * Deprecated in OptiX 3.9. Use texture samplers with layered buffers instead. See @ref rtBufferCreate.
  *
  */
  RTresult RTAPI rtTextureSamplerGetArraySize(RTtexturesampler texturesampler, unsigned int* textureCount);

  /**
  * @brief Sets the wrapping mode of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetWrapMode sets the wrapping mode of
  * \a texturesampler to \a wrapmode for the texture dimension specified
  * by \a dimension.  \a wrapmode can take one of the following values:
  *
  *  - @ref RT_WRAP_REPEAT
  *  - @ref RT_WRAP_CLAMP_TO_EDGE
  *  - @ref RT_WRAP_MIRROR
  *  - @ref RT_WRAP_CLAMP_TO_BORDER
  *
  * The wrapping mode controls the behavior of the texture sampler as
  * texture coordinates wrap around the range specified by the indexing
  * mode.  These values mirror the CUDA behavior of textures.
  * See CUDA programming guide for details.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   dimension        Dimension of the texture
  * @param[in]   wrapmode         The new wrap mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetWrapMode was introduced in OptiX 1.0.
  * @ref RT_WRAP_MIRROR and @ref RT_WRAP_CLAMP_TO_BORDER were introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetWrapMode
  *
  */
  RTresult RTAPI rtTextureSamplerSetWrapMode(RTtexturesampler texturesampler, unsigned int dimension, RTwrapmode wrapmode);

  /**
  * @brief Gets the wrap mode of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetWrapMode gets the texture wrapping mode of \a texturesampler and stores it in \a *wrapmode.
  * See @ref rtTextureSamplerSetWrapMode for a list of values @ref RTwrapmode can take.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[in]   dimension        Dimension for the wrapping
  * @param[out]  wrapmode         The return handle for the wrap mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetWrapMode was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetWrapMode
  *
  */
  RTresult RTAPI rtTextureSamplerGetWrapMode(RTtexturesampler texturesampler, unsigned int dimension, RTwrapmode* wrapmode);

  /**
  * @brief Sets the filtering modes of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetFilteringModes sets the minification, magnification and MIP mapping filter modes for \a texturesampler.
  * RTfiltermode must be one of the following values:
  *
  *  - @ref RT_FILTER_NEAREST
  *  - @ref RT_FILTER_LINEAR
  *  - @ref RT_FILTER_NONE
  *
  * These filter modes specify how the texture sampler will interpolate
  * buffer data that has been attached to it.  \a minification and
  * \a magnification must be one of @ref RT_FILTER_NEAREST or
  * @ref RT_FILTER_LINEAR.  \a mipmapping may be any of the three values but
  * must be @ref RT_FILTER_NONE if the texture sampler contains only a
  * single MIP level or one of @ref RT_FILTER_NEAREST or @ref RT_FILTER_LINEAR
  * if the texture sampler contains more than one MIP level.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   minification     The new minification filter mode of the texture sampler
  * @param[in]   magnification    The new magnification filter mode of the texture sampler
  * @param[in]   mipmapping       The new MIP mapping filter mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetFilteringModes was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetFilteringModes
  *
  */
  RTresult RTAPI rtTextureSamplerSetFilteringModes(RTtexturesampler texturesampler, RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode mipmapping);

  /**
  * @brief Gets the filtering modes of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetFilteringModes gets the minification, magnification and MIP mapping filtering modes from
  * \a texturesampler and stores them in \a *minification, \a *magnification and \a *mipmapping, respectively.  See
  * @ref rtTextureSamplerSetFilteringModes for the values @ref RTfiltermode may take.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[out]  minification     The return handle for the minification filtering mode of the texture sampler
  * @param[out]  magnification    The return handle for the magnification filtering mode of the texture sampler
  * @param[out]  mipmapping       The return handle for the MIP mapping filtering mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetFilteringModes was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetFilteringModes
  *
  */
  RTresult RTAPI rtTextureSamplerGetFilteringModes(RTtexturesampler texturesampler, RTfiltermode* minification, RTfiltermode* magnification, RTfiltermode* mipmapping);

  /**
  * @brief Sets the maximum anisotropy of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetMaxAnisotropy sets the maximum anisotropy of \a texturesampler to \a value.  A float
  * value specifies the maximum anisotropy ratio to be used when doing anisotropic filtering. This value will be clamped to the range [1,16]
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   value            The new maximum anisotropy level of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetMaxAnisotropy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetMaxAnisotropy
  *
  */
  RTresult RTAPI rtTextureSamplerSetMaxAnisotropy(RTtexturesampler texturesampler, float value);

  /**
  * @brief Gets the maximum anisotropy level for a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetMaxAnisotropy gets the maximum anisotropy level for \a texturesampler and stores
  * it in \a *value.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[out]  value            The return handle for the maximum anisotropy level of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetMaxAnisotropy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetMaxAnisotropy
  *
  */
  RTresult RTAPI rtTextureSamplerGetMaxAnisotropy(RTtexturesampler texturesampler, float* value);

  /**
  * @brief Sets the minimum and the maximum MIP level access range of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetMipLevelClamp sets lower end and the upper end of the MIP level range to clamp access to.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   minLevel         The new minimum mipmap level of the texture sampler
  * @param[in]   maxLevel         The new maximum mipmap level of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetMipLevelClamp was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetMipLevelClamp
  *
  */
  RTresult RTAPI rtTextureSamplerSetMipLevelClamp(RTtexturesampler texturesampler, float minLevel, float maxLevel);

  /**
  * @brief Gets the minimum and the maximum MIP level access range for a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetMipLevelClamp gets the minimum and the maximum MIP level access range for \a texturesampler and stores
  * it in \a *minLevel and \a maxLevel.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[out]  minLevel         The return handle for the minimum mipmap level of the texture sampler
  * @param[out]  maxLevel         The return handle for the maximum mipmap level of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetMipLevelClamp was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetMipLevelClamp
  *
  */
  RTresult RTAPI rtTextureSamplerGetMipLevelClamp(RTtexturesampler texturesampler, float* minLevel, float* maxLevel);

  /**
  * @brief Sets the mipmap offset of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetMipLevelBias sets the offset to be applied to the calculated mipmap level.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   value            The new mipmap offset of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetMipLevelBias was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetMipLevelBias
  *
  */
  RTresult RTAPI rtTextureSamplerSetMipLevelBias(RTtexturesampler texturesampler, float value);

  /**
  * @brief Gets the mipmap offset for a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetMipLevelBias gets the mipmap offset for \a texturesampler and stores
  * it in \a *value.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[out]  value            The return handle for the mipmap offset of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetMipLevelBias was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetMipLevelBias
  *
  */
  RTresult RTAPI rtTextureSamplerGetMipLevelBias(RTtexturesampler texturesampler, float* value);

  /**
  * @brief Sets the read mode of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetReadMode sets the data read mode of \a texturesampler to \a readmode.
  * \a readmode can take one of the following values:
  *
  *  - @ref RT_TEXTURE_READ_ELEMENT_TYPE
  *  - @ref RT_TEXTURE_READ_NORMALIZED_FLOAT
  *  - @ref RT_TEXTURE_READ_ELEMENT_TYPE_SRGB
  *  - @ref RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB
  *
  * @ref RT_TEXTURE_READ_ELEMENT_TYPE_SRGB and @ref RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB were introduced in OptiX 3.9
  * and apply sRGB to linear conversion during texture read for 8-bit integer buffer formats.
  * \a readmode controls the returned value of the texture sampler when it is used to sample
  * textures.  @ref RT_TEXTURE_READ_ELEMENT_TYPE will return data of the type of the underlying
  * buffer objects.  @ref RT_TEXTURE_READ_NORMALIZED_FLOAT will return floating point values
  * normalized by the range of the underlying type.  If the underlying type is floating point,
  * @ref RT_TEXTURE_READ_NORMALIZED_FLOAT and @ref RT_TEXTURE_READ_ELEMENT_TYPE are equivalent,
  * always returning the unmodified floating point value.
  *
  * For example, a texture sampler that samples a buffer of type @ref RT_FORMAT_UNSIGNED_BYTE with
  * a read mode of @ref RT_TEXTURE_READ_NORMALIZED_FLOAT will convert integral values from the
  * range [0,255] to floating point values in the range [0,1] automatically as the buffer is
  * sampled from.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   readmode         The new read mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetReadMode was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetReadMode
  *
  */
  RTresult RTAPI rtTextureSamplerSetReadMode(RTtexturesampler texturesampler, RTtexturereadmode readmode);

  /**
  * @brief Gets the read mode of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetReadMode gets the read mode of \a texturesampler and stores it in \a *readmode.
  * See @ref rtTextureSamplerSetReadMode for a list of values @ref RTtexturereadmode can take.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[out]  readmode         The return handle for the read mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetReadMode was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetReadMode
  *
  */
  RTresult RTAPI rtTextureSamplerGetReadMode(RTtexturesampler texturesampler, RTtexturereadmode* readmode);

  /**
  * @brief Sets whether texture coordinates for this texture sampler are normalized
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetIndexingMode sets the indexing mode of \a texturesampler to \a indexmode.  \a indexmode
  * can take on one of the following values:
  *
  *  - @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,
  *  - @ref RT_TEXTURE_INDEX_ARRAY_INDEX
  *
  * These values are used to control the interpretation of texture coordinates.  If the index mode is set to
  * @ref RT_TEXTURE_INDEX_NORMALIZED_COORDINATES, the texture is parameterized over [0,1].  If the index
  * mode is set to @ref RT_TEXTURE_INDEX_ARRAY_INDEX then texture coordinates are interpreted as array indices
  * into the contents of the underlying buffer objects.
  *
  * @param[in]   texturesampler   The texture sampler object to be changed
  * @param[in]   indexmode        The new indexing mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetIndexingMode was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetIndexingMode
  *
  */
  RTresult RTAPI rtTextureSamplerSetIndexingMode(RTtexturesampler texturesampler, RTtextureindexmode indexmode);

  /**
  * @brief Gets the indexing mode of a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetIndexingMode gets the indexing mode of \a texturesampler and stores it in \a *indexmode.
  * See @ref rtTextureSamplerSetIndexingMode for the values @ref RTtextureindexmode may take.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried
  * @param[out]  indexmode        The return handle for the indexing mode of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetIndexingMode was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetIndexingMode
  *
  */
  RTresult RTAPI rtTextureSamplerGetIndexingMode(RTtexturesampler texturesampler, RTtextureindexmode* indexmode);

  /**
  * @brief Attaches a buffer object to a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerSetBuffer attaches \a buffer to \a texturesampler.
  *
  * @param[in]   texturesampler      The texture sampler object that will contain the buffer
  * @param[in]   deprecated0         Deprecated in OptiX 3.9, must be 0
  * @param[in]   deprecated1         Deprecated in OptiX 3.9, must be 0
  * @param[in]   buffer              The buffer to be attached to the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerSetBuffer was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerGetBuffer
  *
  */
  RTresult RTAPI rtTextureSamplerSetBuffer(RTtexturesampler texturesampler, unsigned int deprecated0, unsigned int deprecated1, RTbuffer buffer);

  /**
  * @brief Gets a buffer object handle from a texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetBuffer gets a buffer object from
  * \a texturesampler and
  * stores it in \a *buffer.
  *
  * @param[in]   texturesampler      The texture sampler object to be queried for the buffer
  * @param[in]   deprecated0         Deprecated in OptiX 3.9, must be 0
  * @param[in]   deprecated1         Deprecated in OptiX 3.9, must be 0
  * @param[out]  buffer              The return handle to the buffer attached to the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetBuffer was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerSetBuffer
  *
  */
  RTresult RTAPI rtTextureSamplerGetBuffer(RTtexturesampler texturesampler, unsigned int deprecated0, unsigned int deprecated1, RTbuffer* buffer);

  /**
  * @brief Returns the texture ID of this texture sampler
  *
  * @ingroup TextureSampler
  *
  * <B>Description</B>
  *
  * @ref rtTextureSamplerGetId returns a handle to the texture sampler
  * \a texturesampler to be used in OptiX programs on the device to
  * reference the associated texture. The returned ID cannot be used on
  * the host side. If \a textureId is \a NULL, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   texturesampler   The texture sampler object to be queried for its ID
  * @param[out]  textureId        The returned device-side texture ID of the texture sampler
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtTextureSamplerGetId was introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtTextureSamplerCreate
  *
  */
  RTresult RTAPI rtTextureSamplerGetId(RTtexturesampler texturesampler, int* textureId);

/************************************
 **
 **    Buffer object
 **
 ***********************************/

  /**
  * @brief Creates a new buffer object
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferCreate allocates and returns a new handle to a new buffer object in \a *buffer associated
  * with \a context. The backing storage of the buffer is managed by OptiX. A buffer is specified by a bitwise
  * \a or combination of a \a type and \a flags in \a bufferdesc. The supported types are:
  *
  * -  @ref RT_BUFFER_INPUT
  * -  @ref RT_BUFFER_OUTPUT
  * -  @ref RT_BUFFER_INPUT_OUTPUT
  * -  @ref RT_BUFFER_PROGRESSIVE_STREAM
  *
  * The type values are used to specify the direction of data flow from the host to the OptiX devices.
  * @ref RT_BUFFER_INPUT specifies that the host may only write to the buffer and the device may only read from the buffer.
  * @ref RT_BUFFER_OUTPUT specifies the opposite, read only access on the host and write only access on the device.
  * Devices and the host may read and write from buffers of type @ref RT_BUFFER_INPUT_OUTPUT.  Reading or writing to
  * a buffer of the incorrect type (e.g., the host writing to a buffer of type @ref RT_BUFFER_OUTPUT) is undefined.
  * @ref RT_BUFFER_PROGRESSIVE_STREAM is used to receive stream updates generated by progressive launches (see @ref rtContextLaunchProgressive2D).
  *
  * The supported flags are:
  *
  * -  @ref RT_BUFFER_GPU_LOCAL
  * -  @ref RT_BUFFER_COPY_ON_DIRTY
  * -  @ref RT_BUFFER_LAYERED
  * -  @ref RT_BUFFER_CUBEMAP
  * -  @ref RT_BUFFER_DISCARD_HOST_MEMORY
  *
  * If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
  * If RT_BUFFER_CUBEMAP flag is set, buffer depth specifies the number of cube faces, not the depth of a 3D buffer.
  * See details in @ref rtBufferSetSize3D
  *
  * Flags can be used to optimize data transfers between the host and its devices. The flag @ref RT_BUFFER_GPU_LOCAL can only be
  * used in combination with @ref RT_BUFFER_INPUT_OUTPUT. @ref RT_BUFFER_INPUT_OUTPUT and @ref RT_BUFFER_GPU_LOCAL used together specify a buffer
  * that allows the host to \a only write, and the device to read \a and write data. The written data will never be visible
  * on the host side and will generally not be visible on other devices.
  *
  * If @ref rtBufferGetDevicePointer has been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is created with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  *
  * The flag @ref RT_BUFFER_DISCARD_HOST_MEMORY can only be used in combination with @ref RT_BUFFER_INPUT. The data will be
  * synchronized to the devices as soon as the buffer is unmapped from the host using @ref rtBufferUnmap or
  * @ref rtBufferUnmapEx and the memory allocated on the host will be deallocated.
  * It is preferred to map buffers created with the @ref RT_BUFFER_DISCARD_HOST_MEMORY using @ref rtBufferMapEx with the
  * @ref RT_BUFFER_MAP_WRITE_DISCARD option enabled. If it is mapped using @ref rtBufferMap or the @ref RT_BUFFER_MAP_WRITE
  * option instead, the data needs to be synchronized to the host during mapping.
  * Note that the data that is allocated on the devices will not be deallocated until the buffer is destroyed.
  *
  * Returns @ref RT_ERROR_INVALID_VALUE if \a buffer is \a NULL.
  *
  * @param[in]   context      The context to create the buffer in
  * @param[in]   bufferdesc   Bitwise \a or combination of the \a type and \a flags of the new buffer
  * @param[out]  buffer       The return handle for the buffer object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferCreate was introduced in OptiX 1.0.
  *
  * @ref RT_BUFFER_GPU_LOCAL was introduced in OptiX 2.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreateFromGLBO,
  * @ref rtBufferDestroy,
  * @ref rtBufferMarkDirty
  * @ref rtBufferBindProgressiveStream
  *
  */
  RTresult RTAPI rtBufferCreate(RTcontext context, unsigned int bufferdesc, RTbuffer* buffer);

  /**
  * @brief Callback function used to demand load data for a buffer.
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref RTbuffercallback is implemented by the application.  It is invoked by OptiX for each
  * \a requestedPage of the demand loaded \a buffer referenced by the previous launch that was not
  * resident in device memory.  The callback should either fill the provided \a block buffer with
  * the requested \a pageDataSizeInBytes of data and return \a true, or return \a false.  When the
  * callback returns \a false, no data is transferred to the \a buffer.
  *
  * <b>CAUTION</b>: OptiX will invoke callback functions from multiple threads in order to satisfy
  * pending requests in parallel.  A user provided callback function should not allow exceptions to
  * escape from their callback function.
  *
  * @param[in]    callbackData  An arbitrary data pointer from the application when the callback was registered.
  * @param[in]    buffer        Handle of the buffer requesting pages.
  * @param[in]    block         A pointer to the @ref RTmemoryblock describing the memory to be filled with data.
  *
  * <B>Return values</B>
  *
  * \a non-zero   The \a block buffer was filled with \a pageDataSizeInBytes of data.
  * \a zero       No data was written.  No data will be transferred to the \a buffer.
  *               The same \a block may be passed to the callback again after the next launch.
  *
  * <B>History</B>
  *
  * @ref RTbuffercallback was introduced in OptiX 6.1
  *
  * <B>See also</B>
  * @ref RTmemoryblock
  * @ref rtBufferCreateFromCallback
  */
  typedef int (*RTbuffercallback)(void* callbackData, RTbuffer buffer, RTmemoryblock* block);

  /**
  * @brief Creates a buffer whose contents are loaded on demand.
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferCreateFromCallback allocates and returns a new handle to a new buffer object in \a *buffer associated
  * with \a context.  The backing storage of the buffer is managed by OptiX, but is filled on demand by the application.
  * The backing storage is allocated in multiples of pages.  Each page is a uniform size as described by the
  * \a RT_BUFFER_ATTRIBUTE_PAGE_SIZE attribute.  The backing storage may be smaller than the total size of storage needed
  * for the buffer, with OptiX managing the storage in conjunction with the application supplied \a callback.  A buffer
  * is specified by a bitwise \a or combination of a \a type and \a flags in \a bufferdesc.  The only supported type is
  * @ref RT_BUFFER_INPUT as only input buffers can be demand loaded.
  *
  * The supported flags are:
  *
  * -  @ref RT_BUFFER_LAYERED
  * -  @ref RT_BUFFER_CUBEMAP
  *
  * If RT_BUFFER_LAYERED flag is set, buffer depth specifies the number of layers, not the depth of a 3D buffer.
  * If RT_BUFFER_CUBEMAP flag is set, buffer depth specifies the number of cube faces, not the depth of a 3D buffer.
  * See details in @ref rtBufferSetSize3D
  *
  * It is an error to call @ref rtBufferGetDevicePointer, @ref rtBufferMap or @ref rtBufferUnmap for a demand loaded buffer.
  *
  * Returns @ref RT_ERROR_INVALID_VALUE if either \a callback or \a buffer is \a NULL.
  *
  * @param[in]   context      The context to create the buffer in.
  * @param[in]   bufferdesc   Bitwise \a or combination of the \a type and \a flags of the new buffer.
  * @param[in]   callback     The demand load callback.  Most not be NULL.
  * @param[in]   callbackData An arbitrary pointer from the application that is passed to the callback.  This may be \a NULL.
  * @param[out]  buffer       The return handle for the buffer object.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferCreateFromCallback was introduced in OptiX 6.1
  *
  * <B>See also</B>
  * @ref RTbuffercallback
  * @ref rtBufferDestroy
  *
  */
  RTresult RTAPI rtBufferCreateFromCallback(RTcontext context, unsigned int bufferdesc, RTbuffercallback callback, void* callbackData, RTbuffer* buffer);

  /**
  * @brief Destroys a buffer object
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferDestroy removes \a buffer from its context and deletes it.
  * \a buffer should be a value returned by @ref rtBufferCreate.
  * After the call, \a buffer is no longer a valid handle.
  * Any API object that referenced \a buffer will have its reference invalidated.
  *
  * @param[in]   buffer   Handle of the buffer to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferDestroy was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferCreateFromGLBO
  *
  */
  RTresult RTAPI rtBufferDestroy(RTbuffer buffer);

  /**
  * @brief Validates the state of a buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferValidate checks \a buffer for completeness.  If \a buffer has not had its dimensionality, size or format
  * set, this call will return @ref RT_ERROR_INVALID_CONTEXT.
  *
  * @param[in]   buffer   The buffer to validate
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferValidate was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferCreateFromGLBO
  * @ref rtContextValidate
  *
  */
  RTresult RTAPI rtBufferValidate(RTbuffer buffer);

  /**
  * @brief Returns the context object that created this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetContext returns a handle to the context that created \a buffer in \a *context.
  * If \a *context is \a NULL, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   buffer    The buffer to be queried for its context
  * @param[out]  context   The return handle for the buffer's context
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetContext was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtContextCreate
  *
  */
  RTresult RTAPI rtBufferGetContext(RTbuffer buffer, RTcontext* context);

  /**
  * @brief Sets the format of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetFormat changes the \a format of \a buffer to the specified value.
  * The data elements of the buffer will have the specified type and can either be
  * vector formats, or a user-defined type whose size is specified with
  * @ref rtBufferSetElementSize. Possible values for \a format are:
  *
  *   - @ref RT_FORMAT_HALF
  *   - @ref RT_FORMAT_HALF2
  *   - @ref RT_FORMAT_HALF3
  *   - @ref RT_FORMAT_HALF4
  *   - @ref RT_FORMAT_FLOAT
  *   - @ref RT_FORMAT_FLOAT2
  *   - @ref RT_FORMAT_FLOAT3
  *   - @ref RT_FORMAT_FLOAT4
  *   - @ref RT_FORMAT_BYTE
  *   - @ref RT_FORMAT_BYTE2
  *   - @ref RT_FORMAT_BYTE3
  *   - @ref RT_FORMAT_BYTE4
  *   - @ref RT_FORMAT_UNSIGNED_BYTE
  *   - @ref RT_FORMAT_UNSIGNED_BYTE2
  *   - @ref RT_FORMAT_UNSIGNED_BYTE3
  *   - @ref RT_FORMAT_UNSIGNED_BYTE4
  *   - @ref RT_FORMAT_SHORT
  *   - @ref RT_FORMAT_SHORT2
  *   - @ref RT_FORMAT_SHORT3
  *   - @ref RT_FORMAT_SHORT4
  *   - @ref RT_FORMAT_UNSIGNED_SHORT
  *   - @ref RT_FORMAT_UNSIGNED_SHORT2
  *   - @ref RT_FORMAT_UNSIGNED_SHORT3
  *   - @ref RT_FORMAT_UNSIGNED_SHORT4
  *   - @ref RT_FORMAT_INT
  *   - @ref RT_FORMAT_INT2
  *   - @ref RT_FORMAT_INT3
  *   - @ref RT_FORMAT_INT4
  *   - @ref RT_FORMAT_UNSIGNED_INT
  *   - @ref RT_FORMAT_UNSIGNED_INT2
  *   - @ref RT_FORMAT_UNSIGNED_INT3
  *   - @ref RT_FORMAT_UNSIGNED_INT4
  *   - @ref RT_FORMAT_LONG_LONG
  *   - @ref RT_FORMAT_LONG_LONG2
  *   - @ref RT_FORMAT_LONG_LONG3
  *   - @ref RT_FORMAT_LONG_LONG4
  *   - @ref RT_FORMAT_UNSIGNED_LONG_LONG
  *   - @ref RT_FORMAT_UNSIGNED_LONG_LONG2
  *   - @ref RT_FORMAT_UNSIGNED_LONG_LONG3
  *   - @ref RT_FORMAT_UNSIGNED_LONG_LONG4
  *   - @ref RT_FORMAT_UNSIGNED_BC1
  *   - @ref RT_FORMAT_UNSIGNED_BC2
  *   - @ref RT_FORMAT_UNSIGNED_BC3
  *   - @ref RT_FORMAT_UNSIGNED_BC4
  *   - @ref RT_FORMAT_BC4
  *   - @ref RT_FORMAT_UNSIGNED_BC5
  *   - @ref RT_FORMAT_BC5
  *   - @ref RT_FORMAT_UNSIGNED_BC6H
  *   - @ref RT_FORMAT_BC6H
  *   - @ref RT_FORMAT_UNSIGNED_BC7
  *   - @ref RT_FORMAT_USER
  *
  * Buffers of block-compressed formats like @ref RT_FORMAT_BC6H must be sized
  * to a quarter of the uncompressed view resolution in each dimension, i.e.
  * @code rtBufferSetSize2D( buffer, width/4, height/4 ); @endcode
  * The base type of the internal buffer will then correspond to @ref RT_FORMAT_UNSIGNED_INT2
  * for BC1 and BC4 formats and @ref RT_FORMAT_UNSIGNED_INT4 for all other BC formats.
  *
  * @param[in]   buffer   The buffer to have its format set
  * @param[in]   format   The target format of the buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferSetFormat was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetFormat,
  * @ref rtBufferGetFormat,
  * @ref rtBufferGetFormat,
  * @ref rtBufferGetElementSize,
  * @ref rtBufferSetElementSize
  *
  */
  RTresult RTAPI rtBufferSetFormat(RTbuffer buffer, RTformat format);

  /**
  * @brief Gets the format of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetFormat returns, in \a *format, the format of \a buffer.  See @ref rtBufferSetFormat for a listing
  * of @ref RTbuffer values.
  *
  * @param[in]   buffer   The buffer to be queried for its format
  * @param[out]  format   The return handle for the buffer's format
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetFormat was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetFormat,
  * @ref rtBufferGetFormat
  *
  */
  RTresult RTAPI rtBufferGetFormat(RTbuffer buffer, RTformat* format);

  /**
  * @brief Modifies the size in bytes of a buffer's individual elements
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetElementSize modifies the size in bytes of a buffer's user-formatted
  * elements. The target buffer is specified by \a buffer, which should be a
  * value returned by @ref rtBufferCreate and should have format @ref RT_FORMAT_USER.
  * The new size of the buffer's individual elements is specified by
  * \a elementSize and should not be 0. If the buffer has
  * format @ref RT_FORMAT_USER, and \a elementSize is not 0, then the buffer's individual
  * element size is set to \a elemenSize and all storage associated with the buffer is reset.
  * Otherwise, this call has no effect and returns either @ref RT_ERROR_TYPE_MISMATCH if
  * the buffer does not have format @ref RT_FORMAT_USER or @ref RT_ERROR_INVALID_VALUE if the
  * buffer has format @ref RT_FORMAT_USER but \a elemenSize is 0.
  *
  * @param[in]   buffer            Specifies the buffer to be modified
  * @param[in]   elementSize       Specifies the new size in bytes of the buffer's individual elements
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_TYPE_MISMATCH
  *
  * <B>History</B>
  *
  * @ref rtBufferSetElementSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferGetElementSize,
  * @ref rtBufferCreate
  *
  */
  RTresult RTAPI rtBufferSetElementSize(RTbuffer buffer, RTsize elementSize);

  /**
  * @brief Returns the size of a buffer's individual elements
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetElementSize queries the size of a buffer's elements. The target buffer
  * is specified by \a buffer, which should be a value returned by
  * @ref rtBufferCreate. The size, in bytes, of the buffer's
  * individual elements is returned in \a *elementSize.
  * Returns @ref RT_ERROR_INVALID_VALUE if given a \a NULL pointer.
  *
  * @param[in]   buffer                Specifies the buffer to be queried
  * @param[out]  elementSize           Returns the size of the buffer's individual elements
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_UNKNOWN
  *
  * <B>History</B>
  *
  * @ref rtBufferGetElementSize was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetElementSize,
  * @ref rtBufferCreate
  *
  */
  RTresult RTAPI rtBufferGetElementSize(RTbuffer buffer, RTsize* elementSize);

  /**
  * @brief Sets the width and dimensionality of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetSize1D sets the dimensionality of \a buffer to 1 and sets its width to
  * \a width.
  * Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
  *
  * @param[in]   buffer   The buffer to be resized
  * @param[in]   width    The width of the resized buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferSetSize1D was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferSetSize1D(RTbuffer buffer, RTsize width);

  /**
  * @brief Get the width of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetSize1D stores the width of \a buffer in \a *width.
  *
  * @param[in]   buffer   The buffer to be queried for its dimensions
  * @param[out]  width    The return handle for the buffer's width
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetSize1D was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetSize1D(RTbuffer buffer, RTsize* width);

  /**
  * @brief Sets the width, height and dimensionality of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetSize2D sets the dimensionality of \a buffer to 2 and sets its width
  * and height to \a width and \a height, respectively.  If \a width or \a height is
  * zero, they both must be zero.
  * Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
  *
  * @param[in]   buffer   The buffer to be resized
  * @param[in]   width    The width of the resized buffer
  * @param[in]   height   The height of the resized buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferSetSize2D was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferSetSize2D(RTbuffer buffer, RTsize width, RTsize height);

  /**
  * @brief Gets the width and height of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetSize2D stores the width and height of \a buffer in \a *width and
  * \a *height, respectively.
  *
  * @param[in]   buffer   The buffer to be queried for its dimensions
  * @param[out]  width    The return handle for the buffer's width
  * @param[out]  height   The return handle for the buffer's height
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetSize2D was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetSize2D(RTbuffer buffer, RTsize* width, RTsize* height);

  /**
  * @brief Sets the width, height, depth and dimensionality of a buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetSize3D sets the dimensionality of \a buffer to 3 and sets its width,
  * height and depth to \a width, \a height and \a depth, respectively.  If \a width,
  * \a height or \a depth is zero, they all must be zero.
  *
  * A 1D layered mipmapped buffer is allocated if \a height is 1 and the @ref RT_BUFFER_LAYERED flag was set at buffer creating. The number of layers is determined by the \a depth.
  * A 2D layered mipmapped buffer is allocated if the @ref RT_BUFFER_LAYERED flag was set at buffer creating. The number of layers is determined by the \a depth.
  * A cubemap mipmapped buffer is allocated if the @ref RT_BUFFER_CUBEMAP flag was set at buffer creating. \a width must be equal to \a height and the number of cube faces is determined by the \a depth,
  * it must be six or a multiple of six, if the @ref RT_BUFFER_LAYERED flag was also set.
  * Layered, mipmapped and cubemap buffers are supported only as texture buffers.
  *
  * Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
  *
  * @param[in]   buffer   The buffer to be resized
  * @param[in]   width    The width of the resized buffer
  * @param[in]   height   The height of the resized buffer
  * @param[in]   depth    The depth of the resized buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferSetSize3D was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferSetSize3D(RTbuffer buffer, RTsize width, RTsize height, RTsize depth);

  /**
  * @brief Sets the MIP level count of a buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetMipLevelCount sets the number of MIP levels to \a levels. The default number of MIP levels is 1.
  * Fails with @ref RT_ERROR_ALREADY_MAPPED if called on a buffer that is mapped.
  *
  * @param[in]   buffer   The buffer to be resized
  * @param[in]   levels   Number of mip levels
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferSetMipLevelCount was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferSetMipLevelCount(RTbuffer buffer, unsigned int levels);


  /**
  * @brief Gets the width, height and depth of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetSize3D stores the width, height and depth of \a buffer in \a *width,
  * \a *height and \a *depth, respectively.
  *
  * @param[in]   buffer   The buffer to be queried for its dimensions
  * @param[out]  width    The return handle for the buffer's width
  * @param[out]  height   The return handle for the buffer's height
  * @param[out]  depth    The return handle for the buffer's depth
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetSize3D was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetSize3D(RTbuffer buffer, RTsize* width, RTsize* height, RTsize* depth);

  /**
  * @brief Gets the width of buffer specific MIP level
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetMipLevelSize1D stores the width of \a buffer in \a *width.
  *
  * @param[in]   buffer   The buffer to be queried for its dimensions
  * @param[in]   level    The buffer MIP level index to be queried for its dimensions
  * @param[out]  width    The return handle for the buffer's width
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferGetMipLevelSize1D was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetMipLevelSize1D(RTbuffer buffer, unsigned int level, RTsize* width);

  /**
  * @brief Gets the width, height of buffer specific MIP level
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetMipLevelSize2D stores the width, height of \a buffer in \a *width and
  * \a *height respectively.
  *
  * @param[in]   buffer   The buffer to be queried for its dimensions
  * @param[in]   level    The buffer MIP level index to be queried for its dimensions
  * @param[out]  width    The return handle for the buffer's width
  * @param[out]  height   The return handle for the buffer's height
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetMipLevelSize2D was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetMipLevelSize2D(RTbuffer buffer, unsigned int level, RTsize* width, RTsize* height);


  /**
  * @brief Gets the width, height and depth of buffer specific MIP level
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetMipLevelSize3D stores the width, height and depth of \a buffer in \a *width,
  * \a *height and \a *depth, respectively.
  *
  * @param[in]   buffer   The buffer to be queried for its dimensions
  * @param[in]   level    The buffer MIP level index to be queried for its dimensions
  * @param[out]  width    The return handle for the buffer's width
  * @param[out]  height   The return handle for the buffer's height
  * @param[out]  depth    The return handle for the buffer's depth
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferGetMipLevelSize3D was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetMipLevelSize3D(RTbuffer buffer, unsigned int level, RTsize* width, RTsize* height, RTsize* depth);


  /**
  * @brief Sets the dimensionality and dimensions of a buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetSizev sets the dimensionality of \a buffer to \a dimensionality and
  * sets the dimensions of the buffer to the values stored at *\a dims, which must contain
  * a number of values equal to \a dimensionality.  If any of values of \a dims is zero
  * they must all be zero.
  *
  * @param[in]   buffer           The buffer to be resized
  * @param[in]   dimensionality   The dimensionality the buffer will be resized to
  * @param[in]   dims             The array of sizes for the dimension of the resize
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferSetSizev was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferSetSizev(RTbuffer buffer, unsigned int dimensionality, const RTsize* dims);

  /**
  * @brief Gets the dimensions of this buffer
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetSizev stores the dimensions of \a buffer in \a *dims.  The number of
  * dimensions returned is specified by \a dimensionality.  The storage at \a dims must be
  * large enough to hold the number of requested buffer dimensions.
  *
  * @param[in]   buffer           The buffer to be queried for its dimensions
  * @param[in]   dimensionality   The number of requested dimensions
  * @param[out]  dims             The array of dimensions to store to
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetSizev was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetMipLevelCount,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D
  *
  */
  RTresult RTAPI rtBufferGetSizev(RTbuffer buffer, unsigned int dimensionality, RTsize* dims);

  /**
  * @brief Gets the dimensionality of this buffer object
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetDimensionality returns the dimensionality of \a buffer in \a
  * *dimensionality.  The value returned will be one of 1, 2 or 3, corresponding to 1D, 2D
  * and 3D buffers, respectively.
  *
  * @param[in]   buffer           The buffer to be queried for its dimensionality
  * @param[out]  dimensionality   The return handle for the buffer's dimensionality
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetDimensionality was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * \a rtBufferSetSize{1-2-3}D
  *
  */
  RTresult RTAPI rtBufferGetDimensionality(RTbuffer buffer, unsigned int* dimensionality);

  /**
  * @brief Gets the number of mipmap levels of this buffer object
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetMipLevelCount returns the number of mipmap levels. Default number of MIP levels is 1.
  *
  * @param[in]   buffer           The buffer to be queried for its number of mipmap levels
  * @param[out]  level            The return number of mipmap levels
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferGetMipLevelCount was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferSetMipLevelCount,
  * @ref rtBufferSetSize1D,
  * @ref rtBufferSetSize2D,
  * @ref rtBufferSetSize3D,
  * @ref rtBufferSetSizev,
  * @ref rtBufferGetMipLevelSize1D,
  * @ref rtBufferGetMipLevelSize2D,
  * @ref rtBufferGetMipLevelSize3D,
  * @ref rtBufferGetSize1D,
  * @ref rtBufferGetSize2D,
  * @ref rtBufferGetSize3D,
  * @ref rtBufferGetSizev
  *
  */
  RTresult RTAPI rtBufferGetMipLevelCount(RTbuffer buffer, unsigned int* level);

  /**
  * @brief Maps a buffer object to the host
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferMap returns a pointer, accessible by the host, in \a *userPointer that
  * contains a mapped copy of the contents of \a buffer.  The memory pointed to by \a *userPointer
  * can be written to or read from, depending on the type of \a buffer.  For
  * example, this code snippet demonstrates creating and filling an input buffer with
  * floats.
  *
  *@code
  *  RTbuffer buffer;
  *  float* data;
  *  rtBufferCreate(context, RT_BUFFER_INPUT, &buffer);
  *  rtBufferSetFormat(buffer, RT_FORMAT_FLOAT);
  *  rtBufferSetSize1D(buffer, 10);
  *  rtBufferMap(buffer, (void*)&data);
  *  for(int i = 0; i < 10; ++i)
  *    data[i] = 4.f * i;
  *  rtBufferUnmap(buffer);
  *@endcode
  * If \a buffer has already been mapped, returns @ref RT_ERROR_ALREADY_MAPPED.
  * If \a buffer has size zero, the returned pointer is undefined.
  *
  * Note that this call does not stop a progressive render if called on a stream buffer.
  *
  * @param[in]   buffer         The buffer to be mapped
  * @param[out]  userPointer    Return handle to a user pointer where the buffer will be mapped to
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferMap was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferUnmap,
  * @ref rtBufferMapEx,
  * @ref rtBufferUnmapEx
  *
  */
  RTresult RTAPI rtBufferMap(RTbuffer buffer, void** userPointer);

  /**
  * @brief Unmaps a buffer's storage from the host
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferUnmap unmaps a buffer from the host after a call to @ref rtBufferMap.  @ref rtContextLaunch "rtContextLaunch" cannot be called
  * while buffers are still mapped to the host.  A call to @ref rtBufferUnmap that does not follow a matching @ref rtBufferMap
  * call will return @ref RT_ERROR_INVALID_VALUE.
  *
  * Note that this call does not stop a progressive render if called with a stream buffer.
  *
  * @param[in]   buffer   The buffer to unmap
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferUnmap was introduced in OptiX 1.0.
  *
  * <B>See also</B>
  * @ref rtBufferMap,
  * @ref rtBufferMapEx,
  * @ref rtBufferUnmapEx
  *
  */
  RTresult RTAPI rtBufferUnmap(RTbuffer buffer);

  /**
  * @brief Maps mipmap level of buffer object to the host
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferMapEx makes the buffer contents available on the host, either by returning a pointer in \a *optixOwned, or by copying the contents
  * to a memory location pointed to by \a userOwned. Calling @ref rtBufferMapEx with proper map flags can result in better performance than using @ref rtBufferMap, because
  * fewer synchronization copies are required in certain situations.
  * @ref rtBufferMapEx with \a mapFlags = @ref RT_BUFFER_MAP_READ_WRITE and \a level = 0 is equivalent to @ref rtBufferMap.
  *
  * Note that this call does not stop a progressive render if called on a stream buffer.
  *
  * @param[in]   buffer         The buffer to be mapped
  * @param[in]   mapFlags       Map flags, see below
  * @param[in]   level          The mipmap level to be mapped
  * @param[in]   userOwned      Not yet supported. Must be NULL
  * @param[out]  optixOwned     Return handle to a user pointer where the buffer will be mapped to
  *
  * The following flags are supported for mapFlags. They are mutually exclusive:
  *
  * -  @ref RT_BUFFER_MAP_READ
  * -  @ref RT_BUFFER_MAP_WRITE
  * -  @ref RT_BUFFER_MAP_READ_WRITE
  * -  @ref RT_BUFFER_MAP_WRITE_DISCARD
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ALREADY_MAPPED
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferMapEx was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferMap,
  * @ref rtBufferUnmap,
  * @ref rtBufferUnmapEx
  *
  */
  RTresult RTAPI rtBufferMapEx(RTbuffer buffer, unsigned int mapFlags, unsigned int level, void* userOwned, void** optixOwned);

  /**
  * @brief Unmaps mipmap level storage from the host
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferUnmapEx unmaps buffer level from the host after a call to @ref rtBufferMapEx.  @ref rtContextLaunch "rtContextLaunch" cannot be called
  * while buffers are still mapped to the host.  A call to @ref rtBufferUnmapEx that does not follow a matching @ref rtBufferMapEx
  * call will return @ref RT_ERROR_INVALID_VALUE. @ref rtBufferUnmap is equivalent to @ref rtBufferUnmapEx with \a level = 0.
  *
  * Note that this call does not stop a progressive render if called with a stream buffer.
  *
  * @param[in]   buffer   The buffer to unmap
  * @param[in]   level    The mipmap level to unmap
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferUnmapEx was introduced in OptiX 3.9.
  *
  * <B>See also</B>
  * @ref rtBufferMap,
  * @ref rtBufferUnmap,
  * @ref rtBufferMapEx
  *
  */
  RTresult RTAPI rtBufferUnmapEx(RTbuffer buffer, unsigned int level);

  /**
  * @brief Gets an id suitable for use with buffers of buffers
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetId returns an ID for the provided buffer.  The returned ID is used on
  * the device to reference the buffer.  It needs to be copied into a buffer of type @ref
  * RT_FORMAT_BUFFER_ID or used in a @ref rtBufferId object.. If \a *bufferId is \a NULL
  * or the \a buffer is not a valid RTbuffer, returns @ref
  * RT_ERROR_INVALID_VALUE.  @ref RT_BUFFER_ID_NULL can be used as a sentinel for a
  * non-existent buffer, since this value will never be returned as a valid buffer id.
  *
  * @param[in]   buffer      The buffer to be queried for its id
  * @param[out]  bufferId    The returned ID of the buffer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferGetId was introduced in OptiX 3.5.
  *
  * <B>See also</B>
  * @ref rtContextGetBufferFromId
  *
  */
  RTresult RTAPI rtBufferGetId(RTbuffer buffer, int* bufferId);

  /**
  * @brief Gets an RTbuffer corresponding to the buffer id
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtContextGetBufferFromId returns a handle to the buffer in \a *buffer corresponding to
  * the \a bufferId supplied.  If \a bufferId does not map to a valid buffer handle,
  * \a *buffer is \a NULL or if \a context is invalid, returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   context     The context the buffer should be originated from
  * @param[in]   bufferId    The ID of the buffer to query
  * @param[out]  buffer      The return handle for the buffer object corresponding to the bufferId
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtContextGetBufferFromId was introduced in OptiX 3.5.
  *
  * <B>See also</B>
  * @ref rtBufferGetId
  *
  */
  RTresult RTAPI rtContextGetBufferFromId(RTcontext context, int bufferId, RTbuffer* buffer);

  /**
  * @brief Check whether stream buffer content has been updated by a Progressive Launch
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * Returns whether or not the result of a progressive launch in \a buffer has been updated
  * since the last time this function was called. A client application should use this call in its
  * main render/display loop to poll for frame refreshes after initiating a progressive launch. If \a subframeCount and
  * \a maxSubframes are non-null, they will be filled with the corresponding counters if and
  * only if \a ready returns 1.
  *
  * Note that this call does not stop a progressive render.
  *
  * @param[in]   buffer             The stream buffer to be queried
  * @param[out]  ready              Ready flag. Will be set to 1 if an update is available, or 0 if no update is available.
  * @param[out]  subframeCount      The number of subframes accumulated in the latest result
  * @param[out]  maxSubframes       The \a maxSubframes parameter as specified in the call to @ref rtContextLaunchProgressive2D
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferGetProgressiveUpdateReady was introduced in OptiX 3.8.
  *
  * <B>See also</B>
  * @ref rtContextLaunchProgressive2D
  *
  */
  RTresult RTAPI rtBufferGetProgressiveUpdateReady(RTbuffer buffer, int* ready, unsigned int* subframeCount, unsigned int* maxSubframes);

  /**
  * @brief Bind a stream buffer to an output buffer source
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * Binds an output buffer to a progressive stream. The output buffer thereby becomes the
  * data source for the stream. To form a valid output/stream pair, the stream buffer must be
  * of format @ref RT_FORMAT_UNSIGNED_BYTE4, and the output buffer must be of format @ref RT_FORMAT_FLOAT3 or @ref RT_FORMAT_FLOAT4.
  * The use of @ref RT_FORMAT_FLOAT4 is recommended for performance reasons, even if the fourth component is unused.
  * The output buffer must be of type @ref RT_BUFFER_OUTPUT; it may not be of type @ref RT_BUFFER_INPUT_OUTPUT.
  *
  * @param[in]   stream             The stream buffer for which the source is to be specified
  * @param[in]   source             The output buffer to function as the stream's source
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferBindProgressiveStream was introduced in OptiX 3.8.
  *
  * <B>See also</B>
  * @ref rtBufferCreate
  * @ref rtBufferSetAttribute
  * @ref rtBufferGetAttribute
  *
  */
  RTresult RTAPI rtBufferBindProgressiveStream(RTbuffer stream, RTbuffer source);

  /**
  * @brief Set a buffer attribute
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * Sets a buffer attribute. Currently, all available attributes refer to stream buffers only,
  * and attempting to set them on a non-stream buffer will generate an error.
  *
  * Each attribute can have a different size.  The sizes are given in the following list:
  *
  *   - @ref RT_BUFFER_ATTRIBUTE_STREAM_FORMAT      strlen(input_string)
  *   - @ref RT_BUFFER_ATTRIBUTE_STREAM_BITRATE     sizeof(int)
  *   - @ref RT_BUFFER_ATTRIBUTE_STREAM_FPS         sizeof(int)
  *   - @ref RT_BUFFER_ATTRIBUTE_STREAM_GAMMA       sizeof(float)
  *
  * @ref RT_BUFFER_ATTRIBUTE_STREAM_FORMAT sets the encoding format used for streams sent over the network, specified as a string.
  * The default is "auto". Various other common stream and image formats are available (e.g. "h264", "png"). This
  * attribute has no effect if the progressive API is used locally.
  *
  * @ref RT_BUFFER_ATTRIBUTE_STREAM_BITRATE sets the target bitrate for streams sent over the network, if the stream format supports
  * it. The data is specified as a 32-bit integer. The default is 5000000. This attribute has no
  * effect if the progressive API is used locally or if the stream format does not support
  * variable bitrates.
  *
  * @ref RT_BUFFER_ATTRIBUTE_STREAM_FPS sets the target update rate per second for streams sent over the network, if the stream
  * format supports it. The data is specified as a 32-bit integer. The default is 30. This
  * attribute has no effect if the progressive API is used locally or if the stream format does
  * not support variable framerates.
  *
  * @ref RT_BUFFER_ATTRIBUTE_STREAM_GAMMA sets the gamma value for the built-in tonemapping operator. The data is specified as a
  * 32-bit float, the default is 1.0. Tonemapping is executed before encoding the
  * accumulated output into the stream, i.e. on the server side if remote rendering is used.
  * See the section on Buffers below for more details.
  *
  * @param[in]   buffer             The buffer on which to set the attribute
  * @param[in]   attrib             The attribute to set
  * @param[in]   size               The size of the attribute value, in bytes
  * @param[in]   p                  Pointer to the attribute value
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferSetAttribute was introduced in OptiX 3.8.
  *
  * <B>See also</B>
  * @ref rtBufferGetAttribute
  *
  */
  RTresult RTAPI rtBufferSetAttribute(RTbuffer buffer, RTbufferattribute attrib, RTsize size, const void* p);

  /**
  * @brief Query a buffer attribute
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetAttribute is used to query buffer attributes. For a list of available attributes that can be set, please refer to @ref rtBufferSetAttribute.
  * The attribute \a RT_BUFFER_ATTRIBUTE_PAGE_SIZE can only be queried and returns the page size of a demand loaded buffer in bytes.  The size of the data returned
  * for this attribute is \a sizeof(int).
  *
  * @param[in]   buffer             The buffer to query the attribute from
  * @param[in]   attrib             The attribute to query
  * @param[in]   size               The size of the attribute value, in bytes. For string attributes, this is the maximum buffer size the returned string will use (including a terminating null character).
  * @param[out]  p                  Pointer to the attribute value to be filled in. Must point to valid memory of at least \a size bytes.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferGetAttribute was introduced in OptiX 3.8.
  *
  * <B>See also</B>
  * @ref rtBufferSetAttribute
  *
  */
  RTresult RTAPI rtBufferGetAttribute(RTbuffer buffer, RTbufferattribute attrib, RTsize size, void* p);


/************************************
 **
 **    PostprocessingStage object
 **
 ***********************************/


  /**
  * @brief Creates a new post-processing stage
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageCreateBuiltin creates a new post-processing stage selected from a list of
  * pre-defined post-processing stages. The \a context specifies the target context, and should be
  * a value returned by @ref rtContextCreate.
  * Sets \a *stage to the handle of a newly created stage within \a context.
  *
  * @param[in]   context      Specifies the rendering context to which the post-processing stage belongs
  * @param[in]   builtinName  The name of the built-in stage to instantiate
  * @param[out]  stage        New post-processing stage handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageCreateBuiltin was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtPostProcessingStageDestroy,
  * @ref rtPostProcessingStageGetContext,
  * @ref rtPostProcessingStageQueryVariable,
  * @ref rtPostProcessingStageGetVariableCount
  * @ref rtPostProcessingStageGetVariable
  *
  */
  RTresult RTAPI rtPostProcessingStageCreateBuiltin(RTcontext context, const char* builtinName, RTpostprocessingstage* stage);

  /**
  * @brief Destroy a post-processing stage
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageDestroy destroys a post-processing stage from its context and deletes
  * it. The variables built into the stage are destroyed. After the call, \a stage is no longer a valid handle.
  * After a post-processing stage was destroyed all command lists containing that stage are invalidated and
  * can no longer be used.
  *
  * @param[in]  stage        Handle of the post-processing stage to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageDestroy was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtPostProcessingStageCreateBuiltin,
  * @ref rtPostProcessingStageGetContext,
  * @ref rtPostProcessingStageQueryVariable,
  * @ref rtPostProcessingStageGetVariableCount
  * @ref rtPostProcessingStageGetVariable
  *
  */
  RTresult RTAPI rtPostProcessingStageDestroy(RTpostprocessingstage stage);

  /**
  * @brief Declares a new named variable associated with a PostprocessingStage
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageDeclareVariable declares a new variable associated with a
  * postprocessing stage. \a stage specifies the post-processing stage, and should be a value
  * returned by @ref rtPostProcessingStageCreateBuiltin. \a name specifies the name of the variable, and
  * should be a \a NULL-terminated string. If there is currently no variable associated with \a
  * stage named \a name, a new variable named \a name will be created and associated with
  * \a stage.  After the call, \a *v will be set to the handle of the newly-created
  * variable.  Otherwise, \a *v will be set to \a NULL. After declaration, the variable can be
  * queried with @ref rtPostProcessingStageQueryVariable or @ref rtPostProcessingStageGetVariable. A
  * declared variable does not have a type until its value is set with one of the @ref rtVariableSet
  * functions. Once a variable is set, its type cannot be changed anymore.
  *
  * @param[in]   stage   Specifies the associated postprocessing stage
  * @param[in]   name               The name that identifies the variable
  * @param[out]  v                  Returns a handle to a newly declared variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageDeclareVariable was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref Variables,
  * @ref rtPostProcessingStageQueryVariable,
  * @ref rtPostProcessingStageGetVariable
  *
  */
  RTresult RTAPI rtPostProcessingStageDeclareVariable(RTpostprocessingstage stage, const char* name, RTvariable* v);

  /**
  * @brief Returns the context associated with a post-processing stage.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageGetContext queries a stage for its associated context.
  * \a stage specifies the post-processing stage to query, and should be a value
  * returned by @ref rtPostProcessingStageCreateBuiltin. If both parameters are valid,
  * \a *context is set to the context associated with \a stage. Otherwise, the call
  * has no effect and returns @ref RT_ERROR_INVALID_VALUE.
  *
  * @param[in]   stage      Specifies the post-processing stage to query
  * @param[out]  context    Returns the context associated with the material
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageGetContext was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtPostProcessingStageCreateBuiltin,
  * @ref rtPostProcessingStageDestroy,
  * @ref rtPostProcessingStageQueryVariable,
  * @ref rtPostProcessingStageGetVariableCount
  * @ref rtPostProcessingStageGetVariable
  *
  */
  RTresult RTAPI rtPostProcessingStageGetContext(RTpostprocessingstage stage, RTcontext* context);

  /**
  * @brief Returns a handle to a named variable of a post-processing stage
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageQueryVariable queries the handle of a post-processing stage's named
  * variable. \a stage specifies the source post-processing stage, as returned by
  * @ref rtPostProcessingStageCreateBuiltin. \a name specifies the name of the variable, and should be a
  * \a NULL -terminated string. If \a name is the name of a variable attached to \a stage, the call
  * returns a handle to that variable in \a *variable, otherwise \a NULL. Only pre-defined variables of that
  * built-in stage type can be queried. It is not possible to add or remove variables.
  *
  * @param[in]   stage              The post-processing stage to query the variable from
  * @param[in]   name               The name that identifies the variable to be queried
  * @param[out]  variable           Returns the named variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageQueryVariable was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtPostProcessingStageCreateBuiltin,
  * @ref rtPostProcessingStageDestroy,
  * @ref rtPostProcessingStageGetContext,
  * @ref rtPostProcessingStageGetVariableCount
  * @ref rtPostProcessingStageGetVariable
  *
  */
  RTresult RTAPI rtPostProcessingStageQueryVariable(RTpostprocessingstage stage, const char* name, RTvariable* variable);

  /**
  * @brief Returns the number of variables pre-defined in a post-processing stage.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageGetVariableCount returns the number of variables which are pre-defined
  * in a post-processing stage. This can be used to iterate over the variables. Sets \a *count to the
  * number.
  *
  * @param[in]   stage              The post-processing stage to query the number of variables from
  * @param[out]  count              Returns the number of pre-defined variables
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageGetVariableCount was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtPostProcessingStageCreateBuiltin,
  * @ref rtPostProcessingStageDestroy,
  * @ref rtPostProcessingStageGetContext,
  * @ref rtPostProcessingStageQueryVariable,
  * @ref rtPostProcessingStageGetVariable
  *
  */
  RTresult RTAPI rtPostProcessingStageGetVariableCount(RTpostprocessingstage stage , unsigned int* count);

  /**
  * @brief Returns a handle to a variable of a post-processing stage. The variable is defined by index.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtPostProcessingStageGetVariable queries the handle of a post-processing stage's variable which
  * is identified by its index . \a stage specifies the source post-processing stage, as returned by
  * @ref rtPostProcessingStageCreateBuiltin. \a index specifies the index of the variable, and should be a
  * less than the value return by @ref rtPostProcessingStageGetVariableCount. If \a index is in the valid
  * range, the call returns a handle to that variable in \a *variable, otherwise \a NULL.
  *
  * @param[in]   stage              The post-processing stage to query the variable from
  * @param[in]   index              The index identifying the variable to be returned
  * @param[out]  variable           Returns the variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtPostProcessingStageGetVariable was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtPostProcessingStageCreateBuiltin,
  * @ref rtPostProcessingStageDestroy,
  * @ref rtPostProcessingStageGetContext,
  * @ref rtPostProcessingStageQueryVariable,
  * @ref rtPostProcessingStageGetVariableCount
  *
  */
  RTresult RTAPI rtPostProcessingStageGetVariable(RTpostprocessingstage stage, unsigned int index, RTvariable* variable);


  /************************************
   **
   **    CommandList object
   **
   ***********************************/


  /**
  * @brief Creates a new command list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListCreate creates a new command list. The \a context specifies the target
  * context, and should be a value returned by @ref rtContextCreate. The call
  * sets \a *list to the handle of a newly created list within \a context.
  * Returns @ref RT_ERROR_INVALID_VALUE if \a list is \a NULL.
  *
  * A command list can be used to assemble a list of different types of commands and execute them
  * later. At this point, commands can be built-in post-processing stages or context launches. Those
  * are appended to the list using @ref rtCommandListAppendPostprocessingStage, and @ref
  * rtCommandListAppendLaunch2D, respectively. Commands will be executed in the order they have been
  * appended to the list. Thus later commands can use the results of earlier commands. Note that
  * all commands added to the created list must be associated with the same \a context. It is
  * invalid to mix commands from  different contexts.
  *
  * @param[in]   context     Specifies the rendering context of the command list
  * @param[out]  list        New command list handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtCommandListCreate was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListFinalize,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListCreate(RTcontext context, RTcommandlist* list);

  /**
  * @brief Destroy a command list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListDestroy destroys a command list from its context and deletes it. After the
  * call, \a list is no longer a valid handle. Any stages associated with the command list are not destroyed.
  *
  * @param[in]  list        Handle of the command list to destroy
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListDestroy was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListFinalize,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListDestroy(RTcommandlist list);

  /**
  * @brief Append a post-processing stage to the command list \a list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListAppendPostprocessingStage appends a post-processing stage to the command list
  * \a list. The command list must have been created from the same context as the the post-processing
  * stage.
  * The launchWidth and launchHeight specify the launch dimensions and may be different than the
  * input or output buffers associated with each post-processing stage depending on the requirements
  * of the post-processing stage appended.
  * It is invalid to call @ref rtCommandListAppendPostprocessingStage after calling @ref
  * rtCommandListFinalize.
  *
  * NOTE: A post-processing stage can be added to multiple command lists or added to the same command
  * list multiple times.  Also note that destroying a post-processing stage will invalidate all command
  * lists it was added to.
  *
  * @param[in]  list          Handle of the command list to append to
  * @param[in]  stage         The post-processing stage to append to the command list
  * @param[in]  launchWidth   This is a hint for the width of the launch dimensions to use for this stage.
  *                           The stage can ignore this and use a suitable launch width instead.
  * @param[in]  launchHeight  This is a hint for the height of the launch dimensions to use for this stage.
  *                           The stage can ignore this and use a suitable launch height instead.
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListAppendPostprocessingStage was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListFinalize,
  * @ref rtCommandListExecute
  * @ref rtPostProcessingStageCreateBuiltin,
  *
  */
  RTresult RTAPI rtCommandListAppendPostprocessingStage(RTcommandlist list, RTpostprocessingstage stage, RTsize launchWidth, RTsize launchHeight);

  /**
  * @brief Append a 1D launch to the command list \a list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListAppendLaunch1D appends a 1D context launch to the command list \a list. It is
  * invalid to call @ref rtCommandListAppendLaunch1D after calling @ref rtCommandListFinalize.
  *
  * @param[in]  list              Handle of the command list to append to
  * @param[in]  entryPointIndex   The initial entry point into the kernel
  * @param[in]  launchWidth       Width of the computation grid
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListAppendLaunch2D was introduced in OptiX 6.1.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListAppendLaunch3D,
  * @ref rtCommandListFinalize,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListAppendLaunch1D(RTcommandlist list, unsigned int entryPointIndex, RTsize launchWidth);

  /**
  * @brief Append a 2D launch to the command list \a list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListAppendLaunch2D appends a 2D context launch to the command list \a list. It is
  * invalid to call @ref rtCommandListAppendLaunch2D after calling @ref rtCommandListFinalize.
  *
  * @param[in]  list              Handle of the command list to append to
  * @param[in]  entryPointIndex   The initial entry point into the kernel
  * @param[in]  launchWidth       Width of the computation grid
  * @param[in]  launchHeight      Height of the computation grid
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListAppendLaunch2D was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch1D,
  * @ref rtCommandListAppendLaunch3D,
  * @ref rtCommandListFinalize,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListAppendLaunch2D(RTcommandlist list, unsigned int entryPointIndex, RTsize launchWidth, RTsize launchHeight);

  /**
  * @brief Append a 3D launch to the command list \a list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListAppendLaunch3D appends a 3D context launch to the command list \a list. It is
  * invalid to call @ref rtCommandListAppendLaunch3D after calling @ref rtCommandListFinalize.
  *
  * @param[in]  list              Handle of the command list to append to
  * @param[in]  entryPointIndex   The initial entry point into the kernel
  * @param[in]  launchWidth       Width of the computation grid
  * @param[in]  launchHeight      Height of the computation grid
  * @param[in]  launchDepth       Depth of the computation grid
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListAppendLaunch2D was introduced in OptiX 6.1.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch1D,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListFinalize,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListAppendLaunch3D(RTcommandlist list, unsigned int entryPointIndex, RTsize launchWidth, RTsize launchHeight, RTsize launchDepth);

  /**
  * @brief Sets the devices to use for this command list.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListSetDevices specifies a list of hardware devices to use for this command list. This
  * must be a subset of the currently active devices, see @ref rtContextSetDevices. If not set then all the
  * active devices will be used.
  *
  * @param[in]  list      Handle of the command list to set devices for
  * @param[in]  count     The number of devices in the list
  * @param[in]  devices   The list of devices
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * <B>See also</B>
  * @ref rtContextSetDevices,
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListSetDevices(RTcommandlist list, unsigned int count, const int* devices);

    /**
  * @brief Retrieve a list of hardware devices being used by the command list.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListGetDevices retrieves a list of hardware devices used by the command list.
  * Note that the device numbers are  OptiX device ordinals, which may not be the same as CUDA device ordinals.
  * Use @ref rtDeviceGetAttribute with @ref RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL to query the CUDA device
  * corresponding to a particular OptiX device.
  *
  * Note that if the list of set devices is empty then all active devices will be used.
  *
  * @param[in]   list      The command list to which the hardware list is applied
  * @param[out]  devices   Return parameter for the list of devices. The memory must be able to hold entries
  * numbering least the number of devices as returned by @ref rtCommandListGetDeviceCount
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * <B>See also</B>
  * @ref rtCommandListSetDevices,
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListGetDevices(RTcommandlist list, int* devices);

  /**
  * @brief Query the number of devices currently being used by the command list.
  *
  * @ingroup Context
  *
  * <B>Description</B>
  *
  * @ref rtCommandListGetDeviceCount queries the number of devices currently being used.
  *
  * @param[in]   list      The command list containing the devices
  * @param[out]  count     Return parameter for the device count
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * <B>See also</B>
  * @ref rtCommandListSetDevices,
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListGetDeviceCount(RTcommandlist list, unsigned int* count);

  /**
  * @brief Finalize the command list. This must be done before executing the command list.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListFinalize finalizes the command list. This will do all work necessary to
  * prepare the command list for execution. Specifically it will do all work which can be shared
  * between subsequent calls to @ref rtCommandListExecute.
  * It is invalid to call @ref rtCommandListExecute before calling @ref rtCommandListFinalize. It is
  * invalid to call @ref rtCommandListAppendPostprocessingStage or
  * @ref rtCommandListAppendLaunch2D after calling finalize and will result in an error. Also
  * @ref rtCommandListFinalize can only be called once on each command list.
  *
  * @param[in]  list              Handle of the command list to finalize
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListFinalize was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListExecute
  *
  */
  RTresult RTAPI rtCommandListFinalize(RTcommandlist list);

  /**
  * @brief Execute the command list.
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListExecute executes the command list. All added commands will be executed in the
  * order in which they were added. Commands can access the results of earlier executed commands.
  * This must be called after calling @ref rtCommandListFinalize, otherwise an error will be returned
  * and the command list is not executed.
  * @ref rtCommandListExecute can be called multiple times, but only one call may be active at the
  * same time. Overlapping calls from multiple threads will result in undefined behavior.
  *
  * @param[in]  list              Handle of the command list to execute
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListExecute was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtCommandListCreate,
  * @ref rtCommandListDestroy,
  * @ref rtCommandListAppendPostprocessingStage,
  * @ref rtCommandListAppendLaunch2D,
  * @ref rtCommandListFinalize,
  *
  */
  RTresult RTAPI rtCommandListExecute(RTcommandlist list);

  /**
  * @brief Returns the context associated with a command list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListGetContext queries the context associated with a command list. The
  * target command list is specified by \a list. The context of the command list is
  * returned to \a *context if the pointer \a context is not \a NULL. If \a list is
  * not a valid command list, \a *context is set to \a NULL and @ref RT_ERROR_INVALID_VALUE is
  * returned.
  *
  * @param[in]   list       Specifies the command list to be queried
  * @param[out]  context    Returns the context associated with the command list
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListGetContext was introduced in OptiX 5.0.
  *
  * <B>See also</B>
  * @ref rtContextDeclareVariable
  *
  */
  RTresult RTAPI rtCommandListGetContext(RTcommandlist list, RTcontext* context);


  /**
  * @brief Sets the attribute program on a GeometryTriangles object
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesSetAttributeProgram sets for \a geometrytriangles the \a
  * program that performs attribute computation.  RTprograms can be either generated with
  * @ref rtProgramCreateFromPTXFile or @ref rtProgramCreateFromPTXString. An attribute
  * program is optional.  If no attribute program is specified, a default attribute
  * program will be provided.  Attributes are computed after intersection and before any-
  * hit or closest-hit programs that require those attributes.  No assumptions about the
  * precise invocation time should be made.
  * The default attribute program provides the attribute rtTriangleBarycentrics of type float2.
  *
  * Names are case sensitive and types must match.  To use the attribute, declare the following
  *    rtDeclareVariable( float2, barycentrics, attribute rtTriangleBarycentrics, );
  *
  * If you provide an attribute program, the following device side functions will be available:
  *    float2 rtGetTriangleBarycentrics();
  *    unsigned int rtGetPrimitiveIndex();
  *    bool rtIsTriangleHit();
  *    bool rtIsTriangleHitFrontFace();
  *    bool rtIsTriangleHitBackFace();
  *
  * besides other semantics such as the ray time for motion blur.
  *
  * @param[in]   geometrytriangles  The geometrytriangles node for which to set the attribute program
  * @param[in]   program            A handle to the attribute program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesSetAttributeProgram was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetAttributeProgram,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString,
  * @ref rtGetTriangleBarycentrics,
  *
  */
  RTresult RTAPI rtGeometryTrianglesSetAttributeProgram( RTgeometrytriangles geometrytriangles, RTprogram program );


  /**
  * @brief Gets the attribute program of a GeometryTriangles object
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetAttributeProgram gets the attribute \a program of a given
  * \a geometrytriangles object.  If no program has been set, 0 is returned.
  *
  * @param[in]   geometrytriangles  The geometrytriangles node for which to set the attribute program
  * @param[out]  program            A pointer to a handle to the attribute program
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetAttributeProgram was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesDeclareVariable,
  * @ref rtGeometryTrianglesSetAttributeProgram,
  * @ref rtProgramCreateFromPTXFile,
  * @ref rtProgramCreateFromPTXString
  *
  */

  RTresult RTAPI rtGeometryTrianglesGetAttributeProgram( RTgeometrytriangles geometrytriangles, RTprogram* program );


  /**
  * @brief Declares a geometry variable for a GeometryTriangles object
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesDeclareVariable declares a \a variable attribute of a \a geometrytriangles object with
  * a specified \a name.
  *
  * @param[in]   geometrytriangles     A geometry node
  * @param[in]   name                  The name of the variable
  * @param[out]  v                     A pointer to a handle to the variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesDeclareVariable was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetVariable,
  * @ref rtGeometryTrianglesGetVariableCount,
  * @ref rtGeometryTrianglesQueryVariable,
  * @ref rtGeometryTrianglesRemoveVariable
  *
  */

  RTresult RTAPI rtGeometryTrianglesDeclareVariable( RTgeometrytriangles geometrytriangles, const char* name, RTvariable* v );


  /**
  * @brief Queries a variable attached to a GeometryTriangles object
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesQueryVariable gets a variable with a given \a name from
  * a \a geometrytriangles object.
  *
  * @param[in]   geometrytriangles    A geometrytriangles object
  * @param[in]   name                 Thee name of the variable
  * @param[out]  v                    A pointer to a handle to the variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesQueryVariable was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesGetVariable,
  * @ref rtGeometryTrianglesGetVariableCount,
  * @ref rtGeometryTrianglesQueryVariable,
  * @ref rtGeometryTrianglesRemoveVariable
  *
  */

  RTresult RTAPI rtGeometryTrianglesQueryVariable( RTgeometrytriangles geometrytriangles, const char* name, RTvariable* v );


  /**
  * @brief Removes a variable from GeometryTriangles object
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesRemoveVariable removes a variable from
  * a \a geometrytriangles object.
  *
  * @param[in]   geometrytriangles     A geometrytriangles object
  * @param[in]   v                     A pointer to a handle to the variable
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesRemoveVariable was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesDeclareVariable,
  * @ref rtGeometryTrianglesGetVariable,
  * @ref rtGeometryTrianglesGetVariableCount,
  * @ref rtGeometryTrianglesQueryVariable
  *
  */

  RTresult RTAPI rtGeometryTrianglesRemoveVariable( RTgeometrytriangles geometrytriangles, RTvariable v );


  /**
  * @brief Get the number of variables attached to a GeometryTriangles object
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetVariableCount returns a \a count of the number
  * of variables attached to a \a geometrytriangles object.
  *
  * @param[in]   geometrytriangles   A geometrytriangles node
  * @param[out]  count               A pointer to an unsigned int
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetVariableCount was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesDeclareVariable,
  * @ref rtGeometryTrianglesGetVariable,
  * @ref rtGeometryTrianglesQueryVariable,
  * @ref rtGeometryTrianglesRemoveVariable
  *
  */

  RTresult RTAPI rtGeometryTrianglesGetVariableCount( RTgeometrytriangles geometrytriangles, unsigned int* count );


  /**
  * @brief Get a variable attached to a GeometryTriangles object at a specified index.
  *
  * @ingroup GeometryTriangles
  *
  * <B>Description</B>
  *
  * @ref rtGeometryTrianglesGetVariable returns the variable attached at a given
  * index to the specified GeometryTriangles object.
  *
  * @param[in]   geometrytriangles   A geometry node
  * @param[in]   index               The index of the variable
  * @param[out]  v                   A pointer to a variable handle
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtGeometryTrianglesGetVariable was introduced in OptiX 6.0.
  *
  * <B>See also</B>
  * @ref rtGeometryTrianglesDeclareVariable,
  * @ref rtGeometryTrianglesGetVariableCount,
  * @ref rtGeometryTrianglesQueryVariable,
  * @ref rtGeometryTrianglesRemoveVariable
  *
  */

  RTresult RTAPI rtGeometryTrianglesGetVariable( RTgeometrytriangles geometrytriangles, unsigned int index, RTvariable* v );


#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_host_h__ */
