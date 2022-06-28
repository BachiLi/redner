
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
 * @file   optix_gl_interop.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API declarations GLInterop
 *
 * OptiX public API declarations for GL interoperability
 */

#ifndef __optix_optix_gl_interop_h__
#define __optix_optix_gl_interop_h__

#include "optix.h"

/************************************
 **
 **    OpenGL Interop functions
 **
 ***********************************/

/*

On Windows you will need to include windows.h before this file:

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include<windows.h>
#endif
#include <optix_gl_interop.h>

*/

#ifdef __cplusplus
extern "C" {
#endif

  /**
  * @brief Creates a new buffer object from an OpenGL buffer object
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * @ref rtBufferCreateFromGLBO allocates and returns a handle to a new buffer object in *\a buffer associated with 
  * \a context. Supported OpenGL buffer types are:
  *
  * - Pixel Buffer Objects
  *
  * - Vertex Buffer Objects
  *
  * These buffers can be used to share data with OpenGL; changes of the content in \a buffer, either done by OpenGL or OptiX,
  * will be reflected automatically in both APIs. If the size, or format, of an OpenGL buffer is changed, appropriate OptiX 
  * calls have to be used to update \a buffer accordingly. OptiX keeps only a reference to OpenGL data, when \a buffer is
  * destroyed, the state of the \a gl_id object is unaltered.
  *
  * The \a type of this buffer is specified by one of the following values in \a bufferdesc:
  *
  * - @ref RT_BUFFER_INPUT
  * - @ref RT_BUFFER_OUTPUT
  * - @ref RT_BUFFER_INPUT_OUTPUT
  *
  * The type values are used to specify the direction of data flow from the host to the OptiX devices.
  * @ref RT_BUFFER_INPUT specifies that the host may only write to the buffer and the device may only read from the buffer.
  * @ref RT_BUFFER_OUTPUT specifies the opposite, read only access on the host and write only access on the device.
  * Devices and the host may read and write from buffers of type @ref RT_BUFFER_INPUT_OUTPUT.  Reading or writing to
  * a buffer of the incorrect type (e.g., the host writing to a buffer of type @ref RT_BUFFER_OUTPUT) is undefined.
  *
  * Flags can be used to optimize data transfers between the host and it's devices. Currently no \a flags are supported for
  * interop buffers.
  * 
  * @param[in]   context      The context to create the buffer in
  * @param[in]   bufferdesc   Bitwise \a or combination of the \a type and \a flags of the new buffer
  * @param[in]   glId         The OpenGL image object resource handle for use in OptiX
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
  * @ref rtBufferCreateFromGLBO was introduced in OptiX 1.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferDestroy
  * 
  */
  RTresult RTAPI rtBufferCreateFromGLBO           ( RTcontext context, unsigned int bufferdesc, unsigned int glId,  RTbuffer* buffer );
  
  /**
  * @brief Creates a new texture sampler object from an OpenGL image
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * @ref rtTextureSamplerCreateFromGLImage allocates and returns a handle to
  * a new texture sampler object in * \a texturesampler associated with
  * \a context. If the allocated size of the GL texture is 0,
  * @ref RT_ERROR_MEMORY_ALLOCATION_FAILED will be returned. Supported OpenGL
  * image types are:
  *
  * Renderbuffers
  *
  * - GL_TEXTURE_2D
  *
  * - GL_TEXTURE_2D_RECT
  *
  * - GL_TEXTURE_3D
  *
  *
  * These types are reflected by \a target:
  *
  * - @ref RT_TARGET_GL_RENDER_BUFFER
  *
  * - @ref RT_TARGET_GL_TEXTURE_1D
  *
  * - @ref RT_TARGET_GL_TEXTURE_2D
  *
  * - @ref RT_TARGET_GL_TEXTURE_RECTANGLE
  *
  * - @ref RT_TARGET_GL_TEXTURE_3D
  *
  * - @ref RT_TARGET_GL_TEXTURE_1D_ARRAY
  *
  * - @ref RT_TARGET_GL_TEXTURE_2D_ARRAY
  *
  * - @ref RT_TARGET_GL_TEXTURE_CUBE_MAP  
  *
  * - @ref RT_TARGET_GL_TEXTURE_CUBE_MAP_ARRAY  
  *
  * Supported attachment points for renderbuffers are:
  *
  * - GL_COLOR_ATTACHMENT<NUM>
  *
  *
  * These texture samplers can be used to share data with OpenGL; changes
  * of the content and size of \a texturesampler done by OpenGL will be
  * reflected automatically in OptiX. Currently texture sampler data are
  * read only in OptiX programs. OptiX keeps only a reference to
  * OpenGL data, when \a texturesampler is destroyed, the state of the
  * \a gl_id image is unaltered.
  *
  * The array size and number of mipmap levels can't be changed for
  * texture samplers that encapsulate a GL image. Furthermore no buffer
  * objects can be queried.
  *
  * Currently OptiX supports only a limited number of internal OpenGL
  * texture formats. Texture formats with an internal type of float,
  * e.g. \a GL_RGBA32F, and many integer formats are supported. Depth formats
  * as well as multisample buffers are also currently not supported.
  * Please refer to the @ref InteropTypes section for a complete list of supported
  * texture formats.
  * 
  * @param[in]   context          The context to create the buffer in
  * @param[in]   glId             The OpenGL image object resoure handle for use in OptiX
  * @param[in]   target           The OpenGL target
  * @param[out]  textureSampler   The return handle for the texture sampler object
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
  * @ref rtTextureSamplerCreateFromGLImage was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreate,
  * @ref rtTextureSamplerDestroy
  * 
  */
  RTresult RTAPI rtTextureSamplerCreateFromGLImage( RTcontext context, unsigned int glId, RTgltarget target, RTtexturesampler* textureSampler );
  
  /**
  * @brief Gets the OpenGL Buffer Object ID associated with this buffer
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * @ref rtBufferGetGLBOId stores the OpenGL buffer object id in \a gl_id if
  * \a buffer was created with @ref rtBufferCreateFromGLBO.  If \a buffer was
  * not created from an OpenGL Buffer Object \a gl_id will be set to 0.
  * 
  * @param[in]   buffer          The buffer to be queried for its OpenGL buffer object id
  * @param[in]   glId            The return handle for the id
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
  * @ref rtBufferGetGLBOId was introduced in OptiX 1.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromGLBO
  * 
  */
  RTresult RTAPI rtBufferGetGLBOId                ( RTbuffer buffer, unsigned int* glId );
  
  /**
  * @brief Gets the OpenGL image object id associated with this texture sampler
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * @ref rtTextureSamplerGetGLImageId stores the OpenGL image object id in
  * \a gl_id if \a textureSampler was created with @ref rtTextureSamplerCreateFromGLImage.
  * If \a textureSampler was not created from an OpenGL image object \a gl_id
  * will be set to 0.
  * 
  * @param[in]   textureSampler          The texture sampler to be queried for its OpenGL buffer object id
  * @param[in]   glId                    The return handle for the id
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
  * @ref rtTextureSamplerGetGLImageId was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromGLImage
  * 
  */
  RTresult RTAPI rtTextureSamplerGetGLImageId     ( RTtexturesampler textureSampler, unsigned int* glId );
  
  /**
  * @brief Declares an OpenGL buffer as immutable and accessible by OptiX
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * Once registered, properties like the size of the original GL buffer cannot be modified anymore.
  * Calls to the corresponding GL functions will return with an error code. 
  * However, the buffer data of the GL buffer can still be read and written by the appropriate GL commands.
  * Returns \a RT_ERROR_RESOURCE_ALREADY_REGISTERED if \a buffer is already registered.
  * A buffer object must be registered in order to be used by OptiX.  
  * If a buffer object is not registered @ref RT_ERROR_INVALID_VALUE will be returned.
  * An OptiX buffer in a registered state can be unregistered via @ref rtBufferGLRegister.
  *
  * @param[in]   buffer          The handle for the buffer object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtBufferGLRegister was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromGLBO,
  * @ref rtBufferGLUnregister
  * 
  */
  RTresult RTAPI rtBufferGLRegister               ( RTbuffer buffer );
  
  /**
  * @brief Declares an OpenGL buffer as mutable and inaccessible by OptiX
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * Once unregistered, properties like the size of the original GL buffer can be changed.
  * As long as a buffer object is unregistered, OptiX will not be able to access the data and calls will fail with @ref RT_ERROR_INVALID_VALUE. 
  * Returns \a RT_ERROR_RESOURCE_NOT_REGISTERED if \a buffer is already unregistered.
  * An OptiX buffer in an unregistered state can be registered to OptiX again via @ref rtBufferGLRegister. 
  *
  * @param[in]   buffer          The handle for the buffer object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_RESOURCE_NOT_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtBufferGLUnregister was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromGLBO,
  * @ref rtBufferGLRegister
  * 
  */
  RTresult RTAPI rtBufferGLUnregister             ( RTbuffer buffer );
  
  /**
  * @brief Declares an OpenGL texture as immutable and accessible by OptiX
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * Registers an OpenGL texture as accessible by OptiX. Once registered, properties like the size of the original GL texture cannot be modified anymore.
  * Calls to the corresponding GL functions will return with an error code. However, the pixel data of the GL texture can still be read and written by the appropriate GL commands.
  * Returns @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED if \a textureSampler is already registered. 
  * A texture sampler must be registered in order to be used by OptiX. Otherwise, @ref RT_ERROR_INVALID_VALUE is returned.
  * An OptiX texture sampler in a registered state can be unregistered via @ref rtTextureSamplerGLUnregister.
  *
  * @param[in]   textureSampler          The handle for the texture object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtTextureSamplerGLRegister was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromGLImage,
  * @ref rtTextureSamplerGLUnregister
  * 
  */
  RTresult RTAPI rtTextureSamplerGLRegister       ( RTtexturesampler textureSampler );
  
  /**
  * @brief Declares an OpenGL texture as mutable and inaccessible by OptiX
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * Once unregistered, properties like the size of the original GL texture can be changed.
  * As long as a texture is unregistered, OptiX will not be able to access the pixel data and calls will fail with @ref RT_ERROR_INVALID_VALUE.
  * Returns \a RT_ERROR_RESOURCE_NOT_REGISTERED if \a textureSampler is already unregistered.
  * An OptiX texture sampler in an unregistered state can be registered to OptiX again via @ref rtTextureSamplerGLRegister.
  *
  * @param[in]   textureSampler          The handle for the texture object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_RESOURCE_NOT_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtTextureSamplerGLUnregister was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromGLImage,
  * @ref rtTextureSamplerGLRegister
  * 
  */
  RTresult RTAPI rtTextureSamplerGLUnregister     ( RTtexturesampler textureSampler );


#if defined(_WIN32)
#if !defined(WGL_NV_gpu_affinity)
  typedef void* HGPUNV;
#endif

  /**
  * @brief returns the OptiX device number associated with the specified GPU
  * 
  * @ingroup ContextFreeFunctions
  * 
  * <B>Description</B>
  * 
  * @ref rtDeviceGetWGLDevice returns in \a device the OptiX device ID of the GPU represented by \a gpu.
  * \a gpu is returned from \a WGL_NV_gpu_affinity, an OpenGL extension.  This enables OptiX to create a context
  * on the same GPU that OpenGL commands will be sent to, improving OpenGL interoperation efficiency.
  *
  * @param[out]   device          A handle to the memory location where the OptiX device ordinal associated with \a gpu will be stored
  * @param[in]    gpu             A handle to a GPU as returned from the \a WGL_NV_gpu_affinity OpenGL extension
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * 
  * <B>History</B>
  * 
  * @ref rtDeviceGetWGLDevice was introduced in OptiX 1.0.
  * 
  * <B>See also</B>
  * @ref rtDeviceGetDeviceCount,
  * \a WGL_NV_gpu_affinity
  * 
  */
  RTresult RTAPI rtDeviceGetWGLDevice(int* device, HGPUNV gpu);
#endif

#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_gl_interop_h__ */
