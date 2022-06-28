
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
 * @file   optix_d3d11_interop.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API declarations D3D11 interop
 *
 * OptiX public API declarations for D3D11 interoperability
 */

#ifndef __optix_optix_dx11_interop_h__
#define __optix_optix_dx11_interop_h__

/************************************
 **
 **    DX11 Interop functions
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
#include <optix_d3d11_interop.h>

*/

#if defined( _WIN32 )

#include "optix.h"

typedef struct IDXGIAdapter   IDXGIAdapter;
typedef struct ID3D11Device   ID3D11Device;
typedef struct ID3D11Resource ID3D11Resource;

#ifdef __cplusplus
  extern "C" 
  {
#endif

  /**
  * @brief Binds a D3D11 device to a context and enables interop
  * 
  * @ingroup Context
  * 
  * <B>Description</B>
  * 
  * @ref rtContextSetD3D11Device binds \a device to \a context and enables D3D11 interop capabilities in \a context. This 
  * function must be executed once for \a context before any call to @ref rtBufferCreateFromD3D11Resource or @ref rtTextureSamplerCreateFromD3D11Resource can 
  * take place. A context can only be bound to one device. Once \a device is bound to \a context, the binding is immutable and remains upon destruction of \a context.
  * 
  * @param[in]   context      The context to bind the device with
  * @param[in]   device       The D3D11 device to be used for interop with the associated context
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
  * @ref rtContextSetD3D11Device was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource,
  * @ref rtTextureSamplerCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtContextSetD3D11Device                ( RTcontext context, ID3D11Device* device );
  
  /**
  * @brief Returns the OptiX device number associated with the pointer to a D3D11 adapter
  * 
  * @ingroup ContextFreeFunctions
  * 
  * <B>Description</B>
  * 
  * @ref rtDeviceGetD3D11Device returns in \a device the OptiX device ID of the adapter represented by \a D3D11Device.
  * \a D3D11Device is a pointer returned from \a D3D11CreateDeviceAndSwapChain. In combination with @ref rtContextSetDevices,
  * this function can be used to restrict OptiX to use only one device. The same device the D3D11 commands will be sent to.
  *
  * This function is only supported on Windows platforms.
  * 
  * @param[in]   device       A handle to the memory location where the OptiX device ordinal associated with \a D3D11Device will be stored
  * @param[in]   pAdapter     A pointer to an \a ID3D11Device as returned from \a D3D11CreateDeviceAndSwapChain
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * 
  * <B>History</B>
  * 
  * @ref rtDeviceGetD3D11Device was introduced in OptiX 2.5.
  * 
  * <B>See also</B>
  * @ref rtDeviceGetDeviceCount
  * 
  */
  RTresult RTAPI rtDeviceGetD3D11Device                 ( int* device, IDXGIAdapter *pAdapter );
  
  /**
  * @brief Creates a new buffer object from a D3D11 resource
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * @ref rtBufferCreateFromD3D11Resource allocates and returns a handle to a new buffer object in \a *buffer associated with 
  * \a context. If the allocated size of the D3D resource is \a 0, @ref RT_ERROR_MEMORY_ALLOCATION_FAILED will be returned. Supported D3D11 buffer types are:
  *
  * - ID3D11Buffer
  *
  * These buffers can be used to share data with D3D11; changes of the content in \a buffer, either done by D3D11 or OptiX,
  * will be reflected automatically in both APIs. If the size, or format, of a D3D11 buffer is changed, appropriate OptiX
  * calls have to be used to update \a buffer accordingly. OptiX keeps only a reference to D3D11 data, when \a buffer is
  * destroyed, the state of \a resource is unaltered.
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
  * @param[in]   context              The context to create the buffer in
  * @param[in]   bufferdesc           Bitwise \a or combination of the \a type and \a flags of the new buffer
  * @param[in]   resource             The D3D11 resource handle for use in OptiX
  * @param[out]  buffer               The return handle for the buffer object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * 
  * <B>History</B>
  * 
  * @ref rtBufferCreateFromD3D11Resource was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferDestroy
  * 
  */
  RTresult RTAPI rtBufferCreateFromD3D11Resource        ( RTcontext context, unsigned int bufferdesc, ID3D11Resource* resource,  RTbuffer* buffer );
  
  /**
  * @brief Creates a new texture sampler object from a D3D11 resource
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * @ref rtTextureSamplerCreateFromD3D11Resource allocates and returns a
  * handle to a new texture sampler object in \a *texturesampler
  * associated with \a context. If the allocated size of the D3D resource
  * is \a 0, @ref RT_ERROR_MEMORY_ALLOCATION_FAILED will be returned. Supported
  * D3D11 texture types are:
  *
  * - ID3D11Texture1D
  * - ID3D11Texture2D
  * - ID3D11Texture3D
  *
  * These texture samplers can be used to share data with D3D11; changes of
  * the content and size of \a texturesampler done by D3D11 will be
  * reflected automatically in OptiX. Currently texture sampler data are
  * read only in OptiX programs. OptiX keeps only a reference to
  * D3D11 data, when \a texturesampler is destroyed, the state of the
  * \a resource is unaltered.
  *
  * The array size and number of mipmap levels can't be changed for
  * texture samplers that encapsulate a D3D11 resource. Furthermore no
  * buffer objects can be queried. Please refer to the @ref InteropTypes for a
  * complete list of supported texture formats.
  * 
  * @param[in]   context              The context to create the texture sampler in
  * @param[in]   resource             The D3D11 resource handle for use in OptiX
  * @param[out]  textureSampler       The return handle for the texture sampler object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  * 
  * <B>History</B>
  * 
  * @ref rtTextureSamplerCreateFromD3D11Resource was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreate,
  * @ref rtTextureSamplerDestroy
  * 
  */
  RTresult RTAPI rtTextureSamplerCreateFromD3D11Resource( RTcontext context, ID3D11Resource* resource,  RTtexturesampler* textureSampler );
  
  /**
  * @brief Gets the D3D11 resource associated with this buffer
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * @ref rtBufferGetD3D11Resource stores the D3D11 resource pointer in \a **resource if \a buffer was created with
  * @ref rtBufferCreateFromD3D11Resource.  If \a buffer was not created from a D3D11 resource \a **resource will be \a 0 after
  * the call and @ref RT_ERROR_INVALID_VALUE is returned.
  * 
  * @param[in]    buffer              The buffer to be queried for its D3D11 resource
  * @param[out]   resource            The return handle for the resource
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
  * @ref rtBufferGetD3D11Resource was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtBufferGetD3D11Resource               ( RTbuffer buffer, ID3D11Resource** resource );
  
  /**
  * @brief Gets the D3D11 resource associated with this texture sampler
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * @ref rtTextureSamplerGetD3D11Resource stores the D3D11 resource pointer in \a **resource if \a sampler was created with
  * @ref rtTextureSamplerGetD3D11Resource.  If \a sampler was not created from a D3D11 resource \a resource will be 0 after
  * the call and @ref RT_ERROR_INVALID_VALUE is returned
  * 
  * @param[in]    textureSampler              The texture sampler to be queried for its D3D11 resource
  * @param[out]   resource                    The return handle for the resource
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
  * @ref rtTextureSamplerGetD3D11Resource was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtTextureSamplerGetD3D11Resource       ( RTtexturesampler textureSampler, ID3D11Resource** resource );
  
  /**
  * @brief Declares a D3D11 buffer as immutable and accessible by OptiX
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * An OptiX buffer in an unregistered state can be registered to OptiX again via @ref rtBufferD3D11Register. Once registered, 
  * properties like the size of the original D3D11 resource cannot be modified anymore. Calls to the corresponding D3D11 functions 
  * will return with an error code. However, the data of the D3D11 resource can still be read and written by the appropriate D3D11 commands. 
  * When a buffer is already in a registered state @ref rtBufferD3D11Register will return @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED. A resource 
  * must be registered in order to be used by OptiX. If a resource is not registered @ref RT_ERROR_INVALID_VALUE will be returned.
  * 
  * @param[in]    buffer              The handle for the buffer object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtBufferD3D11Register was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtBufferD3D11Register                  ( RTbuffer buffer );
  
  /**
  * @brief Declares a D3D11 buffer as mutable and inaccessible by OptiX
  * 
  * @ingroup Buffer
  * 
  * <B>Description</B>
  * 
  * An OptiX buffer in a registered state can be unregistered via @ref rtBufferD3D11Register. Once unregistered, 
  * properties like the size of the original D3D11 resource can be changed. As long as a resource is unregistered, 
  * OptiX will not be able to access the data and will fail with @ref RT_ERROR_INVALID_VALUE. When a buffer is already 
  * in an unregistered state @ref rtBufferD3D11Unregister will return @ref RT_ERROR_RESOURCE_NOT_REGISTERED.  
  * 
  * @param[in]    buffer              The handle for the buffer object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_NOT_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtBufferD3D11Unregister was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtBufferCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtBufferD3D11Unregister                ( RTbuffer buffer );
  
  /**
  * @brief Declares a D3D11 texture as immutable and accessible by OptiX
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * An OptiX texture sampler in an unregistered state can be registered to OptiX again via @ref rtTextureSamplerD3D11Register. 
  * Once registered, properties like the size of the original D3D11 resource cannot be modified anymore. Calls to the corresponding 
  * D3D11 functions will return with an error code. However, the data of the D3D11 resource can still be read and written by the appropriate 
  * D3D11 commands. When a texture sampler is already in a registered state @ref rtTextureSamplerD3D11Register will return @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED. 
  * A resource must be registered in order to be used by OptiX. If a resource is not registered @ref RT_ERROR_INVALID_VALUE will be returned.
  * 
  * @param[in]    textureSampler              The handle for the texture object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_ALREADY_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtTextureSamplerD3D11Register was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtTextureSamplerD3D11Register          ( RTtexturesampler textureSampler );
  
  /**
  * @brief Declares a D3D11 texture as mutable and inaccessible by OptiX
  * 
  * @ingroup TextureSampler
  * 
  * <B>Description</B>
  * 
  * An OptiX texture sampler in a registered state can be unregistered via @ref rtTextureSamplerD3D11Unregister. Once unregistered, 
  * properties like the size of the original D3D11 resource can be changed. As long as a resource is unregistered, OptiX will not 
  * be able to access the data and will fail with @ref RT_ERROR_INVALID_VALUE. When a buffer is already in an unregistered state
  * @ref rtBufferD3D11Unregister will return @ref RT_ERROR_RESOURCE_NOT_REGISTERED. 
  * 
  * @param[in]    textureSampler              The handle for the texture object
  * 
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_INVALID_VALUE
  * - @ref RT_ERROR_INVALID_CONTEXT
  * - @ref RT_ERROR_RESOURCE_NOT_REGISTERED
  * 
  * <B>History</B>
  * 
  * @ref rtTextureSamplerD3D11Unregister was introduced in OptiX 2.0.
  * 
  * <B>See also</B>
  * @ref rtTextureSamplerCreateFromD3D11Resource
  * 
  */
  RTresult RTAPI rtTextureSamplerD3D11Unregister        ( RTtexturesampler textureSampler );

#ifdef __cplusplus
  }

#endif

#endif /* defined( _WIN32 ) */

#endif /* __optix_optix_dx11_interop_h__ */
