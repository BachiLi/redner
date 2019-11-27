
/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software, related documentation and any
 * modifications thereto.  Any use, reproduction, disclosure or distribution of
 * this software and related documentation without an express license agreement
 * from NVIDIA Corporation is strictly prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
 * *AS IS* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
 * OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL
 * NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR
 * CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR
 * LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS
 * INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGES
 */

/**
 * @file   optix_prime.h
 * @author NVIDIA Corporation
 * @brief  OptiX Prime public API
 *
 * OptiX Prime public API
 */

#ifndef __optix_optix_prime_h__
#define __optix_optix_prime_h__

#define OPTIX_PRIME_VERSION 60500  /* major =  OPTIX_PRIME_VERSION/10000,        *
                                    * minor = (OPTIX_PRIME_VERSION%10000)/100,   *
                                    * micro =  OPTIX_PRIME_VERSION%100           */

#ifndef RTPAPI
#  if defined( _WIN32 )
#    define RTPAPI __declspec(dllimport)
#  else
#    define RTPAPI
#  endif
#endif

#include "optix_prime_declarations.h"

/****************************************
 *
 * PLATFORM-DEPENDENT TYPES
 *
 ****************************************/

#if defined( _WIN64 )
typedef unsigned __int64    RTPsize;
#elif defined( _WIN32 )
typedef unsigned int        RTPsize;
#else
typedef long unsigned int   RTPsize;
#endif

/** Opaque type.  Note that the *_api type should never be used directly. Only the typedef target name will be guaranteed to remain unchanged. */
typedef struct RTPcontext_api*    RTPcontext;
/** Opaque type.  Note that the *_api type should never be used directly. Only the typedef target name will be guaranteed to remain unchanged. */
typedef struct RTPmodel_api*      RTPmodel;
/** Opaque type.  Note that the *_api type should never be used directly. Only the typedef target name will be guaranteed to remain unchanged. */
typedef struct RTPquery_api*      RTPquery;
/** Opaque type.  Note that the *_api type should never be used directly. Only the typedef target name will be guaranteed to remain unchanged. */
typedef struct RTPbufferdesc_api* RTPbufferdesc;

/****************************************
 *
 * FORWARD DECLARATIONS
 *
 ****************************************/

typedef struct CUstream_st *cudaStream_t;

#ifdef __cplusplus
extern "C" {
#endif

  /****************************************
   *
   * CONTEXT
   *
   * The context manages the creation of API objects and encapsulates a
   * particular computational backend.
   *
   ****************************************/

  /**
   * @brief   Creates an OptiX Prime context
   *
   * @ingroup Prime_Context
   *
   * By default, a context created with type @ref RTP_CONTEXT_TYPE_CUDA will
   * use the fastest available CUDA device, but note that specific devices can be selected using
   * @ref rtpContextSetCudaDeviceNumbers. The fastest device will be set as the current device when the
   * function returns. If no CUDA device features compute capability 3.0 or greater,
   * the context creation will fail unless RTP_CONTEXT_TYPE_CPU was specified.
   *
   * @param[in]  type       The type of context to create
   * @param[out] context    Pointer to the new OptiX Prime context
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_OBJECT_CREATION_FAILED
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_MEMORY_ALLOCATION_FAILED
   *
   * Example Usage:
   @code
   RTPcontext context;
   if(rtpContextCreate( RTP_CONTEXT_TYPE_CUDA, &context ) == RTP_SUCCESS ) {
     int deviceNumbers[] = {0,1};
     rtpContextSetCudaDeviceNumbers( 2, deviceNumbers );
   }
   else
     rtpContextCreate( RTP_CONTEXT_TYPE_CPU, &context ); // Fallback to CPU
   @endcode
   
   */
  RTPresult RTPAPI rtpContextCreate( RTPcontexttype type, /*out*/ RTPcontext* context );

  /**
   * @brief   Sets the CUDA devices used by a context
   *
   * @ingroup Prime_Context
   *
   * The fastest device provided in deviceNumbers will be used as the *primary
   * device*. Acceleration structures will be built on that primary device and 
   * copied to the others. All devices must be of compute capability 3.0 or greater.
   * Note that this distribution can be rather costly if the rays are stored in device memory though.
   * For maximum efficiency it is recommended to only ever select one device per context.
   * The current device will be set to the primary device when this function returns.
   *
   * If \a deviceCount==0, then the primary device is selected automatically and
   * all available devices are selected for use. \a deviceNumbers is ignored.
   *
   * @param[in] context        OptiX Prime context
   * @param[in] deviceCount    Number of devices supplied in \a deviceNumbers or 0
   * @param[in] deviceNumbers  Array of integer device indices, or NULL if \a deviceCount==0
   *
   * This function will return an error if the provided context is not of type @ref RTP_CONTEXT_TYPE_CUDA
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpContextSetCudaDeviceNumbers( RTPcontext  context, unsigned deviceCount, const unsigned* deviceNumbers );


  /**
   * @brief   Sets the number of CPU threads used by a CPU context
   *
   * @ingroup Prime_Context
   *
   * This function will return an error if the provided \a context is not of type
   * @ref RTP_CONTEXT_TYPE_CPU.
   *
   * By default, one ray tracing thread is created per CPU core.
   *
   * @param[in] context            OptiX Prime context
   * @param[in] numThreads         Number of threads used for the CPU context
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpContextSetCpuThreads( RTPcontext context, unsigned numThreads );

  /**
   * @brief   Destroys an OptiX Prime context
   *
   * @ingroup Prime_Context
   *
   * Ongoing work is finished before \a context is destroyed. All OptiX Prime
   * objects associated with \a context are aslo destroyed when \a context is
   * destroyed.
   *
   * @param[in] context        OptiX Prime context to destroy
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpContextDestroy( RTPcontext context );

  /**
   * @brief   Returns a string describing last error encountered
   *
   * @ingroup Prime_Context
   *
   * This function returns an error string for the last error encountered in \a context
   * that may contain invocation-specific details beyond the simple @ref RTPresult error
   * code. Note that this function may return errors from previous asynchronous
   * launches or from calls by other threads.
   *
   * @param[in] context         OptiX Prime context
   * @param[out] return_string  String with error details
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   *
   * <B>See also</B>
   * @ref rtpGetErrorString
   *
   */
  RTPresult RTPAPI rtpContextGetLastErrorString( RTPcontext context, /*out*/ const char** return_string );



  /****************************************
   *
   * BUFFER DESCRIPTOR
   *
   * A buffer descriptor provides information about a buffer's type, format,
   * and location. It also describes the region of the buffer to use.
   *
   ****************************************/

  /**
   * @brief   Create a buffer descriptor
   *
   * @ingroup Prime_BufferDesc
   *
   * This function creates a buffer descriptor with the specified element
   * format and buffertype.  A buffer of type
   * @ref RTP_BUFFER_TYPE_CUDA_LINEAR is assumed to reside on the current 
   * device. The device number can be changed by calling @ref
   * rtpBufferDescSetCudaDeviceNumber.
   *
   * @param[in]  context      OptiX Prime context
   * @param[in]  format       Format of the buffer
   * @param[in]  type         Type of the buffer
   * @param[in]  buffer       Pointer to buffer data
   * @param[out] desc         Pointer to the new buffer descriptor
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   * Example Usage:
   @code
   RTPbufferdesc verticesBD;
   rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, vertices, &verticesBD);
   @endcode
  */
  RTPresult RTPAPI rtpBufferDescCreate( RTPcontext context, RTPbufferformat format, RTPbuffertype type, void* buffer, /*out*/ RTPbufferdesc* desc );

  /**
   * @brief   Gets the context object associated with the provided buffer descriptor
   *
   * @ingroup Prime_BufferDesc
   *
   * @param[in]  desc      Buffer descriptor 
   * @param[out] context   Returned context 
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpBufferDescGetContext( RTPbufferdesc desc, /*out*/ RTPcontext* context );

  /**
   * @brief   Sets the element range of a buffer to use
   *
   * @ingroup Prime_BufferDesc
   *
   * The range is specified in terms of number of elements. By default, the range
   * for a buffer is 0 to the number of elements in the buffer.
   *
   * @param[in] desc        Buffer descriptor
   * @param[in] begin       Start index of the range
   * @param[in] end         End index of the range (exclusive, one past the index of the last element)
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpBufferDescSetRange( RTPbufferdesc desc, RTPsize begin, RTPsize end );

  /**
   * @brief   Sets the stride for elements in a buffer
   *
   * @ingroup Prime_BufferDesc
   *
   * This function is only valid for buffers of format
   * @ref RTP_BUFFER_FORMAT_VERTEX_FLOAT3. This function is useful for vertex
   * buffers that contain interleaved vertex attributes. For buffers that are
   * transferred between the host and a device it is recommended that only
   * buffers with default stride be used to avoid transferring data that will
   * not be used.
   *
   * @param[in] desc           Buffer descriptor
   * @param[in] strideBytes    Stride in bytes. The default value of 0 indicates
   *                           that elements are contiguous in memory.
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   * Example Usage:
   @code
   struct Vertex {
      float3 pos, normal, color;
   };
   ...
   RTPbufferdesc vertsBD;
   rtpBufferDescCreate(context, RTP_BUFFER_FORMAT_VERTEX_FLOAT3, RTP_BUFFER_TYPE_HOST, verts, &vertsBD);
   rtpBufferDescSetRange(vertsBD, 0, numVerts);
   rtpBufferDescSetStride(vertsBD, sizeof(Vertex));
   @endcode
  */
  RTPresult RTPAPI rtpBufferDescSetStride( RTPbufferdesc desc, unsigned strideBytes );

  /**
   * @brief   Sets the CUDA device number for a buffer
   *
   * @ingroup Prime_BufferDesc
   *
   * A buffer of type @ref RTP_BUFFER_TYPE_CUDA_LINEAR is assumed to reside
   * on the device that was current when its buffer descriptor was created unless
   * otherwise specified using this function.
   *
   * @param[in] desc           Buffer descriptor
   * @param[in] deviceNumber   CUDA device number
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpBufferDescSetCudaDeviceNumber( RTPbufferdesc desc, unsigned deviceNumber );

  /**
   * @brief   Destroys a buffer descriptor
   *
   * @ingroup Prime_BufferDesc
   *
   * Buffer descriptors can be destroyed immediately after it is used as a
   * function parameter.  The buffer contents associated with a buffer
   * descriptor, however, must remain valid until they are no longer 
   * used by any OptiX Prime objects.
   *
   * @param[in] desc      Buffer descriptor
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpBufferDescDestroy( RTPbufferdesc desc );



  /****************************************
   *
   * MODEL
   *
   * A model is a combination of a set of triangles and an acceleration
   * structure
   *
   ****************************************/

  /**
   * @brief   Creates a model
   *
   * @ingroup Prime_Model
   *
   * @param[in] context      OptiX Prime context
   * @param[out] model       Pointer to the new model
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelCreate( RTPcontext context, /*out*/ RTPmodel* model );

  /**
   * @brief   Gets the context object associated with the model
   *
   * @ingroup Prime_Model
   *
   * @param[in]  model     Model to obtain the context from
   * @param[out] context   Returned context
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelGetContext( RTPmodel model, RTPcontext* context );

  /**
   * @brief   Sets the triangle data for a model
   *
   * @ingroup Prime_Model
   *
   * The index buffer specifies triplet of vertex indices. If the index buffer
   * descriptor is not specified (e.g. indices==NULL), the vertex buffer is
   * considered to be a flat list of triangles, with every three vertices
   * forming a triangle. The buffers are not used until @ref rtpModelUpdate is
   * called.
   *
   * @param[in] model      Model
   * @param[in] indices    Buffer descriptor for triangle vertex indices, or NULL
   * @param[in] vertices   Buffer descriptor for triangle vertices
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelSetTriangles( RTPmodel model, RTPbufferdesc indices, RTPbufferdesc vertices );

  /**
   * @brief   Sets the instance data for a model
   *
   * @ingroup Prime_Model
   *
   * The \a instances buffer specifies a list of model instances, and
   * the \a transforms buffer holds a transformation matrix for each
   * instance. The instance buffer type must be @ref RTP_BUFFER_TYPE_HOST.
   *
   * Instance buffers must be of format @ref RTP_BUFFER_FORMAT_INSTANCE_MODEL,
   * and transform buffers of format @ref RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x4 or
   * RTP_BUFFER_FORMAT_TRANSFORM_FLOAT4x3. If a stride is specified
   * for the transformations, it must be a multiple of 16
   * bytes. Furthermore, the matrices must be stored in row-major
   * order. Only affine transformations are supported, and the last
   * row is always assumed to be [0.0, 0.0, 0.0, 1.0].
   *
   * All instance models in the \a instances buffer must belong to the same context as the model itself.
   * Additionally, the build parameter @ref RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES must be the same
   * for all models (if applied). Setting @ref RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES for a model
   * which contains instances has no effect.
   *
   * The buffers are not used until @ref rtpModelUpdate is called.
   *
   * @param[in] model        Model
   * @param[in] instances    Buffer descriptor for instances
   * @param[in] transforms   Buffer descriptor for 4x4 transform matrices
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelSetInstances( RTPmodel model, RTPbufferdesc instances, RTPbufferdesc transforms );

  /**
   * @brief   Updates data, or creates an acceleration structure over triangles or instances
   *
   * @ingroup Prime_Model
   *
   * Depending on the specified hints, rtpModelUpdate performs different operations:
   *
   * If the flag @ref RTP_MODEL_HINT_ASYNC is specified, some or all of the
   * acceleration structure update may run asynchronously and @ref rtpModelUpdate
   * may return before the update is finished. In the case of  @ref RTP_MODEL_HINT_NONE, 
   * the acceleration structure build is blocking. It is important that buffers 
   * specified in @ref rtpModelSetTriangles and @ref rtpModelSetInstances not be modified
   * until the update has finished. @ref rtpModelFinish blocks the current
   * thread until the update is finished.  @ref rtpModelGetFinished can be used
   * to poll until the update is finished. Once the update has finished the
   * input buffers can be modified.
   *
   * The acceleration structure build performed by rtpModelUpdate uses a fast, high quality
   * algorithm, but has the cost of requiring additional working memory. The amount of working
   * memory is controlled by @ref RTP_BUILDER_PARAM_CHUNK_SIZE.
   *
   * The flag @ref RTP_MODEL_HINT_MASK_UPDATE should be used to inform Prime
   * when visibility mask data changed (after calling rtpModelSetTriangles 
   * with the updated values), e.g. when the indices format RTP_BUFFER_FORMAT_INDICES_INT3_MASK_INT 
   * is used. RTP_MODEL_HINT_MASK_UPDATE can be combined with RTP_MODEL_HINT_ASYNC to
   * perform asynchronous data updates.
   *
   * Hint RTP_MODEL_HINT_USER_TRIANGLES_AFTER_COPY_SET should be used when a triangle 
   * model has been copied (with the user triangle build flag set), and new user triangles
   * have been set (by calling @ref rtpModelSetTriangles again with the updated values).
   * RTP_MODEL_HINT_USER_TRIANGLES_AFTER_COPY_SET can be combined with RTP_MODEL_HINT_ASYNC
   * to perform asynchronous data updates.
   *
   * @param[in] model      Model
   * @param[in] hints      A combination of flags from @ref RTPmodelhint
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   * Example Usage:
   @code
   RTPmodel model;
   rtpModelCreate(context, &model);
   rtpModelSetTriangles(model, 0, vertsBD);
   rtpModelUpdate(model, RTP_MODEL_HINT_ASYNC);

   // ... do useful work on CPU while GPU is busy

   rtpModelFinish(model);

   // It is now safe to modify vertex buffer
   @endcode
  */
  RTPresult RTPAPI rtpModelUpdate( RTPmodel model, unsigned hints );

  /**
   * @brief   Blocks current thread until model update is finished
   *
   * @ingroup Prime_Model
   *
   * This function can be called multiple times. It will return immediately if
   * the previous update has already finished.
   *
   * @param [in] model      Model
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelFinish( RTPmodel model );

  /**
   * @brief   Polls the status of a model update
   *
   * @ingroup Prime_Model
   *
   * @param[in] model        Model
   * @param[out] isFinished  Returns finished status
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelGetFinished( RTPmodel model, /*out*/ int* isFinished );

  /**
   * @brief   Copies one model to another
   *
   * @ingroup Prime_Model
   *
   * This function copies a model from one OptiX Prime
   * context to another for user-managed multi-GPU operation where one context is
   * allocated per device. Only triangle models can be copied, not instance models.
   * Furthermore, when a \a srcModel has the @ref RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES 
   * build parameter set to 1, and it is intended that the triangle data is automatically
   * transfered to the other context, the destination (\a model) should have the build parameter 
   * set to 0 before the copy call. If the destination model also has the has the build parameter set to 1,
   * its triangles must be set by calling @ref rtpModelSetTriangles followed by @ref rtpModelUpdate
   * using RTP_MODEL_HINT_USER_TRIANGLES_AFTER_COPY_SET.
   *
   * @param[in] model        Destination model
   * @param[in] srcModel     Source model
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelCopy( RTPmodel model, RTPmodel srcModel );

  /**
   * @brief   Specifies a builder parameter for a model
   *
   * @ingroup Prime_Model
   *
   * The following builder parameters are supported:
   *
   * @ref RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES : \a int
   *
   * If the value for @ref RTP_BUILDER_PARAM_USE_CALLER_TRIANGLES is
   * set to 0 (default), Prime uses an internal representation for 
   * triangles (which requires additional memory) to improve query
   * performance and does not reference the user's vertex buffer
   * during a query. If set to 1, Prime uses the provided triangle
   * data as-is, which may result in slower query performance, but reduces
   * memory usage.
   *
   *
   * @ref RTP_BUILDER_PARAM_CHUNK_SIZE : \a RTPsize
   *
   * Acceleration structures are built in chunks to reduce the amount
   * of scratch memory needed. The size of the scratch memory chunk is
   * specified in bytes by @ref RTP_BUILDER_PARAM_CHUNK_SIZE. If set
   * to -1, the chunk size has no limit. If set to 0 (default) the
   * chunk size is chosen automatically, currently as 10% of the total
   * available video memory for GPU builds and 512MB for CPU builds.
   *
   *
   * @param[in] model_api  Model
   * @param[in] param      Builder parameter to set
   * @param[in] size       Size in bytes of the parameter being set
   * @param[in] ptr        Pointer to where the value of the attribute will be copied from. This must point to at least \a size bytes of memory
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelSetBuilderParameter( RTPmodel model_api, RTPbuilderparam param, RTPsize size, const void* ptr );

  /**
   * @brief   Destroys a model
   *
   * @ingroup Prime_Model
   *
   * Any queries created on the model are also destroyed with the model. The
   * queries are allowed to finish before they are destroyed.
   *
   * @param[in] model      Model
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpModelDestroy( RTPmodel model );



  /****************************************
   *
   *  QUERY
   *
   *  A query encapsulates a ray tracing computation against a model
   *
   ****************************************/

  /**
   * @brief   Creates a query on a model
   *
   * @ingroup Prime_Query
   *
   * If the model to which a query is bound destroyed with rtpModelDestroy() the
   * query will be destroyed as well.
   *
   * @param[in]  model          Model to use for this query
   * @param[in]  queryType      Type of the query
   * @param[out] query          Pointer to the new query
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQueryCreate( RTPmodel model, RTPquerytype queryType, /*out*/ RTPquery* query );

  /**
   * @brief   Gets the context object associated with a query
   *
   * @ingroup Prime_Query
   *
   * @param[in]  query     Query to obtain the context from
   * @param[out] context   Returned context
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQueryGetContext( RTPquery query, /*out*/ RTPcontext* context );

  /**
   * @brief   Sets the rays buffer for a query
   *
   * @ingroup Prime_Query
   *
   * The rays buffer is not accessed until rtpQueryExecute() is called.
   * The ray directions must be unit length for correct results.
   *
   * @param[in] query      Query
   * @param[in] rays       Buffer descriptor for rays
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQuerySetRays( RTPquery query, RTPbufferdesc rays );

  /**
   * @brief   Sets the hits buffer for a query
   *
   * @ingroup Prime_Query
   *
   * A hit is reported for every ray in the query. Therefore the size of the
   * range in the hit buffer must match that of the ray buffer.
   *
   * @param[in] query      Query
   * @param[in] hits       Buffer descriptor for hits
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQuerySetHits( RTPquery query, RTPbufferdesc hits );

  /**
   * @brief   Executes a raytracing query
   *
   * @ingroup Prime_Query
   *
   * If the flag @ref RTP_QUERY_HINT_ASYNC is specified, rtpQueryExecute may
   * return before the query is actually finished. @ref rtpQueryFinish can be
   * called to block the current thread until the query is finished, or
   * @ref rtpQueryGetFinished can be used to poll until the query is finished.
   *
   * @param[in] query      Query
   * @param[in] hints      A combination of flags from @ref RTPqueryhint
   *
   * Once the query has finished all of the hits are guaranteed to have been
   * returned, and it is safe to modify the ray buffer.
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   * Example Usage:
   @code
   RTPquery query;
   rtpQueryCreate(model, RTP_QUERY_TYPE_CLOSEST, &query);
   rtpQuerySetRays(query, raysBD);
   rtpQuerySetHits(hits, hitsBD);
   rtpQueryExecute(query, 0);
   // safe to modify ray buffer and process hits
   @endcode
  */
  RTPresult RTPAPI rtpQueryExecute( RTPquery query, unsigned hints );

  /**
   * @brief   Blocks current thread until query is finished
   *
   * @ingroup Prime_Query
   *
   * This function can be called multiple times. It will return immediately if
   * the query has already finished.
   *
   * @param[in] query      Query
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQueryFinish( RTPquery query );

  /**
   * @brief   Polls the status of a query
   *
   * @ingroup Prime_Query
   *
   * @param[in]  query        Query
   * @param[out] isFinished   Returns finished status
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQueryGetFinished( RTPquery query, /*out*/ int* isFinished );

  /**
   * @brief   Sets a sync stream for a query
   *
   * @ingroup Prime_Query
   *
   * Specify a Cuda stream used for synchronization. If no stream is specified,
   * the default 0-stream is used. A stream can only be specified for
   * contexts with type @ref RTP_CONTEXT_TYPE_CUDA.
   *
   * @param[in] query      Query
   * @param[in] stream     A cuda stream
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQuerySetCudaStream( RTPquery query, cudaStream_t stream );

  /**
   * @brief   Destroys a query
   *
   * @ingroup Prime_Query
   *
   * The query is finished before it is destroyed
   *
   * @param[in] query      Query to be destroyed
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   * - @ref RTP_ERROR_UNKNOWN
   *
   */
  RTPresult RTPAPI rtpQueryDestroy( RTPquery query );

  /****************************************
   *
   * MISC
   *
   ****************************************/
   
  /**
   * @brief   Page-locks a host buffer
   *
   * @ingroup Prime_Misc
   *
   * Transfers between the host and device are faster if the host buffers are
   * page-locked. However, page-locked memory is a limited resource and should
   * be used judiciously.
   *
   * @param[in] buffer       Buffer on the host
   * @param[in] size         Size of the buffer
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   *
   */
  RTPresult RTPAPI rtpHostBufferLock( void* buffer, RTPsize size );

  /**
   * @brief   Unlocks a previously page-locked host buffer
   *
   * @ingroup Prime_Misc
   *
   * Transfers between the host and device are faster if the host buffers are
   * page-locked. However, page-locked memory is a limited resource and should
   * be used judiciously. Use this function on buffers previous page-locked
   * with @ref rtpHostBufferLock.
   *
   * @param[in] buffer      Buffer on the host
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   *
   */
  RTPresult RTPAPI rtpHostBufferUnlock( void* buffer );

  /**
   * @brief   Translates an RTPresult error code to a string
   *
   * @ingroup Prime_Misc
   *
   * Translates an RTPresult error code to a string describing the error.
   *
   * @param[in]  errorCode      Error code to be translated
   * @param[out] errorString    Returned error string
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   *
   * <B>See also</B>
   * @ref rtpContextGetLastErrorString
   *
   */
  RTPresult RTPAPI rtpGetErrorString( RTPresult errorCode, /*out*/ const char** errorString );

  /**
   * @brief   Gets OptiX Prime version number
   *
   * @ingroup Prime_Misc
   *
   * The encoding for the version number prior to OptiX 4.0.0 is major*1000 + minor*10 + micro.  
   * For versions 4.0.0 and higher, the encoding is major*10000 + minor*100 + micro.
   * For example, for version 3.5.1 this function would return 3051, and for version 4.1.2 it would return 40102.
   *
   * @param[out] version     Returned version
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   * - @ref RTP_ERROR_INVALID_VALUE
   *
   */
  RTPresult RTPAPI rtpGetVersion( /*out*/ unsigned* version );

  /**
   * @brief   Gets OptiX Prime version string
   *
   * @ingroup Prime_Misc
   *
   * Returns OptiX Prime version string and other information in a
   * human-readable format.
   *
   * @param[in] versionString   Returned version information
   *
   * <B>Return values</B>
   *
   * Relevant return values:
   * - @ref RTP_SUCCESS
   *
   */
  RTPresult RTPAPI rtpGetVersionString( const char** versionString );

#ifdef __cplusplus
}
#endif

#endif /* #ifndef __optix_optix_prime_h__ */
