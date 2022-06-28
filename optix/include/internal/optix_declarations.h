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
 * @file   optix_declarations.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API declarations
 *
 * OptiX public API declarations
 */

/******************************************************************************\
 *
 * Contains declarations used by both optix host and device headers.
 *
\******************************************************************************/

#ifndef __optix_optix_declarations_h__
#define __optix_optix_declarations_h__

/************************************
 **
 **    Preprocessor macros 
 **
 ***********************************/

#if defined(__CUDACC__) || defined(__CUDABE__)
#  define RT_HOSTDEVICE __host__ __device__
#else
#  define RT_HOSTDEVICE
#endif


/************************************
 **
 **    Enumerated values
 **
 ***********************************/

#ifdef __cplusplus
extern "C" {
#endif

/*! OptiX formats */
typedef enum
{
  RT_FORMAT_UNKNOWN              = 0x100, /*!< Format unknown       */
  RT_FORMAT_FLOAT,                        /*!< Float                */
  RT_FORMAT_FLOAT2,                       /*!< sizeof(float)*2      */
  RT_FORMAT_FLOAT3,                       /*!< sizeof(float)*3      */
  RT_FORMAT_FLOAT4,                       /*!< sizeof(float)*4      */
  RT_FORMAT_BYTE,                         /*!< BYTE                 */
  RT_FORMAT_BYTE2,                        /*!< sizeof(CHAR)*2       */
  RT_FORMAT_BYTE3,                        /*!< sizeof(CHAR)*3       */
  RT_FORMAT_BYTE4,                        /*!< sizeof(CHAR)*4       */
  RT_FORMAT_UNSIGNED_BYTE,                /*!< UCHAR                */
  RT_FORMAT_UNSIGNED_BYTE2,               /*!< sizeof(UCHAR)*2      */
  RT_FORMAT_UNSIGNED_BYTE3,               /*!< sizeof(UCHAR)*3      */
  RT_FORMAT_UNSIGNED_BYTE4,               /*!< sizeof(UCHAR)*4      */
  RT_FORMAT_SHORT,                        /*!< SHORT                */
  RT_FORMAT_SHORT2,                       /*!< sizeof(SHORT)*2      */
  RT_FORMAT_SHORT3,                       /*!< sizeof(SHORT)*3      */
  RT_FORMAT_SHORT4,                       /*!< sizeof(SHORT)*4      */
  RT_FORMAT_UNSIGNED_SHORT,               /*!< USHORT               */
  RT_FORMAT_UNSIGNED_SHORT2,              /*!< sizeof(USHORT)*2     */
  RT_FORMAT_UNSIGNED_SHORT3,              /*!< sizeof(USHORT)*3     */
  RT_FORMAT_UNSIGNED_SHORT4,              /*!< sizeof(USHORT)*4     */
  RT_FORMAT_INT,                          /*!< INT                  */
  RT_FORMAT_INT2,                         /*!< sizeof(INT)*2        */
  RT_FORMAT_INT3,                         /*!< sizeof(INT)*3        */
  RT_FORMAT_INT4,                         /*!< sizeof(INT)*4        */
  RT_FORMAT_UNSIGNED_INT,                 /*!< sizeof(UINT)         */
  RT_FORMAT_UNSIGNED_INT2,                /*!< sizeof(UINT)*2       */
  RT_FORMAT_UNSIGNED_INT3,                /*!< sizeof(UINT)*3       */
  RT_FORMAT_UNSIGNED_INT4,                /*!< sizeof(UINT)*4       */
  RT_FORMAT_USER,                         /*!< User Format          */
  RT_FORMAT_BUFFER_ID,                    /*!< Buffer Id            */
  RT_FORMAT_PROGRAM_ID,                   /*!< Program Id           */
  RT_FORMAT_HALF,                         /*!< half float           */
  RT_FORMAT_HALF2,                        /*!< sizeof(half float)*2 */
  RT_FORMAT_HALF3,                        /*!< sizeof(half float)*3 */
  RT_FORMAT_HALF4,                        /*!< sizeof(half float)*4 */
  RT_FORMAT_LONG_LONG,                    /*!< LONG_LONG            */
  RT_FORMAT_LONG_LONG2,                   /*!< sizeof(LONG_LONG)*2  */
  RT_FORMAT_LONG_LONG3,                   /*!< sizeof(LONG_LONG)*3  */
  RT_FORMAT_LONG_LONG4,                   /*!< sizeof(LONG_LONG)*4  */
  RT_FORMAT_UNSIGNED_LONG_LONG,           /*!< sizeof(ULONG_LONG)   */
  RT_FORMAT_UNSIGNED_LONG_LONG2,          /*!< sizeof(ULONG_LONG)*2 */
  RT_FORMAT_UNSIGNED_LONG_LONG3,          /*!< sizeof(ULONG_LONG)*3 */
  RT_FORMAT_UNSIGNED_LONG_LONG4,          /*!< sizeof(ULONG_LONG)*4 */
  RT_FORMAT_UNSIGNED_BC1,                 /*!< Block Compressed RGB + optional 1-bit alpha BC1,
                                               sizeof(UINT)*2 */
  RT_FORMAT_UNSIGNED_BC2,                 /*!< Block Compressed RGB + 4-bit alpha BC2,
                                               sizeof(UINT)*4 */
  RT_FORMAT_UNSIGNED_BC3,                 /*!< Block Compressed RGBA BC3,
                                               sizeof(UINT)*4 */
  RT_FORMAT_UNSIGNED_BC4,                 /*!< Block Compressed unsigned grayscale BC4,
                                               sizeof(UINT)*2 */
  RT_FORMAT_BC4,                          /*!< Block Compressed signed   grayscale BC4,
                                               sizeof(UINT)*2 */
  RT_FORMAT_UNSIGNED_BC5,                 /*!< Block Compressed unsigned 2 x grayscale BC5,
                                               sizeof(UINT)*4 */
  RT_FORMAT_BC5,                          /*!< Block compressed signed   2 x grayscale BC5,
                                               sizeof(UINT)*4 */
  RT_FORMAT_UNSIGNED_BC6H,                /*!< Block compressed BC6 unsigned half-float,
                                               sizeof(UINT)*4 */
  RT_FORMAT_BC6H,                         /*!< Block compressed BC6 signed half-float,
                                               sizeof(UINT)*4 */
  RT_FORMAT_UNSIGNED_BC7                  /*!< Block compressed BC7,
                                               sizeof(UINT)*4 */
} RTformat;

/*! OptiX Object Types */
typedef enum
{
  RT_OBJECTTYPE_UNKNOWN          = 0x200,   /*!< Object Type Unknown       */
  RT_OBJECTTYPE_GROUP,                      /*!< Group Type                */
  RT_OBJECTTYPE_GEOMETRY_GROUP,             /*!< Geometry Group Type       */
  RT_OBJECTTYPE_TRANSFORM,                  /*!< Transform Type            */
  RT_OBJECTTYPE_SELECTOR,                   /*!< Selector Type             */
  RT_OBJECTTYPE_GEOMETRY_INSTANCE,          /*!< Geometry Instance Type    */
  RT_OBJECTTYPE_BUFFER,                     /*!< Buffer Type               */
  RT_OBJECTTYPE_TEXTURE_SAMPLER,            /*!< Texture Sampler Type      */
  RT_OBJECTTYPE_OBJECT,                     /*!< Object Type               */
  /* RT_OBJECTTYPE_PROGRAM - see below for entry */

  RT_OBJECTTYPE_MATRIX_FLOAT2x2,            /*!< Matrix Float 2x2          */
  RT_OBJECTTYPE_MATRIX_FLOAT2x3,            /*!< Matrix Float 2x3          */
  RT_OBJECTTYPE_MATRIX_FLOAT2x4,            /*!< Matrix Float 2x4          */
  RT_OBJECTTYPE_MATRIX_FLOAT3x2,            /*!< Matrix Float 3x2          */
  RT_OBJECTTYPE_MATRIX_FLOAT3x3,            /*!< Matrix Float 3x3          */
  RT_OBJECTTYPE_MATRIX_FLOAT3x4,            /*!< Matrix Float 3x4          */
  RT_OBJECTTYPE_MATRIX_FLOAT4x2,            /*!< Matrix Float 4x2          */
  RT_OBJECTTYPE_MATRIX_FLOAT4x3,            /*!< Matrix Float 4x3          */
  RT_OBJECTTYPE_MATRIX_FLOAT4x4,            /*!< Matrix Float 4x4          */

  RT_OBJECTTYPE_FLOAT,                      /*!< Float Type                    */
  RT_OBJECTTYPE_FLOAT2,                     /*!< Float2 Type                   */
  RT_OBJECTTYPE_FLOAT3,                     /*!< Float3 Type                   */
  RT_OBJECTTYPE_FLOAT4,                     /*!< Float4 Type                   */
  RT_OBJECTTYPE_INT,                        /*!< 32 Bit Integer Type           */
  RT_OBJECTTYPE_INT2,                       /*!< 32 Bit Integer2 Type          */
  RT_OBJECTTYPE_INT3,                       /*!< 32 Bit Integer3 Type          */
  RT_OBJECTTYPE_INT4,                       /*!< 32 Bit Integer4 Type          */
  RT_OBJECTTYPE_UNSIGNED_INT,               /*!< 32 Bit Unsigned Integer Type  */
  RT_OBJECTTYPE_UNSIGNED_INT2,              /*!< 32 Bit Unsigned Integer2 Type */
  RT_OBJECTTYPE_UNSIGNED_INT3,              /*!< 32 Bit Unsigned Integer3 Type */
  RT_OBJECTTYPE_UNSIGNED_INT4,              /*!< 32 Bit Unsigned Integer4 Type */
  RT_OBJECTTYPE_USER,                       /*!< User Object Type              */

  RT_OBJECTTYPE_PROGRAM,                    /*!< Object Type Program - Added in OptiX 3.0              */
  RT_OBJECTTYPE_COMMANDLIST,                /*!< Object Type Command List - Added in OptiX 5.0         */
  RT_OBJECTTYPE_POSTPROCESSINGSTAGE,        /*!< Object Type Postprocessing Stage - Added in OptiX 5.0 */
  
  RT_OBJECTTYPE_LONG_LONG,                  /*!< 64 Bit Integer Type - Added in Optix 6.0              */
  RT_OBJECTTYPE_LONG_LONG2,                 /*!< 64 Bit Integer2 Type - Added in Optix 6.0             */
  RT_OBJECTTYPE_LONG_LONG3,                 /*!< 64 Bit Integer3 Type - Added in Optix 6.0             */
  RT_OBJECTTYPE_LONG_LONG4,                 /*!< 64 Bit Integer4 Type - Added in Optix 6.0             */
  RT_OBJECTTYPE_UNSIGNED_LONG_LONG,         /*!< 64 Bit Unsigned Integer Type - Added in Optix 6.0     */
  RT_OBJECTTYPE_UNSIGNED_LONG_LONG2,        /*!< 64 Bit Unsigned Integer2 Type - Added in Optix 6.0    */
  RT_OBJECTTYPE_UNSIGNED_LONG_LONG3,        /*!< 64 Bit Unsigned Integer3 Type - Added in Optix 6.0    */
  RT_OBJECTTYPE_UNSIGNED_LONG_LONG4         /*!< 64 Bit Unsigned Integer4 Type - Added in Optix 6.0    */
} RTobjecttype;


/*! Wrap mode */
typedef enum
{
  RT_WRAP_REPEAT,           /*!< Wrap repeat     */
  RT_WRAP_CLAMP_TO_EDGE,    /*!< Clamp to edge   */
  RT_WRAP_MIRROR,           /*!< Mirror          */
  RT_WRAP_CLAMP_TO_BORDER   /*!< Clamp to border */
} RTwrapmode;

/*! Filter mode */
typedef enum
{
  RT_FILTER_NEAREST,      /*!< Nearest     */
  RT_FILTER_LINEAR,       /*!< Linear      */
  RT_FILTER_NONE          /*!< No filter   */
} RTfiltermode;

/*! Texture read mode */
typedef enum
{
  RT_TEXTURE_READ_ELEMENT_TYPE = 0,           /*!< Read element type                                                                                              */
  RT_TEXTURE_READ_NORMALIZED_FLOAT = 1,       /*!< Read normalized float                                                                                          */
  RT_TEXTURE_READ_ELEMENT_TYPE_SRGB = 2,      /*!< Read element type and apply sRGB to linear conversion during texture read for 8-bit integer buffer formats     */
  RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB = 3   /*!< Read normalized float and apply sRGB to linear conversion during texture read for 8-bit integer buffer formats */
} RTtexturereadmode;

/*! GL Target */
typedef enum
{
  RT_TARGET_GL_TEXTURE_2D,            /*!< GL texture 2D           */
  RT_TARGET_GL_TEXTURE_RECTANGLE,     /*!< GL texture rectangle    */
  RT_TARGET_GL_TEXTURE_3D,            /*!< GL texture 3D           */
  RT_TARGET_GL_RENDER_BUFFER,         /*!< GL render buffer        */
  RT_TARGET_GL_TEXTURE_1D,            /*!< GL texture 1D           */
  RT_TARGET_GL_TEXTURE_1D_ARRAY,      /*!< GL array of 1D textures */
  RT_TARGET_GL_TEXTURE_2D_ARRAY,      /*!< GL array of 2D textures */
  RT_TARGET_GL_TEXTURE_CUBE_MAP,      /*!< GL cube map texture     */
  RT_TARGET_GL_TEXTURE_CUBE_MAP_ARRAY /*!< GL array of cube maps   */
} RTgltarget;

/*! Texture index mode */
typedef enum
{
  RT_TEXTURE_INDEX_NORMALIZED_COORDINATES,    /*!< Texture Index normalized coordinates */
  RT_TEXTURE_INDEX_ARRAY_INDEX                /*!< Texture Index Array */
} RTtextureindexmode;

/*! Buffer type */
typedef enum
{
  RT_BUFFER_INPUT                = 0x1,                               /*!< Input buffer for the GPU          */
  RT_BUFFER_OUTPUT               = 0x2,                               /*!< Output buffer for the GPU         */
  RT_BUFFER_INPUT_OUTPUT         = RT_BUFFER_INPUT | RT_BUFFER_OUTPUT,/*!< Ouput/Input buffer for the GPU    */
  RT_BUFFER_PROGRESSIVE_STREAM   = 0x10,                              /*!< Progressive stream buffer         */
} RTbuffertype;

/*! Buffer flags */
typedef enum
{
  RT_BUFFER_GPU_LOCAL            = 0x4,  /*!< An @ref RT_BUFFER_INPUT_OUTPUT has separate copies on each device that are not synchronized                               */
  RT_BUFFER_COPY_ON_DIRTY        = 0x8,  /*!< A CUDA Interop buffer will only be synchronized across devices when dirtied by @ref rtBufferMap or @ref rtBufferMarkDirty */
  RT_BUFFER_DISCARD_HOST_MEMORY  = 0x20, /*!< An @ref RT_BUFFER_INPUT for which a synchronize is forced on unmapping from host and the host memory is freed */
  RT_BUFFER_LAYERED              = 0x200000, /*!< Depth specifies the number of layers, not the depth of a 3D array */
  RT_BUFFER_CUBEMAP              = 0x400000, /*!< Enables creation of cubemaps. If this flag is set, Width must be equal to Height, and Depth must be six. If the @ref RT_BUFFER_LAYERED flag is also set, then Depth must be a multiple of six */
} RTbufferflag;

/*! Buffer mapping flags */
typedef enum
{
  RT_BUFFER_MAP_READ            = 0x1, /*!< Map buffer memory for reading */  
  RT_BUFFER_MAP_READ_WRITE      = 0x2, /*!< Map buffer memory for both reading and writing */
  RT_BUFFER_MAP_WRITE           = 0x4, /*!< Map buffer memory for writing */
  RT_BUFFER_MAP_WRITE_DISCARD   = 0x8  /*!< Map buffer memory for writing, with the previous contents being undefined*/
} RTbuffermapflag;

/** Exceptions
  *
  * <B>See also</B>
  * @ref rtContextSetExceptionEnabled,
  * @ref rtContextGetExceptionEnabled,
  * @ref rtGetExceptionCode,
  * @ref rtThrow,
  * @ref rtPrintf
  */
typedef enum
{
  RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS      = 0x3EB,    /*!< Payload access out of bounds - Added in OptiX 6.0                   */
  RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS = 0x3EC,    /*!< Exception code of user exception out of bounds - Added in OptiX 6.0 */
  RT_EXCEPTION_TRACE_DEPTH_EXCEEDED              = 0x3ED,    /*!< Trace depth exceeded - Added in Optix 6.0                           */
  RT_EXCEPTION_PROGRAM_ID_INVALID                = 0x3EE,    /*!< Program ID not valid                                                */
  RT_EXCEPTION_TEXTURE_ID_INVALID                = 0x3EF,    /*!< Texture ID not valid                                                */
  RT_EXCEPTION_BUFFER_ID_INVALID                 = 0x3FA,    /*!< Buffer ID not valid                                                 */
  RT_EXCEPTION_INDEX_OUT_OF_BOUNDS               = 0x3FB,    /*!< Index out of bounds                                                 */
  RT_EXCEPTION_STACK_OVERFLOW                    = 0x3FC,    /*!< Stack overflow                                                      */
  RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS        = 0x3FD,    /*!< Buffer index out of bounds                                          */
  RT_EXCEPTION_INVALID_RAY                       = 0x3FE,    /*!< Invalid ray                                                         */
  RT_EXCEPTION_INTERNAL_ERROR                    = 0x3FF,    /*!< Internal error                                                      */
  RT_EXCEPTION_USER                              = 0x400,    /*!< First user exception code                                           */
  RT_EXCEPTION_USER_MAX                          = 0xFFFF,   /*!< Last user exception code                                            */

  RT_EXCEPTION_ALL                               = 0x7FFFFFFF  /*!< All exceptions */
} RTexception;

/*! Result */
typedef enum
{
    RT_SUCCESS = 0, /*!< Success                      */

    RT_TIMEOUT_CALLBACK = 0x100, /*!< Timeout callback             */

    RT_ERROR_INVALID_CONTEXT          = 0x500, /*!< Invalid Context              */
    RT_ERROR_INVALID_VALUE            = 0x501, /*!< Invalid Value                */
    RT_ERROR_MEMORY_ALLOCATION_FAILED = 0x502, /*!< Timeout callback             */
    RT_ERROR_TYPE_MISMATCH            = 0x503, /*!< Type Mismatch                */
    RT_ERROR_VARIABLE_NOT_FOUND       = 0x504, /*!< Variable not found           */
    RT_ERROR_VARIABLE_REDECLARED      = 0x505, /*!< Variable redeclared          */
    RT_ERROR_ILLEGAL_SYMBOL           = 0x506, /*!< Illegal symbol               */
    RT_ERROR_INVALID_SOURCE           = 0x507, /*!< Invalid source               */
    RT_ERROR_VERSION_MISMATCH         = 0x508, /*!< Version mismatch             */

    RT_ERROR_OBJECT_CREATION_FAILED  = 0x600, /*!< Object creation failed       */
    RT_ERROR_NO_DEVICE               = 0x601, /*!< No device                    */
    RT_ERROR_INVALID_DEVICE          = 0x602, /*!< Invalid device               */
    RT_ERROR_INVALID_IMAGE           = 0x603, /*!< Invalid image                */
    RT_ERROR_FILE_NOT_FOUND          = 0x604, /*!< File not found               */
    RT_ERROR_ALREADY_MAPPED          = 0x605, /*!< Already mapped               */
    RT_ERROR_INVALID_DRIVER_VERSION  = 0x606, /*!< Invalid driver version       */
    RT_ERROR_CONTEXT_CREATION_FAILED = 0x607, /*!< Context creation failed      */
    RT_ERROR_RESOURCE_NOT_REGISTERED     = 0x608, /*!< Resource not registered           */
    RT_ERROR_RESOURCE_ALREADY_REGISTERED = 0x609, /*!< Resource already registered       */
    RT_ERROR_OPTIX_NOT_LOADED            = 0x60A, /*!< OptiX DLL failed to load          */
    RT_ERROR_DENOISER_NOT_LOADED         = 0x60B, /*!< Denoiser DLL failed to load       */
    RT_ERROR_SSIM_PREDICTOR_NOT_LOADED   = 0x60C, /*!< SSIM predictor DLL failed to load */
    RT_ERROR_DRIVER_VERSION_FAILED       = 0x60D, /*!< Driver version retrieval failed   */
    RT_ERROR_DATABASE_FILE_PERMISSIONS   = 0x60E, /*!< No write permission on disk cache file */

    RT_ERROR_LAUNCH_FAILED = 0x900, /*!< Launch failed                */

    RT_ERROR_NOT_SUPPORTED = 0xA00, /*!< Not supported                */

    RT_ERROR_CONNECTION_FAILED         = 0xB00, /*!< Connection failed            */
    RT_ERROR_AUTHENTICATION_FAILED     = 0xB01, /*!< Authentication failed        */
    RT_ERROR_CONNECTION_ALREADY_EXISTS = 0xB02, /*!< Connection already exists    */
    RT_ERROR_NETWORK_LOAD_FAILED       = 0xB03, /*!< Network component failed to load */
    RT_ERROR_NETWORK_INIT_FAILED       = 0xB04, /*!< Network initialization failed*/
    RT_ERROR_CLUSTER_NOT_RUNNING       = 0xB06, /*!< No cluster is running        */
    RT_ERROR_CLUSTER_ALREADY_RUNNING   = 0xB07, /*!< Cluster is already running   */
    RT_ERROR_INSUFFICIENT_FREE_NODES   = 0xB08, /*!< Not enough free nodes        */

    RT_ERROR_INVALID_GLOBAL_ATTRIBUTE = 0xC00, /*!< Invalid global attribute     */

    RT_ERROR_UNKNOWN = ~0 /*!< Error unknown                */
} RTresult;

/*! Device attributes */
typedef enum
{
  RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,      /*!< Max Threads per Block sizeof(int) */
  RT_DEVICE_ATTRIBUTE_CLOCK_RATE,                 /*!< Clock rate sizeof(int) */
  RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,       /*!< Multiprocessor count sizeof(int) */
  RT_DEVICE_ATTRIBUTE_EXECUTION_TIMEOUT_ENABLED,  /*!< Execution timeout enabled sizeof(int) */
  RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, /*!< Hardware Texture count sizeof(int) */
  RT_DEVICE_ATTRIBUTE_NAME,                       /*!< Attribute Name */
  RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY,         /*!< Compute Capabilities sizeof(int2) */
  RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY,               /*!< Total Memory sizeof(RTsize) */
  RT_DEVICE_ATTRIBUTE_TCC_DRIVER,                 /*!< TCC driver sizeof(int) */
  RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL,        /*!< CUDA device ordinal sizeof(int) */
  RT_DEVICE_ATTRIBUTE_PCI_BUS_ID,                 /*!< PCI Bus Id */
  RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES,         /*!< Ordinals of compatible devices sizeof(int=N) + N*sizeof(int) */
  RT_DEVICE_ATTRIBUTE_RTCORE_VERSION              /*!< RT core version (0 for no support, 10 for version 1.0) sizeof(int) */
} RTdeviceattribute;

/*! Global attributes */
typedef enum
{
  RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR=1,                /*!< sizeof(int)    */
  RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR,                  /*!< sizeof(int)    */
  RT_GLOBAL_ATTRIBUTE_ENABLE_RTX = 0x10000000,                       /*!< sizeof(int)    */
  RT_GLOBAL_ATTRIBUTE_DEVELOPER_OPTIONS                              /*!< Knobs string */
} RTglobalattribute;

/*! Context attributes */
typedef enum
{
  RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT,                    /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS,                      /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY,                     /*!< sizeof(RTsize)    */
  RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE,                    /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF,                /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_DISK_CACHE_ENABLED,                   /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES,               /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS,          /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_OPTIX_SALT,                           /*!< 32                */
  RT_CONTEXT_ATTRIBUTE_VENDOR_SALT,                          /*!< 32                */
  RT_CONTEXT_ATTRIBUTE_PUBLIC_VENDOR_KEY,                    /*!< variable          */
  RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION,                  /*!< sizeof(char*)     */
  RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS,             /*!< sizeof(RTsize[2]) */
  RT_CONTEXT_ATTRIBUTE_PREFER_WATERTIGHT_TRAVERSAL,          /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_MAX_CONCURRENT_LAUNCHES,              /*!< sizeof(int)       */
  RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY = 0x10000000  /*!< sizeof(RTsize)    */
} RTcontextattribute;

/*! Buffer attributes */
typedef enum
{
  RT_BUFFER_ATTRIBUTE_STREAM_FORMAT,                          /*!< Format string */
  RT_BUFFER_ATTRIBUTE_STREAM_BITRATE,                         /*!< sizeof(int) */
  RT_BUFFER_ATTRIBUTE_STREAM_FPS,                             /*!< sizeof(int) */
  RT_BUFFER_ATTRIBUTE_STREAM_GAMMA,                           /*!< sizeof(float) */
  RT_BUFFER_ATTRIBUTE_PAGE_SIZE                               /*!< sizeof(int) */
} RTbufferattribute;

/*! Motion border modes*/
typedef enum
{
  RT_MOTIONBORDERMODE_CLAMP,                      /*!< Clamp outside of bounds  */
  RT_MOTIONBORDERMODE_VANISH                      /*!< Vanish outside of bounds */
} RTmotionbordermode;

/*! Motion key type */
typedef enum
{
  RT_MOTIONKEYTYPE_NONE = 0,                     /*!< No motion keys set               */
  RT_MOTIONKEYTYPE_MATRIX_FLOAT12,               /*!< Affine matrix format - 12 floats */
  RT_MOTIONKEYTYPE_SRT_FLOAT16                   /*!< SRT format - 16 floats           */
} RTmotionkeytype;

/*! GeometryX build flags */
typedef enum {
  RT_GEOMETRY_BUILD_FLAG_NONE            = 0x00, /*!< No special flags set */
  RT_GEOMETRY_BUILD_FLAG_RELEASE_BUFFERS = 0x10, /*!< User buffers are released after consumption by acceleration structure build */
} RTgeometrybuildflags;

/*! Material-dependent flags set on Geometry/GeometryTriangles */
typedef enum {
  RT_GEOMETRY_FLAG_NONE            = 0x00, /*!< No special flags set */
  RT_GEOMETRY_FLAG_DISABLE_ANYHIT  = 0x01, /*!< Disable any-hit program execution (execution will be skipped,including the no-op any-hit program
                                                used when an any-hit program is not specified).
                                                Can be overridden by ray and instance flags, precedence: RTrayflags > RTinstanceflags > RTgeometryflags */
  RT_GEOMETRY_FLAG_NO_SPLITTING    = 0x02, /*!< Disable primitive splitting to avoid potential multiple any-hit program execution for a single intersection */
} RTgeometryflags;

/*! Instance flags which override the behavior of geometry. */
typedef enum {
  RT_INSTANCE_FLAG_NONE                     = 0u,       /*!< No special flag set */
  RT_INSTANCE_FLAG_DISABLE_TRIANGLE_CULLING = 1u << 0,  /*!< Prevent triangles from getting culled due to face orientation (overrides ray culling flags). */
  RT_INSTANCE_FLAG_FLIP_TRIANGLE_FACING     = 1u << 1,  /*!< Flip triangle orientation. This affects front/back face culling. */
  RT_INSTANCE_FLAG_DISABLE_ANYHIT           = 1u << 2,  /*!< Disable any-hit program execution (including the no-op any-hit program
                                                             used when an any-hit program is not specified).
                                                             This may yield significantly higher performance even in cases
                                                             where no any-hit programs are set.
                                                             Mutually exclusive with RT_INSTANCE_FLAG_FORCE_ANYHIT.
                                                             If set, overrides any potentially set @ref RT_RAY_FLAG_FORCE_ANYHIT, @ref RT_RAY_FLAG_DISABLE_ANYHIT, @ref RT_GEOMETRY_FLAG_DISABLE_ANYHIT.
                                                             Can be overridden by ray flag @ref RT_RAY_FLAG_FORCE_ANYHIT.
                                                             Precedence: RTrayflags > RTinstanceflags > RTgeometryflags */
  RT_INSTANCE_FLAG_FORCE_ANYHIT             = 1u << 3   /*!< Force any-hit program execution.
                                                             Mutually exclusive with RT_INSTANCE_FLAG_DISABLE_ANYHIT.
                                                             If set, overrides any potentially set @ref RT_RAY_FLAG_FORCE_ANYHIT, @ref RT_RAY_FLAG_DISABLE_ANYHIT, @ref RT_GEOMETRY_FLAG_DISABLE_ANYHIT.
                                                             Can be overridden by ray flag @ref RT_RAY_FLAG_DISABLE_ANYHIT.
                                                             Overriding precedence: RTrayflags > RTinstanceflags > RTgeometryflags */
} RTinstanceflags;

/*! Ray flags */
typedef enum {
  RT_RAY_FLAG_NONE                          = 0u,
  RT_RAY_FLAG_DISABLE_ANYHIT                = 1u << 0, /*!< Disable any-hit program execution for the ray (execution will be skipped,including the no-op any-hit program
                                                            used when an any-hit program is not specified).
                                                            Mutually exclusive with RT_RAY_FLAG_FORCE_ANYHIT.
                                                            If set, overrides any potentially set @ref RT_INSTANCE_FLAG_FORCE_ANYHIT.
                                                            Overriding precedence: RTrayflags > RTinstanceflags > RTgeometryflags */
  RT_RAY_FLAG_FORCE_ANYHIT                  = 1u << 1, /*!< Force any-hit program execution for the ray. See @ref RT_RAY_FLAG_DISABLE_ANYHIT.
                                                            Mutually exclusive with RT_RAY_FLAG_DISABLE_ANYHIT.
                                                            If set, overrides any potentially set @ref RT_GEOMETRY_FLAG_DISABLE_ANYHIT, @ref RT_INSTANCE_FLAG_DISABLE_ANYHIT.
                                                            Overriding precedence: RTrayflags > RTinstanceflags > RTgeometryflags */
  RT_RAY_FLAG_TERMINATE_ON_FIRST_HIT        = 1u << 2, /*!< Terminate the ray after the first hit, also reports the first hit as closest hit. */
  RT_RAY_FLAG_DISABLE_CLOSESTHIT            = 1u << 3, /*!< Disable closest-hit program execution for the ray. */
  RT_RAY_FLAG_CULL_BACK_FACING_TRIANGLES    = 1u << 4, /*!< Do not intersect triangle back faces. */
  RT_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES   = 1u << 5, /*!< Do not intersect triangle front faces. */
  RT_RAY_FLAG_CULL_DISABLED_ANYHIT          = 1u << 6, /*!< Do not intersect geometry which disables any-hit programs (due to any geometry, instance, or ray flag). */
  RT_RAY_FLAG_CULL_ENABLED_ANYHIT           = 1u << 7  /*!< Do not intersect geometry which executes any-hit programs (i.e., forced or not disabled any-hit program execution, this includes a potential no-op any-hit program). */
} RTrayflags;

typedef unsigned int RTvisibilitymask;

enum {
  RT_VISIBILITY_ALL = 0xFFu             /*!< Default @ref RTvisibilitymask */
};

/*! Sentinel values */
typedef enum { 
  RT_BUFFER_ID_NULL       = 0 /*!< sentinel for describing a non-existent buffer id  */ 
} RTbufferidnull;
typedef enum {
  RT_PROGRAM_ID_NULL      = 0 /*!< sentinel for describing a non-existent program id */ 
} RTprogramidnull;
typedef enum {
  RT_TEXTURE_ID_NULL      = 0 /*!< sentinel for describing a non-existent texture id */ 
} RTtextureidnull;
typedef enum {
  RT_COMMAND_LIST_ID_NULL = 0 /*!< sentinel for describing a non-existent command list id */ 
} RTcommandlistidnull;
typedef enum {
  RT_POSTPROCESSING_STAGE_ID_NULL = 0 /*!< sentinel for describing a non-existent post-processing stage id */
} RTpostprocessingstagenull;

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* __optix_optix_declarations_h__ */
