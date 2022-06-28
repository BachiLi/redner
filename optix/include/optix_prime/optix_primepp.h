
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

///
/// \file optix_primepp.h
/// \brief A C++ wrapper around the OptiX Prime API.
/// 

#ifndef __optix_optix_primepp_h__
#define __optix_optix_primepp_h__

#include <string>
#include <vector>

#include "optix_prime.h"
#include "internal/Handle.h"

namespace optix {
  namespace prime {

    /****************************************
     *
     * FORWARD DECLARATIONS
     *
     ****************************************/

    class BufferDescObj;
    class ContextObj;
    class ModelObj;
    class QueryObj;

    /****************************************
     *
     * HANDLE - TYPEDEFS
     *
     ****************************************/

    /// \ingroup optixprimepp
    /// @{
    typedef Handle<BufferDescObj> BufferDesc; ///< Use this to manipulate RTPbufferdesc objects.
    typedef Handle<ContextObj>    Context;    ///< Use this to manipulate RTPcontext objects.
    typedef Handle<ModelObj>      Model;      ///< Use this to manipulate RTPmodel objects.
    typedef Handle<QueryObj>      Query;      ///< Use this to manipulate RTPquery objects.
    /// @}


    /****************************************
     *
     * GLOBAL FUNCTIONS
     *
     ****************************************/

    /// Returns a string describing the version of the OptiX Prime being used. See @ref rtpGetVersionString
    std::string getVersionString();

    /****************************************
     *
     * REFERENCE COUNTED API - OBJECTS
     *
     ****************************************/

    /// \ingroup optixprimepp
    ///
    /// \brief Wraps the OptiX Prime C API @ref RTPcontext opaque type and its associated function set representing an OptiX Prime context.
    ///
    class ContextObj : public RefCountedObj {
    public:
      //
      // OBJECT CREATION - FUNCTIONS
      //

      /// Creates a Context object.  See @ref rtpContextCreate
      static Context create( RTPcontexttype type );
      
      /// Creates a BufferDesc object.  See @ref rtpBufferDescCreate
      BufferDesc createBufferDesc( RTPbufferformat format, RTPbuffertype type, void* buffer );
      
      /// Creates a Model object.  See @ref rtpModelCreate
      Model createModel();
      
      //
      // API-FUNCTIONS
      //

      /// Sets the CUDA devices used by a context.  See @ref rtpContextSetCudaDeviceNumbers
      /// Note that this distribution can be rather costly if the rays are stored in device memory though.
      /// For maximum efficiency it is recommended to only ever select one device per context.
      void setCudaDeviceNumbers( const std::vector<unsigned>& deviceNumbers );

      /// Sets the CUDA devices used by a context.  See @ref rtpContextSetCudaDeviceNumbers
      /// Note that this distribution can be rather costly if the rays are stored in device memory though.
      /// For maximum efficiency it is recommended to only ever select one device per context.
      void setCudaDeviceNumbers( unsigned deviceCount, const unsigned* deviceNumbers );

      /// Sets the number of CPU threads used by a CPU context. See @ref rtpContextSetCpuThreads
      void setCpuThreads( unsigned numThreads );

      /// Returns a string describing last error encountered. See @ref rtpContextGetLastErrorString
      std::string getLastErrorString();
    
      /// Returns the @ref RTPcontext context stored within this object.
      RTPcontext getRTPcontext();

    private:

      friend class QueryObj;
      friend class ModelObj;
      friend class BufferDescObj;

      Context getContext();

      ContextObj( RTPcontexttype type );
      ~ContextObj();
      operator Context();

      RTPcontext m_ctx;
    };

    /// \ingroup optixprimepp
    ///
    /// \brief Encapsulates an OptiX Prime buffer descriptor. The purpose of a buffer descriptor is to provide information about a buffer's type, format,
    /// and location. It also describes the region of the buffer to use.
    class BufferDescObj : public RefCountedObj {
    public:
      //
      // API-FUNCTIONS
      //

      /// Returns the context associated within this object.
      Context getContext();

      /// Sets the range of a buffer to be used. See @ref rtpBufferDescSetRange
      void setRange( RTPsize begin, RTPsize end );

      /// Sets the stride for elements in a buffer. See @ref rtpBufferDescSetStride
      void setStride( unsigned strideBytes );

      /// Sets the CUDA device number for a buffer. See @ref rtpBufferDescSetCudaDeviceNumber
      void setCudaDeviceNumber( unsigned deviceNumber );

      /// Returns the @ref RTPbufferdesc descriptor stored within this object.
      RTPbufferdesc getRTPbufferdesc();

    private:

      friend class ContextObj;
      friend class ModelObj;
      friend class QueryObj;

      BufferDescObj( const Context& ctx, RTPbufferformat format, RTPbuffertype type, void* buffer );
      ~BufferDescObj();

      RTPbufferdesc m_desc;

      Context m_ctx;
    };

    /// \ingroup optixprimepp
    ///
    /// \brief Encapsulates an OptiX Prime model. The purpose of a model is to represent a set of triangles and an acceleration structure.
    class ModelObj : public RefCountedObj {
    public:
      //
      // OBJECT CREATION - FUNCTIONS
      //

      /// Creates a Query object.  See @ref rtpQueryCreate
      Query createQuery( RTPquerytype queryType );

      //
      // API-FUNCTIONS
      //

      /// Returns the context associated within this object.
      Context getContext();

      /// Blocks current thread until model update is finished. See @ref rtpModelFinish
      void finish();
      
      /// Polls the status of a model update. See @ref rtpModelGetFinished
      int  isFinished();

      /// Creates the acceleration structure over the triangles. See @ref rtpModelUpdate
      void update( unsigned hints );

      /// Copies one model to another. See @ref rtpModelCopy
      void copy( const Model& srcModel );

      /// Sets the triangle data for a model. This function creates a buffer descriptor of the specified type, populates it with the supplied data and assigns it to the model.
      /// The list of vertices is assumed to be a flat list of triangles and each three vertices form a single triangle. See @ref rtpModelSetTriangles for additional information
      void setTriangles( RTPsize triCount, RTPbuffertype type, const void* vertPtr, unsigned stride=0 );
      
      /// Sets the triangle data for a model. This function creates buffer descriptors of the specified types, populates them with the supplied data and assigns them to the model.
      /// The list of vertices uses the indices list to determine the triangles. See @ref rtpModelSetTriangles for additional information
      void setTriangles( RTPsize triCount, RTPbuffertype type, const  void* indexPtr, RTPsize vertCount, RTPbuffertype vertType, const void* vertPtr, unsigned stride=0 );

      /// Sets the triangle data for a model using the supplied buffer descriptor of vertices. The list of vertices is assumed to be a flat list of triangles and each three vertices shape a single triangle.
      /// See @ref rtpModelSetTriangles for additional information
      void setTriangles( const BufferDesc& vertices );
      
      /// Sets the triangle data for a model using the supplied buffer descriptor of vertices. The list of vertices uses the indices list to determine the triangles.
      /// See @ref rtpModelSetTriangles for additional information
      void setTriangles( const BufferDesc& indices, const BufferDesc& vertices );

      /// Sets the instance data for a model. This function creates buffer descriptors of the specified types and formats, populates them with the supplied data and assigns them to the model.
      /// See @ref rtpModelSetInstances for additional information
      void setInstances(RTPsize count, RTPbuffertype instanceType, const RTPmodel* instanceList, RTPbufferformat transformFormat, RTPbuffertype transformType, const void* transformList);

      /// Sets the instance data for a model using the supplied buffer descriptors.
      /// See @ref rtpModelSetInstances for additional information
      void setInstances( const BufferDesc& instances, const BufferDesc& transforms );

      /// Sets a model build parameter
      /// See @ref rtpModelSetBuilderParameter for additional information
      void setBuilderParameter( RTPbuilderparam param, RTPsize size, const void* p );

      /// Sets a model build parameter
      /// See @ref rtpModelSetBuilderParameter for additional information
      template<typename T> 
      void setBuilderParameter( RTPbuilderparam param, const T& val );

      /// Returns the @ref RTPmodel model stored within this object.
      RTPmodel getRTPmodel() { return m_model; }

    private:

      friend class ContextObj;
      friend class QueryObj;

      ModelObj( const Context& ctx );
      ~ModelObj();
      operator Model();

      Context m_ctx;
      RTPmodel m_model;
    };



    /// \ingroup optixprimepp
    ///
    /// \brief Encapsulates an OptiX Prime query. The purpose of a query is to coordinate the intersection of rays with a model.
    class QueryObj : public RefCountedObj {
    public:
      //
      // API-FUNCTIONS
      //

      /// Returns the context associated within this object.
      Context getContext();

      /// Blocks current thread until query is finished. See @ref rtpQueryFinish
      void finish();
      /// Polls the status of a query. See @ref rtpQueryGetFinished
      int  isFinished();

      /// Sets a stream for a query. See @ref rtpQuerySetCudaStream
      void setCudaStream( cudaStream_t stream );

      /// Creates a buffer descriptor and sets the rays of a query. See @ref rtpQuerySetRays
      void setRays( RTPsize count, RTPbufferformat format, RTPbuffertype type, void* rays );
      
      /// Sets the rays of a query from a buffer descriptor. See @ref rtpQuerySetRays
      void setRays( const BufferDesc& rays );

      /// Sets a hit buffer for the query. See @ref rtpQuerySetHits
      void setHits( RTPsize count, RTPbufferformat format, RTPbuffertype type, void* hits );
      
      /// Sets a hit buffer for the query from a buffer description. See @ref rtpQuerySetHits
      void setHits( const BufferDesc& hits );

      /// Executes a raytracing query. See @ref rtpQueryExecute
      void execute( unsigned hint );

      /// Returns the @ref RTPquery query stored within this object.
      RTPquery getRTPquery() { return m_query; }

    private:

      friend class ContextObj;
      friend class ModelObj;

      QueryObj( const Model& model, RTPquerytype queryType );
      ~QueryObj();

      Model m_model;
      RTPquery m_query;
    };

    /****************************************
     *
     * EXCEPTION
     *
     ****************************************/

    /// \ingroup optixprimepp
    ///
    /// \brief Encapsulates an OptiX Prime exception.
    class Exception : public std::exception {
    public:
      /// Returns a string describing last error encountered. See @ref rtpGetErrorString
      static Exception makeException( RTPresult code );

      /// Returns a string describing last error encountered. See @ref rtpContextGetLastErrorString
      static Exception makeException( RTPresult code, RTPcontext context );

      virtual ~Exception() throw() {}

      /// Stores the @ref RTPresult error code for this exception
      RTPresult getErrorCode() const;

      /// Stores the human-readable error string associated with this exception
      const std::string& getErrorString() const;

      virtual const char* what() const throw();

    private:
      Exception( const std::string& message, RTPresult error_code = RTP_ERROR_UNKNOWN );

      std::string m_errorMessage;
      RTPresult   m_errorCode;
    };

    ///////////////////////////////////////////////////////////////////////////////
    //                                                                           //
    //                                IMPLEMENTATION                             //
    //                                                                           //
    ///////////////////////////////////////////////////////////////////////////////

    //
    // HELPER - FUNCTIONS AND MACROS
    //

    ///
    void checkError( RTPresult code );
    void checkError( RTPresult code, RTPcontext context );
    #define CHK( code ) checkError( code, getContext()->getRTPcontext() )

    //
    // GLOBALS
    //
    
    inline std::string getVersionString()
    {
      const char* versionString;
      checkError( rtpGetVersionString( &versionString ) );
      return versionString;
    }


    //
    // CONTEXT
    //

    inline Context ContextObj::create( RTPcontexttype type )
    {
      Context h( new ContextObj(type) );
      return h;
    }

    inline BufferDesc ContextObj::createBufferDesc( RTPbufferformat format, RTPbuffertype type, void* buffer )
    {
      BufferDesc h( new BufferDescObj(*this, format, type, buffer) );
      return h;
    }
    
    inline Model ContextObj::createModel()
    {
      Model h( new ModelObj(*this) );
      return h;
    }
    
    inline ContextObj::ContextObj( RTPcontexttype type ) : m_ctx(0) {
      RTPresult r = rtpContextCreate( type, &m_ctx );
      if( r!=RTP_SUCCESS )
        m_ctx = 0;

      checkError( r, m_ctx );
    }

    inline ContextObj::~ContextObj() {
      if( m_ctx ) {
        RTPresult r = rtpContextDestroy( m_ctx );
        if( r!=RTP_SUCCESS )
          m_ctx = 0;        
      }
    }

    inline void ContextObj::setCudaDeviceNumbers( const std::vector<unsigned>& deviceNumbers )
    {
      if( deviceNumbers.size()==0 ) {
        CHK( rtpContextSetCudaDeviceNumbers(m_ctx, 0, NULL) );
      } else {
        CHK( rtpContextSetCudaDeviceNumbers(m_ctx, (unsigned int)deviceNumbers.size(), &deviceNumbers[0]) );
      }
    }

    inline void ContextObj::setCudaDeviceNumbers( unsigned deviceCount, const unsigned* deviceNumbers )
    {
        CHK( rtpContextSetCudaDeviceNumbers(m_ctx, deviceCount, deviceNumbers) );
    }

    inline void ContextObj::setCpuThreads( unsigned numThreads )
    {
      CHK( rtpContextSetCpuThreads(m_ctx, numThreads) );
    }

    inline std::string ContextObj::getLastErrorString()
    {
      const char* str;
	    rtpContextGetLastErrorString( m_ctx, &str );
      return str;
    }

    inline RTPcontext ContextObj::getRTPcontext()
    {
      return m_ctx;
    }

    inline ContextObj::operator Context()
    {
      Context context( this ); context->ref();
      return context;
    }

    inline Context ContextObj::getContext()
    {
      return Context( *this );
    }

    //
    // BUFFERDESC
    //

    inline BufferDescObj::BufferDescObj( const Context& ctx, RTPbufferformat format, RTPbuffertype type, void* buffer ) : m_desc(0) {
      m_ctx = ctx;

      CHK( rtpBufferDescCreate(m_ctx->getRTPcontext(), format, type, buffer, &m_desc) );
    }

    inline BufferDescObj::~BufferDescObj() {
      if( m_desc ) {
        rtpBufferDescDestroy(m_desc);
      }
    }

    inline Context BufferDescObj::getContext()
    {
      return m_ctx;
    }

    inline void BufferDescObj::setRange( RTPsize begin, RTPsize end )
    {
      CHK( rtpBufferDescSetRange(m_desc, begin, end) );
    }

    inline void BufferDescObj::setStride( unsigned strideBytes )
    {
      CHK( rtpBufferDescSetStride(m_desc, strideBytes) );
    }

    inline void BufferDescObj::setCudaDeviceNumber( unsigned deviceNumber )
    {
      CHK( rtpBufferDescSetCudaDeviceNumber(m_desc, deviceNumber) );
    }

    inline RTPbufferdesc BufferDescObj::getRTPbufferdesc()
    {
      return m_desc;
    }

    //
    // MODEL
    //

    inline ModelObj::ModelObj( const Context& ctx ) : m_model(0) {
      m_ctx = ctx;

      CHK( rtpModelCreate(m_ctx->getRTPcontext(), &m_model) );
    }

    inline ModelObj::~ModelObj() {
      if( m_model ) {
        rtpModelDestroy(m_model);
      }
    }

    inline Context ModelObj::getContext()
    {
      return m_ctx;
    }

    inline void ModelObj::setTriangles( const BufferDesc& indices, const BufferDesc& vertices )
    {
      if( indices.isValid() ) {
        CHK( rtpModelSetTriangles(m_model, indices->getRTPbufferdesc(), vertices->getRTPbufferdesc()) );
      } else {
        CHK( rtpModelSetTriangles(m_model, 0, vertices->getRTPbufferdesc()) );
      }
    }

    inline void ModelObj::setTriangles( const BufferDesc& vertices )
    {
      CHK( rtpModelSetTriangles(m_model, 0, vertices->getRTPbufferdesc()) );
    }

    inline void ModelObj::setTriangles( RTPsize triCount, RTPbuffertype type, const void* vertPtr, unsigned stride/*=0 */ )
    {
      BufferDesc vtxBufDesc( m_ctx->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT3, type, const_cast<void*>( vertPtr ) ) );
      vtxBufDesc->setRange( 0, 3*triCount );

      if( stride ) {
        vtxBufDesc->setStride( stride );
      }

      BufferDesc idxBufDesc;
      setTriangles( idxBufDesc, vtxBufDesc );
    }

    inline void ModelObj::setTriangles( RTPsize triCount, RTPbuffertype indexType, const void* indexPtr, RTPsize vertCount, RTPbuffertype vertType, const void* vertPtr, unsigned stride/*=0 */ )
    {
      BufferDesc idxBufDesc( m_ctx->createBufferDesc(RTP_BUFFER_FORMAT_INDICES_INT3, indexType, const_cast<void*>( indexPtr ) ) );
      BufferDesc vtxBufDesc( m_ctx->createBufferDesc(RTP_BUFFER_FORMAT_VERTEX_FLOAT3, vertType, const_cast<void*>( vertPtr ) ) );

      idxBufDesc->setRange( 0, triCount );
      vtxBufDesc->setRange( 0, vertCount );

      if( stride ) {
        vtxBufDesc->setStride( stride );
      }

      setTriangles( idxBufDesc, vtxBufDesc );
    }


    inline void ModelObj::setInstances(RTPsize count, RTPbuffertype instanceType, const RTPmodel* instanceList, RTPbufferformat transformFormat, RTPbuffertype transformType, const void* transformList)
    {
      BufferDesc instances(m_ctx->createBufferDesc(RTP_BUFFER_FORMAT_INSTANCE_MODEL, instanceType, const_cast<RTPmodel*>(instanceList)));
      instances->setRange(0, count);
      BufferDesc transforms(m_ctx->createBufferDesc(transformFormat, transformType, const_cast<void*>(transformList)));
      transforms->setRange(0, count);

      CHK(rtpModelSetInstances(m_model, instances->getRTPbufferdesc(), transforms->getRTPbufferdesc()));
    }

    inline void ModelObj::setInstances( const BufferDesc& instances, const BufferDesc& transforms )
    {
      CHK( rtpModelSetInstances(m_model, instances->getRTPbufferdesc(), transforms->getRTPbufferdesc()) );
    }

    inline void ModelObj::update( unsigned hints )
    {
      CHK( rtpModelUpdate(m_model, hints) );
    }

    inline void ModelObj::finish()
    {
      CHK( rtpModelFinish(m_model) );
    }

    inline int ModelObj::isFinished()
    {
      int finished = 0;
      CHK( rtpModelGetFinished(m_model, &finished) );

      return finished;
    }

    inline void ModelObj::copy( const Model& srcModel )
    {
      CHK( rtpModelCopy(m_model, srcModel->getRTPmodel()) );
    }

    inline ModelObj::operator Model()
    {
      Model model( this ); model->ref();
      return model;
    }

    inline Query ModelObj::createQuery( RTPquerytype queryType )
    {
      Query h( new QueryObj(*this, queryType) );
      return h;
    }

    inline void ModelObj::setBuilderParameter( RTPbuilderparam param, RTPsize size, const void* p )
    {
      CHK( rtpModelSetBuilderParameter(m_model, param, size, p) );
    }

    template<typename T>
    void optix::prime::ModelObj::setBuilderParameter( RTPbuilderparam param, const T& val )
    {
      setBuilderParameter( param, sizeof(T), &val );
    }

    //
    // QUERY
    //

    inline QueryObj::QueryObj( const Model& model, RTPquerytype queryType ) : m_query(0) {
      m_model = model;

      CHK( rtpQueryCreate(model->getRTPmodel(), queryType, &m_query) );
    }

    inline QueryObj::~QueryObj() {
      if( m_query ) {
        rtpQueryDestroy(m_query);
      }
    }

    inline Context QueryObj::getContext()
    {
      return m_model->getContext();
    }

    inline void QueryObj::setRays( const BufferDesc& rays )
    {
      CHK( rtpQuerySetRays(m_query, rays->getRTPbufferdesc()) );
    }

    inline void QueryObj::setRays( RTPsize count, RTPbufferformat format, RTPbuffertype type, void* rays )
    {
      BufferDesc desc(m_model->m_ctx->createBufferDesc(format, type, rays) );
      desc->setRange( 0, count );

      setRays( desc );
    }

    inline void QueryObj::setHits( const BufferDesc& hits )
    {
      CHK( rtpQuerySetHits(m_query, hits->getRTPbufferdesc()) );
    }

    inline void QueryObj::setHits( RTPsize count, RTPbufferformat format, RTPbuffertype type, void* hits )
    {
      BufferDesc desc(m_model->m_ctx->createBufferDesc(format, type, hits) );
      desc->setRange( 0, count );

      setHits( desc );
    }

    inline void QueryObj::execute( unsigned hint )
    {
      CHK( rtpQueryExecute(m_query, hint) );
    }

    inline void QueryObj::finish()
    {
      CHK( rtpQueryFinish(m_query) );
    }

    inline int QueryObj::isFinished()
    {
      int finished = 0;
      CHK( rtpQueryGetFinished(m_query, &finished) );

      return finished;
    }

    inline void QueryObj::setCudaStream( cudaStream_t stream )
    {
      CHK( rtpQuerySetCudaStream(m_query, stream) );
    }

    //
    // EXCEPTION
    //

    inline Exception Exception::makeException( RTPresult code )
    {
      const char* str;
	    rtpGetErrorString( code, &str );
      Exception h( std::string(str), code );

      return h;
    }

    inline Exception Exception::makeException( RTPresult code, RTPcontext ctx )
    {
      const char* str;
	    rtpContextGetLastErrorString( ctx, &str );
      Exception h( std::string(str), code );

      return h;
    }

    inline Exception::Exception( const std::string& message, RTPresult error_code )
    : m_errorMessage(message), m_errorCode( error_code )
    {
    }

    inline RTPresult Exception::getErrorCode() const
    {
      return m_errorCode;
    }

    inline const std::string& Exception::getErrorString() const
    {
      return m_errorMessage;
    }

    inline const char* Exception::what() const throw()
    {
      return m_errorMessage.c_str();
    }

    //
    // HELPER - FUNCTIONS
    //

    inline void checkError( RTPresult code )
    {
      if( code != RTP_SUCCESS ) {
        throw Exception::makeException( code );
      }
    }

    inline void checkError( RTPresult code, RTPcontext context )
    {
      if( code != RTP_SUCCESS ) {
        throw Exception::makeException( code, context );
      }
    }


  } // end namespace prime
} // end namespace optix

#endif // #ifndef __optix_optix_primepp_h__
