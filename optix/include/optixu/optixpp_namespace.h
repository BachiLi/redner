
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
/// \file optixpp_namespace.h
/// \brief A C++ wrapper around the OptiX API.
/// 


#ifndef __optixu_optixpp_namespace_h__
#define __optixu_optixpp_namespace_h__

#include "../optix.h"

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#endif
#include "../optix_gl_interop.h"
#include "../optix_cuda_interop.h"

#include <iterator>
#include <string>
#include <vector>

#include "optixu_vector_types.h"

//-----------------------------------------------------------------------------
//
// Doxygen group specifications
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// 
// C++ API
//
//-----------------------------------------------------------------------------

namespace optix {
  
  class AccelerationObj;
  class BufferObj;
  class ContextObj;
  class GeometryObj;
  class GeometryTrianglesObj;
  class GeometryGroupObj;
  class GeometryInstanceObj;
  class GroupObj;
  class MaterialObj;
  class ProgramObj;
  class SelectorObj;
  class TextureSamplerObj;
  class TransformObj;
  class VariableObj;
  class PostprocessingStageObj;
  class CommandListObj;

  class APIObj;
  class ScopedObj;
  

  ///  \ingroup optixpp
  ///
  ///  \brief The Handle class is a reference counted handle class used to
  ///  manipulate API objects.
  ///
  ///  All interaction with API objects should be done via these handles and the
  ///  associated typedefs rather than direct usage of the objects.
  ///
  template<class T>
  class Handle {
  public:
    /// Default constructor initializes handle to null pointer
    Handle() : ptr(0) {}

    /// Takes a raw pointer to an API object and creates a handle
    Handle(T* ptr) : ptr(ptr) { ref(); }

    /// Takes a raw pointer of arbitrary type and creates a handle 
    template<class U>
    Handle(U* ptr) : ptr(ptr) { ref(); }

    /// Takes a handle of the same type and creates a handle
    Handle(const Handle<T>& copy) : ptr(copy.ptr) { ref(); }
    
    /// Takes a handle of some other type and creates a handle
    template<class U>
    Handle(const Handle<U>& copy) : ptr(copy.ptr) { ref(); }

    /// Assignment of handle with same underlying object type 
    Handle<T>& operator=(const Handle<T>& copy)
    { if(ptr != copy.ptr) { unref(); ptr = copy.ptr; ref(); } return *this; }
    
    /// Assignment of handle with different underlying object type 
    template<class U>
    Handle<T>& operator=( const Handle<U>& copy)
    { if(ptr != copy.ptr) { unref(); ptr = copy.ptr; ref(); } return *this; }

    /// Decrements reference count on the handled object
    ~Handle() { unref(); }

    /// Takes a base optix api opaque type and creates a handle to optixpp wrapper type 
    static Handle<T> take( typename T::api_t p ) { return p? new T(p) : 0; }
    /// Special version that takes an RTobject which must be cast up to the appropriate
    /// OptiX API opaque type.
    static Handle<T> take( RTobject p ) { return p? new T(static_cast<typename T::api_t>(p)) : 0; }

    /// Dereferences the handle
          T* operator->()           { return ptr; }
    const T* operator->() const     { return ptr; }

    /// Retrieve the handled object
          T* get()                  { return ptr; }
    const T* get() const            { return ptr; }

    /// implicit bool cast based on NULLness of wrapped pointer
    operator bool() const  { return ptr != 0; }

    /// Variable access operator.  This operator will query the API object for
    /// a variable with the given name, creating a new variable instance if
    /// necessary. Only valid for ScopedObjs.
    Handle<VariableObj> operator[](const std::string& varname);

    /// \brief Variable access operator.  Identical to operator[](const std::string& varname)
    ///
    /// Explicitly define char* version to avoid ambiguities between builtin
    /// operator[](int, char*) and Handle::operator[]( std::string ).  The
    /// problem lies in that a Handle can be cast to a bool then to an int
    /// which implies that:
    /// \code
    ///    Context context;
    ///    context["var"];
    /// \endcode
    /// can be interpreted as either
    /// \code
    ///    1["var"]; // Strange but legal way to index into a string (same as "var"[1] )
    /// \endcode
    /// or
    /// \code
    ///    context[ std::string("var") ];
    /// \endcode
    Handle<VariableObj> operator[](const char* varname);

    /// Static object creation.  Only valid for contexts.
    static Handle<T> create() { return T::create(); }

    /// Query the machine device count.  Only valid for contexts
    static unsigned int getDeviceCount() { return T::getDeviceCount(); }

  private:
    inline void ref() { if(ptr) ptr->addReference(); }
    inline void unref() { if(ptr && ptr->removeReference() == 0) delete ptr; }
    T* ptr;
  };


  //----------------------------------------------------------------------------

  /// \ingroup optixpp
  /// @{
  typedef Handle<AccelerationObj>        Acceleration;        ///< Use this to manipulate RTacceleration objects.
  typedef Handle<BufferObj>              Buffer;              ///< Use this to manipulate RTbuffer objects.
  typedef Handle<ContextObj>             Context;             ///< Use this to manipulate RTcontext objects.
  typedef Handle<GeometryObj>            Geometry;            ///< Use this to manipulate RTgeometry objects.
  typedef Handle<GeometryTrianglesObj>   GeometryTriangles;   ///< Use this to manipulate RTgeometrytriangles objects.
  typedef Handle<GeometryGroupObj>       GeometryGroup;       ///< Use this to manipulate RTgeometrygroup objects.
  typedef Handle<GeometryInstanceObj>    GeometryInstance;    ///< Use this to manipulate RTgeometryinstance objects.
  typedef Handle<GroupObj>               Group;               ///< Use this to manipulate RTgroup objects.
  typedef Handle<MaterialObj>            Material;            ///< Use this to manipulate RTmaterial objects.
  typedef Handle<ProgramObj>             Program;             ///< Use this to manipulate RTprogram objects.
  typedef Handle<SelectorObj>            Selector;            ///< Use this to manipulate RTselector objects.
  typedef Handle<TextureSamplerObj>      TextureSampler;      ///< Use this to manipulate RTtexturesampler objects.
  typedef Handle<TransformObj>           Transform;           ///< Use this to manipulate RTtransform objects.
  typedef Handle<VariableObj>            Variable;            ///< Use this to manipulate RTvariable objects.
  typedef Handle<PostprocessingStageObj> PostprocessingStage; ///< Use this to manipulate RTpostprocessingstage objects.
  typedef Handle<CommandListObj>         CommandList;         ///< Use this to manipulate RTcommandlist objects.
  /// @}


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Exception class for error reporting from the OptiXpp API.
  /// 
  /// Encapsulates an error message, often the direct result of a failed OptiX C
  /// API function call and subsequent rtContextGetErrorString call.
  ///
  class Exception: public std::exception {
  public:
    /// Create exception
    Exception( const std::string& message, RTresult error_code = RT_ERROR_UNKNOWN )
      : m_message(message), m_error_code( error_code ) {}

    /// Virtual destructor (needed for virtual function calls inherited from
    /// std::exception).
    virtual ~Exception() throw() {}

    /// Retrieve the error message
    const std::string& getErrorString() const { return m_message; }
  
    /// Retrieve the error code 
    RTresult getErrorCode() const { return m_error_code; }

    /// Helper for creating exceptions from an RTresult code origination from
    /// an OptiX C API function call.
    static Exception makeException( RTresult code, RTcontext context );

    /// From std::exception
    virtual const char* what() const throw() { return getErrorString().c_str(); }
  private:
    std::string m_message;
    RTresult    m_error_code;
  };

  inline Exception Exception::makeException( RTresult code, RTcontext context )
  {
    const char* str;
    rtContextGetErrorString( context, code, &str);
    return Exception( std::string(str), code );
  }


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Base class for all reference counted wrappers around OptiX C API
  /// opaque types.
  ///  
  /// Wraps:
  ///   - RTcontext
  ///   - RTbuffer
  ///   - RTgeometry
  ///   - RTgeometrytriangles
  ///   - RTgeometryinstance
  ///   - RTgeometrygroup
  ///   - RTgroup
  ///   - RTmaterial
  ///   - RTprogram
  ///   - RTselector
  ///   - RTtexturesampler
  ///   - RTtransform
  ///   - RTvariable
  ///
  class APIObj {
  public:
    APIObj() : ref_count(0) {}
    virtual ~APIObj() {}

    /// Increment the reference count for this object
    void addReference()    { ++ref_count; }
    /// Decrement the reference count for this object
    int  removeReference() { return --ref_count; }

    /// Retrieve the context this object is associated with.  See rt[ObjectType]GetContext.
    virtual Context getContext()const=0;

    /// Check the given result code and throw an error with appropriate message
    /// if the code is not RTsuccess
    virtual void checkError(RTresult code)const;
    virtual void checkError(RTresult code, Context context )const;
    
    void checkErrorNoGetContext(RTresult code)const;

    /// For backwards compatability.  Use Exception::makeException instead.
    static Exception makeException( RTresult code, RTcontext context );
  private:
    int ref_count;
  };

  inline Exception APIObj::makeException( RTresult code, RTcontext context )
  {
    return Exception::makeException( code, context );
  }


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Base class for all wrapper objects which can be destroyed and validated.
  ///
  /// Wraps:
  ///   - RTcontext
  ///   - RTgeometry
  ///   - RTgeometrytriangles
  ///   - RTgeometryinstance
  ///   - RTgeometrygroup
  ///   - RTgroup
  ///   - RTmaterial
  ///   - RTprogram
  ///   - RTselector
  ///   - RTtexturesampler
  ///   - RTtransform
  ///
  class DestroyableObj : public APIObj {
  public:
    virtual ~DestroyableObj() {}

    /// call rt[ObjectType]Destroy on the underlying OptiX C object 
    virtual void destroy() = 0;

    /// call rt[ObjectType]Validate on the underlying OptiX C object 
    virtual void validate() = 0;
  };


  
  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Base class for all objects which are OptiX variable containers.
  ///
  /// Wraps:
  ///   - RTcontext
  ///   - RTgeometry
  ///   - RTgeometryinstance
  ///   - RTmaterial
  ///   - RTprogram
  ///
  class ScopedObj : public DestroyableObj {
  public:
    virtual ~ScopedObj() {}

    /// Declare a variable associated with this object.  See rt[ObjectType]DeclareVariable.
    /// Note that this function is wrapped by the convenience function Handle::operator[].
    virtual Variable declareVariable(const std::string& name) = 0;
    /// Query a variable associated with this object by name.  See rt[ObjectType]QueryVariable.
    /// Note that this function is wrapped by the convenience function Handle::operator[].
    virtual Variable queryVariable(const std::string& name) const = 0;
    /// Remove a variable associated with this object
    virtual void removeVariable(Variable v) = 0;
    /// Query the number of variables associated with this object.  Used along
    /// with ScopedObj::getVariable to iterate over variables in an object.
    /// See rt[ObjectType]GetVariableCount
    virtual unsigned int getVariableCount() const = 0;
    /// Query variable by index.  See rt[ObjectType]GetVariable.
    virtual Variable getVariable(unsigned int index) const = 0;
  };

  
  
  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Variable object wraps OptiX C API RTvariable type and its related function set.
  /// 
  /// See the OptiX Programming Guide for a complete description of
  /// the usage and behavior of RTvariable objects.  Creation and querying of
  /// Variables can be performed via the Handle::operator[] function of the scope
  /// object associated with the variable. For example: 
  /// \code 
  ///   my_context["new_variable"]->setFloat( 1.0f );
  /// \endcode
  /// will create a variable named \p new_variable on the object \p my_context if
  /// it does not already exist.  It will then set the value of that variable to
  /// be a float 1.0f.
  ///
  class VariableObj : public APIObj {
  public:

    Context getContext() const;

    /// \name Float setters
    /// Set variable to have a float value. 
    //@{
    /// Set variable value to a scalar float
    void setFloat(float f1);
    /// Set variable value to a float2 
    void setFloat(optix::float2 f);
    /// Set variable value to a float2 
    void setFloat(float f1, float f2);
    /// Set variable value to a float3 
    void setFloat(optix::float3 f);
    /// Set variable value to a float3 
    void setFloat(float f1, float f2, float f3);
    /// Set variable value to a float4 
    void setFloat(optix::float4 f);
    /// Set variable value to a float4 
    void setFloat(float f1, float f2, float f3, float f4);
    /// Set variable value to a scalar float
    void set1fv(const float* f);
    /// Set variable value to a float2 
    void set2fv(const float* f);
    /// Set variable value to a float3 
    void set3fv(const float* f);
    /// Set variable value to a float4 
    void set4fv(const float* f);
    //@}

    /// \name Int setters
    /// Set variable to have an int value. 
    //@{
    void setInt(int i1);
    void setInt(int i1, int i2);
    void setInt(optix::int2 i);
    void setInt(int i1, int i2, int i3);
    void setInt(optix::int3 i);
    void setInt(int i1, int i2, int i3, int i4);
    void setInt(optix::int4 i);
    void set1iv(const int* i);
    void set2iv(const int* i);
    void set3iv(const int* i);
    void set4iv(const int* i);
    //@}

    /// \name Unsigned int  setters
    /// Set variable to have an unsigned int value. 
    //@{
    void setUint(unsigned int u1);
    void setUint(unsigned int u1, unsigned int u2);
    void setUint(unsigned int u1, unsigned int u2, unsigned int u3);
    void setUint(unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4);
    void setUint(optix::uint2 u);
    void setUint(optix::uint3 u);
    void setUint(optix::uint4 u);
    void set1uiv(const unsigned int* u);
    void set2uiv(const unsigned int* u);
    void set3uiv(const unsigned int* u);
    void set4uiv(const unsigned int* u);
    //@}

    /// \name Long Long setters
    /// Set variable to have a long long value. 
    //@{
    void setLongLong(long long i1);
    void setLongLong(long long i1, long long i2);
    void setLongLong(long long i1, long long i2, long long i3);
    void setLongLong(long long i1, long long i2, long long i3, long long i4);
    void setLongLong(optix::longlong2 i);
    void setLongLong(optix::longlong3 i);
    void setLongLong(optix::longlong4 i);
    void set1llv(const long long* i);
    void set2llv(const long long* i);
    void set3llv(const long long* i);
    void set4llv(const long long* i);
    //@}

    /// \name Unsigned Long Long setters
    /// Set variable to have an unsigned long long value. 
    //@{
    void setULongLong(unsigned long long u1);
    void setULongLong(unsigned long long u1, unsigned long long u2);
    void setULongLong(unsigned long long u1, unsigned long long u2, unsigned long long u3);
    void setULongLong(unsigned long long u1, unsigned long long u2, unsigned long long u3, 
        unsigned long long u4);
    void setULongLong(optix::ulonglong2 u);
    void setULongLong(optix::ulonglong3 u);
    void setULongLong(optix::ulonglong4 u);
    void set1ullv(const unsigned long long* u);
    void set2ullv(const unsigned long long* u);
    void set3ullv(const unsigned long long* u);
    void set4ullv(const unsigned long long* u);

    //@}
    /// \name Matrix setters
    /// Set variable to have a Matrix value 
    //@{
    void setMatrix2x2fv(bool transpose, const float* m);
    void setMatrix2x3fv(bool transpose, const float* m);
    void setMatrix2x4fv(bool transpose, const float* m);
    void setMatrix3x2fv(bool transpose, const float* m);
    void setMatrix3x3fv(bool transpose, const float* m);
    void setMatrix3x4fv(bool transpose, const float* m);
    void setMatrix4x2fv(bool transpose, const float* m);
    void setMatrix4x3fv(bool transpose, const float* m);
    void setMatrix4x4fv(bool transpose, const float* m);
    //@}

    /// \name Numeric value getters 
    /// Query value of a variable with numeric value 
    //@{
    
    float         getFloat()  const;
    optix::float2 getFloat2() const;
    optix::float3 getFloat3() const;
    optix::float4 getFloat4() const;
    void          getFloat(float& f1) const;
    void          getFloat(float& f1, float& f2) const;
    void          getFloat(float& f1, float& f2, float& f3) const;
    void          getFloat(float& f1, float& f2, float& f3, float& f4) const;

    unsigned      getUint()  const;
    optix::uint2  getUint2() const;
    optix::uint3  getUint3() const;
    optix::uint4  getUint4() const;
    void          getUint(unsigned& u1) const;
    void          getUint(unsigned& u1, unsigned& u2) const;
    void          getUint(unsigned& u1, unsigned& u2, unsigned& u3) const;
    void          getUint(unsigned& u1, unsigned& u2, unsigned& u3,
                          unsigned& u4) const;

    int           getInt()  const;
    optix::int2   getInt2() const;
    optix::int3   getInt3() const;
    optix::int4   getInt4() const;
    void          getInt(int& i1) const;
    void          getInt(int& i1, int& i2) const;
    void          getInt(int& i1, int& i2, int& i3) const;
    void          getInt(int& i1, int& i2, int& i3, int& i4) const;

    unsigned long long getULongLong()  const;
    optix::ulonglong2  getULongLong2() const;
    optix::ulonglong3  getULongLong3() const;
    optix::ulonglong4  getULongLong4() const;
    void               getULongLong(unsigned long long& ull1) const;
    void               getULongLong(unsigned long long& ull1, unsigned long long& ull2) const;
    void               getULongLong(unsigned long long& ull1, unsigned long long& ull2,
        unsigned long long& ull3) const;
    void               getULongLong(unsigned long long& ull1, unsigned long long& ull2, 
        unsigned long long& ull3, unsigned long long& ull4) const;

    long long          getLongLong()  const;
    optix::longlong2   getLongLong2() const;
    optix::longlong3   getLongLong3() const;
    optix::longlong4   getLongLong4() const;
    void               getLongLong(long long& ll1) const;
    void               getLongLong(long long& ll1, long long& ll2) const;
    void               getLongLong(long long& ll1, long long& ll2, long long& ll3) const;
    void               getLongLong(long long& ll1, long long& ll2, long long& ll3, long long& ll4) const;

    void getMatrix2x2(bool transpose, float* m) const;
    void getMatrix2x3(bool transpose, float* m) const;
    void getMatrix2x4(bool transpose, float* m) const;
    void getMatrix3x2(bool transpose, float* m) const;
    void getMatrix3x3(bool transpose, float* m) const;
    void getMatrix3x4(bool transpose, float* m) const;
    void getMatrix4x2(bool transpose, float* m) const;
    void getMatrix4x3(bool transpose, float* m) const;
    void getMatrix4x4(bool transpose, float* m) const;

    //@}


    /// \name OptiX API object setters 
    /// Set variable to have an OptiX API object as its value
    //@{
    void setBuffer(Buffer buffer);
    void set(Buffer buffer);
    void setTextureSampler(TextureSampler texturesample);
    void set(TextureSampler texturesample);
    void set(GeometryGroup group);
    void set(Group group);
    void set(Program program);
    void setProgramId(Program program);
    void set(Selector selector);
    void set(Transform transform);
    //@}

    /// \name OptiX API object getters 
    /// Retrieve OptiX API object value from a variable
    //@{
    Buffer           getBuffer() const;
    GeometryGroup    getGeometryGroup() const;
    GeometryInstance getGeometryInstance() const;
    Group            getGroup() const;
    Program          getProgram() const;
    Selector         getSelector() const;
    TextureSampler   getTextureSampler() const;
    Transform        getTransform() const;
    //@}

    /// \name User data variable accessors 
    //@{
    /// Set the variable to a user defined type given the sizeof the user object
    void setUserData(RTsize size, const void* ptr);
    /// Retrieve a user defined type given the sizeof the user object
    void getUserData(RTsize size,       void* ptr) const;
    //@}

    /// Retrieve the name of the variable
    std::string getName() const;
    
    /// Retrieve the annotation associated with the variable 
    std::string getAnnotation() const;

    /// Query the object type of the variable
    RTobjecttype getType() const;

    /// Get the OptiX C API object wrapped by this instance
    RTvariable get();

    /// Get the size of the variable data in bytes (eg, float4 returns 4*sizeof(float) )
    RTsize getSize() const;

  private:
    typedef RTvariable api_t;

    RTvariable m_variable;
    VariableObj(RTvariable variable) : m_variable(variable) {}
    friend class Handle<VariableObj>;

  };

  template<class T>
  Handle<VariableObj> Handle<T>::operator[](const std::string& varname)
  {
    Variable v = ptr->queryVariable( varname );
    if( v.operator->() == 0)
      v = ptr->declareVariable( varname );
    return v;
  }

  template<class T>
  Handle<VariableObj> Handle<T>::operator[](const char* varname) 
  {
    return (*this)[ std::string( varname ) ]; 
  }

  
  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Context object wraps the OptiX C API RTcontext opaque type and its associated function set.
  ///
  class ContextObj : public ScopedObj {
  public:

    /// Call rtDeviceGetDeviceCount and returns number of valid devices
    static unsigned int getDeviceCount();

    /// Call rtDeviceGetAttribute and return the name of the device
    static std::string getDeviceName(int ordinal);

    /// Call rtDeviceGetAttribute and return the PCI bus id of the device
    static std::string getDevicePCIBusId(int ordinal);
    
    /// Call rtDeviceGetAttribute and return the desired attribute value
    static void getDeviceAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, void* p);

    /// Call rtDeviceGetAttribute and return the list of ordinals compatible with the device; a device is always compatible with itself.
    static std::vector<int> getCompatibleDevices( int ordinal );

    /// Creates a Context object.  See @ref rtContextCreate
    static Context create();

    /// Destroy Context and all of its associated objects.  See @ref rtContextDestroy.
    void destroy();

    /// See @ref rtContextValidate
    void validate();

    /// Retrieve the Context object associated with this APIObject.  In this case,
    /// simply returns itself.
    Context getContext() const;

    /// @{
    /// See @ref APIObj::checkError
    void checkError(RTresult code)const;

    /// See @ref rtContextGetErrorString
    std::string getErrorString( RTresult code ) const;
    /// @}

    /// @{
    /// <B>traverser parameter unused in OptiX 4.0</B> See @ref rtAccelerationCreate.
    Acceleration createAcceleration(const std::string& builder, const std::string& ignored = "");

    /// Create a buffer with given RTbuffertype.  See @ref rtBufferCreate.
    Buffer createBuffer(unsigned int type);
    /// Create a buffer with given RTbuffertype and RTformat.  See @ref rtBufferCreate, @ref rtBufferSetFormat
    Buffer createBuffer(unsigned int type, RTformat format);
    /// Create a buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat and @ref rtBufferSetSize1D.
    Buffer createBuffer(unsigned int type, RTformat format, RTsize width);
    /// Create a mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat, @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize1D.
    Buffer createMipmappedBuffer(unsigned int type, RTformat format, RTsize width, unsigned int levels);
    /// Create a buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat and @ref rtBufferSetSize2D.
    Buffer createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height);
    /// Create a mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat, @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize2D.
    Buffer createMipmappedBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, unsigned int levels);
    /// Create a buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat and @ref rtBufferSetSize3D.
    Buffer createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth);
    /// Create a mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat, @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize3D.
    Buffer createMipmappedBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth, unsigned int levels);  
    
    /// Create a 1D layered mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat,  @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize3D.
    Buffer create1DLayeredBuffer(unsigned int type, RTformat format, RTsize width, RTsize layers, unsigned int levels);
    /// Create a 2D layered mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat,  @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize3D.
    Buffer create2DLayeredBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize layers, unsigned int levels);
    /// Create a cube mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat,  @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize3D.
    Buffer createCubeBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, unsigned int levels);
    /// Create a cube layered mipmapped buffer with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat,  @ref rtBufferSetMipLevelCount, and @ref rtBufferSetSize3D.
    Buffer createCubeLayeredBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize faces, unsigned int levels);

    /// Create a buffer for CUDA with given RTbuffertype.  See @ref rtBufferCreate.
    Buffer createBufferForCUDA(unsigned int type);
    /// Create a buffer for CUDA with given RTbuffertype and RTformat.  See @ref rtBufferCreate, @ref rtBufferSetFormat
    Buffer createBufferForCUDA(unsigned int type, RTformat format);
    /// Create a buffer for CUDA with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat and @ref rtBufferSetSize1D.
    Buffer createBufferForCUDA(unsigned int type, RTformat format, RTsize width);
    /// Create a buffer for CUDA with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat and @ref rtBufferSetSize2D.
    Buffer createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height);
    /// Create a buffer for CUDA with given RTbuffertype, RTformat and dimension.  See @ref rtBufferCreate,
    /// @ref rtBufferSetFormat and @ref rtBufferSetSize3D.
    Buffer createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth);

    /// Create buffer from GL buffer object.  See @ref rtBufferCreateFromGLBO
    Buffer createBufferFromGLBO(unsigned int type, unsigned int vbo);

    /// Create demand loaded buffer from a callback.  See @ref rtBufferCreateFromCallback
    Buffer createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData);
    /// Create a demand loaded buffer from a callback with given format.  See @ref rtBufferCreateFromCallback and @ref rtBufferSetForamt.
    Buffer createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format);
    /// Create a demand loaded buffer from a callback with given RTbuffertype, format and dimension.
    /// See @ref rtBufferCreateFromCallback, @ref rtBufferSetFormat and @ref rtBufferSetSize1D.
    Buffer createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format, RTsize width);
    /// Create a demand loaded buffer from a callback with given RTbuffertype, format and dimension.
    /// See @ref rtBufferCreateFromCallback, @ref rtBufferSetFormat and @ref rtBufferSetSize2D.
    Buffer createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format, RTsize width, RTsize height);
    /// Create a demand loaded buffer from a callback with given RTbuffertype, format and dimension.
    /// See @ref rtBufferCreateFromCallback, @ref rtBufferSetFormat and @ref rtBufferSetSize3D.
    Buffer createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format, RTsize width, RTsize height, RTsize depth);

    /// Create TextureSampler from GL image.  See @ref rtTextureSamplerCreateFromGLImage
    TextureSampler createTextureSamplerFromGLImage(unsigned int id, RTgltarget target);

    /// Queries the Buffer object from a given buffer id obtained from a previous call to
    /// @ref BufferObj::getId.  See @ref BufferObj::getId and @ref rtContextGetBufferFromId.
    Buffer getBufferFromId(int buffer_id);
       
    /// Queries the Program object from a given program id obtained from a previous call to
    /// @ref ProgramObj::getId.  See @ref ProgramObj::getId and @ref rtContextGetProgramFromId.
    Program getProgramFromId(int program_id);
       
    /// Queries the TextureSampler object from a given sampler id obtained from a previous call to
    /// @ref TextureSamplerObj::getId.  See @ref TextureSamplerObj::getId and @ref rtContextGetTextureSamplerFromId.
    TextureSampler getTextureSamplerFromId(int sampler_id);

    /// See @ref rtGeometryCreate
    Geometry createGeometry();
    /// See @ref rtGeometryTrianglesCreate
    GeometryTriangles createGeometryTriangles();
    /// See @ref rtGeometryInstanceCreate
    GeometryInstance createGeometryInstance();
    /// Create a geometry instance with a Geometry object and a set of associated materials.  See
    /// @ref rtGeometryInstanceCreate, @ref rtGeometryInstanceSetMaterialCount, and @ref rtGeometryInstanceSetMaterial
    template<class Iterator>
    GeometryInstance createGeometryInstance( Geometry geometry, Iterator matlbegin, Iterator matlend );
    /// Create a geometry instance with a GeometryTriangles object and a set of associated materials.  See
    /// @ref rtGeometryInstanceCreate, @ref rtGeometryInstanceSetMaterialCount, and @ref rtGeometryInstanceSetMaterial
    template<class Iterator>
    GeometryInstance createGeometryInstance( GeometryTriangles geometrytriangles, Iterator matlbegin, Iterator matlend );
    GeometryInstance createGeometryInstance( GeometryTriangles geometry, Material mat );

    /// See @ref rtGroupCreate
    Group createGroup();
    /// Create a Group with a set of child nodes.  See @ref rtGroupCreate, @ref rtGroupSetChildCount and
    /// @ref rtGroupSetChild
    template<class Iterator>
    Group createGroup( Iterator childbegin, Iterator childend );

    /// See @ref rtGeometryGroupCreate
    GeometryGroup createGeometryGroup();
    /// Create a GeometryGroup with a set of child nodes.  See @ref rtGeometryGroupCreate,
    /// @ref rtGeometryGroupSetChildCount and @ref rtGeometryGroupSetChild
    template<class Iterator>
    GeometryGroup createGeometryGroup( Iterator childbegin, Iterator childend );

    /// See @ref rtTransformCreate
    Transform createTransform();

    /// See @ref rtMaterialCreate
    Material createMaterial();

    /// See @ref rtProgramCreateFromPTXFile
    Program createProgramFromPTXFile  ( const std::string& filename, const std::string& program_name );
    /// See @ref rtProgramCreateFromPTXFiles
    Program createProgramFromPTXFiles  ( const std::vector<std::string>& filenames, const std::string& program_name );
    Program createProgramFromPTXFiles  ( const std::vector<const char*>& filenames, const std::string& program_name );
    /// See @ref rtProgramCreateFromPTXString
    Program createProgramFromPTXString( const std::string& ptx, const std::string& program_name );
    /// See @ref rtProgramCreateFromPTXStrings
    Program createProgramFromPTXStrings( const std::vector<std::string>& ptxStrings, const std::string& program_name );
    Program createProgramFromPTXStrings( const std::vector<const char*>& ptxStrings, const std::string& program_name );
    Program createProgramFromProgram( Program program_in );

    /// See @ref rtSelectorCreate
    Selector createSelector();

    /// See @ref rtTextureSamplerCreate
    TextureSampler createTextureSampler();

    /// @{
    /// Create a builtin postprocessing stage. See @ref rtPostProcessingStageCreateBuiltin.
    PostprocessingStage createBuiltinPostProcessingStage(const std::string& builtin_name);

    /// Create a new command list. See @ref rtCommandListCreate.
    CommandList createCommandList();
    /// @}

    /// @{
    /// See @ref rtContextSetDevices
    template<class Iterator>
    void setDevices(Iterator begin, Iterator end);

    /// See @ref rtContextGetDevices.  This returns the list of currently enabled devices.
    std::vector<int> getEnabledDevices() const;

    /// See @ref rtContextGetDeviceCount.  As opposed to getDeviceCount, this returns only the
    /// number of enabled devices.
    unsigned int getEnabledDeviceCount() const;
    /// @}

    /// @{
    /// See @ref rtContextGetAttribute
    int getMaxTextureCount() const;

    /// See @ref rtContextGetAttribute
    int getCPUNumThreads() const;

    /// See @ref rtContextGetAttribute
    RTsize getUsedHostMemory() const;

    /// See @ref rtContextGetAttribute
    bool getPreferFastRecompiles() const;

    /// See @ref rtContextGetAttribute
    bool getForceInlineUserFunctions() const;

    /// <B>Deprecated in OptiX 4.0</B> See @ref rtContextGetAttribute
    int getGPUPagingActive() const;

    /// <B>Deprecated in OptiX 4.0</B> See @ref rtContextGetAttribute
    int getGPUPagingForcedOff() const;

    /// See @ref rtContextGetAttribute
    RTsize getAvailableDeviceMemory(int ordinal) const;
    /// @}

    /// @{
    /// See @ref rtContextSetAttribute
    void setCPUNumThreads(int cpu_num_threads);

    /// See @ref rtContextSetAttribute
    void setPreferFastRecompiles( bool enabled );

    /// See @ref rtContextSetAttribute
    void setForceInlineUserFunctions( bool enabled );

    /// See @ref rtContextSetAttribute
    void setDiskCacheLocation( const std::string& path );

    /// See @ref rtContextGetAttribute
    std::string getDiskCacheLocation();

    /// See @ref rtContextSetAttribute
    void setDiskCacheMemoryLimits( RTsize lowWaterMark, RTsize highWaterMark );

    /// See @ref rtContextGetAttribute
    void getDiskCacheMemoryLimits( RTsize& lowWaterMark, RTsize& highWaterMark );

    /// <B>Deprecated in OptiX 4.0</B> See @ref rtContextSetAttribute
    void setGPUPagingForcedOff(int gpu_paging_forced_off);

    /// See rtContextSetAttribute
    template<class T>
    void setAttribute(RTcontextattribute attribute, const T& val);
    /// @}

    /// @{    
    /// See @ref rtContextSetStackSize
    void setStackSize(RTsize  stack_size_bytes);
    /// See @ref rtContextGetStackSize
    RTsize getStackSize() const;

    /// @{    
    /// See @ref rtContextSetMaxCallableProgramDepth
    void setMaxCallableProgramDepth(unsigned int  max_depth);
    /// See @ref rtContextGetMaxCallableProgramDepth
    unsigned int getMaxCallableProgramDepth() const;

    /// @{    
    /// See @ref rtContextSetMaxTraceDepth
    void setMaxTraceDepth(unsigned int  max_depth);
    /// See @ref rtContextGetMaxTraceDepth
    unsigned int getMaxTraceDepth() const;

    /// See @ref rtContextSetTimeoutCallback
    /// RTtimeoutcallback is defined as typedef int (*RTtimeoutcallback)(void).
    void setTimeoutCallback(RTtimeoutcallback callback, double min_polling_seconds);

    /// See @ref rtContextSetUsageReportCallback
    /// RTusagereportcallback is defined as typedef void (*RTusagereportcallback)(int, const char*, const char*, void*).
    void setUsageReportCallback(RTusagereportcallback callback, int verbosity, void* cbdata);

    /// See @ref rtContextSetEntryPointCount
    void setEntryPointCount(unsigned int  num_entry_points);
    /// See @ref rtContextGetEntryPointCount
    unsigned int getEntryPointCount() const;

    /// See @ref rtContextSetRayTypeCount
    void setRayTypeCount(unsigned int  num_ray_types);
    /// See @ref rtContextGetRayTypeCount
    unsigned int getRayTypeCount() const;
    /// @}

    /// @{
    /// See @ref rtContextSetRayGenerationProgram
    void setRayGenerationProgram(unsigned int entry_point_index, Program  program);
    /// See @ref rtContextGetRayGenerationProgram
    Program getRayGenerationProgram(unsigned int entry_point_index) const;

    /// See @ref rtContextSetExceptionProgram
    void setExceptionProgram(unsigned int entry_point_index, Program  program);
    /// See @ref rtContextGetExceptionProgram
    Program getExceptionProgram(unsigned int entry_point_index) const;

    /// See @ref rtContextSetExceptionEnabled
    void setExceptionEnabled( RTexception exception, bool enabled );
    /// See @ref rtContextGetExceptionEnabled
    bool getExceptionEnabled( RTexception exception ) const;

    /// See @ref rtContextSetMissProgram
    void setMissProgram(unsigned int ray_type_index, Program  program);
    /// See @ref rtContextGetMissProgram
    Program getMissProgram(unsigned int ray_type_index) const;
    /// @}

    /// <B>Deprecated in OptiX 4.0</B> See @ref rtContextCompile
    void compile();

    /// @{
    /// See @ref rtContextLaunch "rtContextLaunch"
    void launch(unsigned int entry_point_index, RTsize image_width);
    /// See @ref rtContextLaunch "rtContextLaunch"
    void launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height);
    /// See @ref rtContextLaunch "rtContextLaunch"
    void launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth);
    /// @}

    /// @{
    /// See @ref rtContextLaunchProgressive2D "rtContextLaunchProgressive"
    void launchProgressive(unsigned int entry_point_index, RTsize image_width, RTsize image_height, unsigned int max_subframes); 

    /// See @ref rtContextStopProgressive
    void stopProgressive();
    /// @}

    /// See @ref rtContextGetRunningState
    int getRunningState() const;

    /// @{
    /// See @ref rtContextSetPrintEnabled
    void setPrintEnabled(bool enabled);
    /// See @ref rtContextGetPrintEnabled
    bool getPrintEnabled() const;
    /// See @ref rtContextSetPrintBufferSize
    void setPrintBufferSize(RTsize buffer_size_bytes);
    /// See @ref rtContextGetPrintBufferSize
    RTsize getPrintBufferSize() const;
    /// See @ref rtContextSetPrintLaunchIndex.
    void setPrintLaunchIndex(int x, int y=-1, int z=-1);
    /// See @ref rtContextGetPrintLaunchIndex
    optix::int3 getPrintLaunchIndex() const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Return the OptiX C API RTcontext object
    RTcontext get();
  private:
    typedef RTcontext api_t;

    virtual ~ContextObj() {}
    RTcontext m_context;
    ContextObj(RTcontext context) : m_context(context) {}
    friend class Handle<ContextObj>;
  };


  //----------------------------------------------------------------------------
  

  /// \ingroup optixpp
  ///
  /// \brief Program object wraps the OptiX C API RTprogram opaque type and its associated function set.
  ///
  class ProgramObj : public ScopedObj {
  public:
    void destroy();
    void validate();

    Context getContext() const;

    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    void setCallsitePotentialCallees( const std::string& callSiteName, const std::vector<int>& calleeIds );
    /// @{
    /// Returns the device-side ID of this program object. See @ref rtProgramGetId
    int getId() const;
    /// @}

    RTprogram get();
  private:
    typedef RTprogram api_t;
    virtual ~ProgramObj() {}
    RTprogram m_program;
    ProgramObj(RTprogram program) : m_program(program) {}
    friend class Handle<ProgramObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Group wraps the OptiX C API RTgroup opaque type and its associated function set.
  ///
  class GroupObj : public DestroyableObj {
  public:
    void destroy();
    void validate();

    Context getContext() const;

    /// @{
    /// Set the Acceleration structure for this group.  See @ref rtGroupSetAcceleration.
    void setAcceleration(Acceleration acceleration);
    /// Query the Acceleration structure for this group.  See @ref rtGroupGetAcceleration.
    Acceleration getAcceleration() const;
    /// @}

    /// @{
    /// Set the number of children for this group.  See @ref rtGroupSetChildCount.
    void setChildCount(unsigned int  count);
    /// Query the number of children for this group.  See @ref rtGroupGetChildCount.
    unsigned int getChildCount() const;

    /// Set an indexed child within this group.  See @ref rtGroupSetChild.
    template< typename T > void setChild(unsigned int index, T child);
    /// Query an indexed child within this group.  See @ref rtGroupGetChild.
    template< typename T > T getChild(unsigned int index) const;
    /// Query indexed child's type.  See @ref rtGroupGetChildType
    RTobjecttype getChildType(unsigned int index) const;

    /// Set a new child in this group and returns its new index.  See @ref rtGroupSetChild
    template< typename T > unsigned int addChild(T child);
    /// Remove a child in this group. Note: this function is not order-preserving. 
    /// Returns the position of the removed element if succeeded.
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    template< typename T > unsigned int removeChild(T child);
    /// Remove a child in this group. Note: this function is not order-preserving. 
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    void removeChild(int index);
    /// Remove a child in this group. Note: this function is not order-preserving. 
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    void removeChild(unsigned int index);
    /// Query a child in this group for its index. See @ref rtGroupGetChild
    template< typename T > unsigned int getChildIndex(T child) const;
    /// @}

    /// @{
    /// See @ref rtGroupSetVisibilityMask
    void setVisibilityMask( RTvisibilitymask );
    RTvisibilitymask getVisibilityMask() const;
    /// @}
    /// Get the underlying OptiX C API RTgroup opaque pointer.
    RTgroup get();

  private:
    typedef RTgroup api_t;
    virtual ~GroupObj() {}
    RTgroup m_group;
    GroupObj(RTgroup group) : m_group(group) {}
    friend class Handle<GroupObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief GeometryGroup wraps the OptiX C API RTgeometrygroup opaque type and its associated function set.
  ///
  class GeometryGroupObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the Acceleration structure for this group.  See @ref rtGeometryGroupSetAcceleration.
    void setAcceleration(Acceleration acceleration);
    /// Query the Acceleration structure for this group.  See @ref rtGeometryGroupGetAcceleration.
    Acceleration getAcceleration() const;
    /// @}

    /// @{
    /// Set the number of children for this group.  See @ref rtGeometryGroupSetChildCount.
    void setChildCount(unsigned int  count);
    /// Query the number of children for this group.  See @ref rtGeometryGroupGetChildCount.
    unsigned int getChildCount() const;

    /// Set an indexed GeometryInstance child of this group.  See @ref rtGeometryGroupSetChild.
    void setChild(unsigned int index, GeometryInstance geometryinstance);
    /// Query an indexed GeometryInstance within this group.  See @ref rtGeometryGroupGetChild.
    GeometryInstance getChild(unsigned int index) const;

    /// Set a new child in this group and return its new index.  See @ref rtGeometryGroupSetChild
    unsigned int addChild(GeometryInstance child);
    /// Remove a child in this group. Note: this function is not order-preserving. 
    /// Returns the position of the removed element if succeeded.
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    unsigned int removeChild(GeometryInstance child);
    /// Remove a child in this group. Note: this function is not order-preserving. 
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    void removeChild(int index);
    /// Remove a child in this group. Note: this function is not order-preserving. 
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    void removeChild(unsigned int index);
    /// Query a child in this group for its index. See @ref rtGeometryGroupGetChild
    unsigned int getChildIndex(GeometryInstance child) const;
    /// @}

    /// @{
    /// See @ref rtGeometryGroupSetFlags
    void setFlags( RTinstanceflags flags );
    RTinstanceflags getFlags() const;
    /// See @ref rtGeometryGroupSetVisibilityMask
    void setVisibilityMask( RTvisibilitymask mask );
    RTvisibilitymask getVisibilityMask() const;
    /// @}

    /// Get the underlying OptiX C API RTgeometrygroup opaque pointer.
    RTgeometrygroup get();

  private:
    typedef RTgeometrygroup api_t;
    virtual ~GeometryGroupObj() {}
    RTgeometrygroup m_geometrygroup;
    GeometryGroupObj(RTgeometrygroup geometrygroup) : m_geometrygroup(geometrygroup) {}
    friend class Handle<GeometryGroupObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Transform wraps the OptiX C API RTtransform opaque type and its associated function set.
  ///
  class TransformObj  : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the child node of this transform.  See @ref rtTransformSetChild.
    template< typename T > void setChild(T child);
    /// Set the child node of this transform.  See @ref rtTransformGetChild.
    template< typename T > T getChild() const;
    /// Query child's type.  See @ref rtTransformGetChildType.
    RTobjecttype getChildType() const;
    /// @}

    /// @{
    /// Set the transform matrix for this node.  See @ref rtTransformSetMatrix.
    void setMatrix(bool transpose, const float* matrix, const float* inverse_matrix);
    /// Get the transform matrix for this node.  See @ref rtTransformGetMatrix.
    void getMatrix(bool transpose, float* matrix, float* inverse_matrix) const;
    /// @}
    
    /// @{
    /// Set the motion time range for this transform. See @ref rtTransformSetMotionRange.
    void setMotionRange( float timeBegin, float timeEnd );
    /// Query the motion time range for this transform. See @ref rtTransformGetMotionRange.
    void getMotionRange( float& timeBegin, float& timeEnd );
    /// Set the motion border mode for this transform. See @ref rtTransformSetMotionBorderMode.
    void setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode );
    /// Query the motion border mode for this transform. See @ref rtTransformGetMotionBorderMode.
    void getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode );
    /// Set the motion keys for this transform. See @ref rtTransformSetMotionKeys.
    void setMotionKeys( unsigned int n, RTmotionkeytype type, const float* keys );
    /// Query the number of motion keys for this transform.  See @ref rtTransformGetMotionKeyCount.
    unsigned int getMotionKeyCount();
    /// Query the motion key type for this transform.  See @ref rtTransformGetMotionKeyType.
    RTmotionkeytype getMotionKeyType();
    /// Query the motion keys for this transform. See @ref rtTransformGetMotionKeys.
    void getMotionKeys( float* keys );
    /// @}

    /// Get the underlying OptiX C API RTtransform opaque pointer.
    RTtransform get();

  private:
    typedef RTtransform api_t;
    virtual ~TransformObj() {}
    RTtransform m_transform;
    TransformObj(RTtransform transform) : m_transform(transform) {}
    friend class Handle<TransformObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Selector wraps the OptiX C API RTselector opaque type and its associated function set.
  ///
  class SelectorObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the visitor program for this selector.  See @ref rtSelectorSetVisitProgram
    void setVisitProgram(Program  program);
    /// Get the visitor program for this selector.  See @ref rtSelectorGetVisitProgram
    Program getVisitProgram() const;
    /// @}

    /// @{
    /// Set the number of children for this group.  See @ref rtSelectorSetChildCount.
    void setChildCount(unsigned int  count);
    /// Query the number of children for this group.  See @ref rtSelectorGetChildCount.
    unsigned int getChildCount() const;

    /// Set an indexed child child of this group.  See @ref rtSelectorSetChild
    template< typename T > void setChild(unsigned int index, T child);
    /// Query an indexed child within this group.  See @ref rtSelectorGetChild
    template< typename T > T getChild(unsigned int index) const;
    /// Query indexed child's type.  See @ref rtSelectorGetChildType
    RTobjecttype getChildType(unsigned int index) const;
    
    /// Set a new child in this group and returns its new index.  See @ref rtSelectorSetChild
    template< typename T > unsigned int addChild(T child);
    /// Remove a child in this group and returns the index to the deleted element in case of success.
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    /// Note: this function shifts down all the elements next to the removed one.
    template< typename T > unsigned int removeChild(T child);
    /// Remove a child in this group by its index.
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    /// Note: this function shifts down all the elements next to the removed one.
    void removeChild(int index);
    /// Remove a child in this group by its index.
    /// Throws @ref RT_ERROR_INVALID_VALUE if the parameter is invalid.
    /// Note: this function shifts down all the elements next to the removed one.
    void removeChild(unsigned int index);
    /// Query a child in this group for its index. See @ref rtSelectorGetChild
    template< typename T > unsigned int getChildIndex(T child) const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTselector opaque pointer.
    RTselector get();

  private:
    typedef RTselector api_t;
    virtual ~SelectorObj() {}
    RTselector m_selector;
    SelectorObj(RTselector selector) : m_selector(selector) {}
    friend class Handle<SelectorObj>;
  };

  
  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Acceleration wraps the OptiX C API RTacceleration opaque type and its associated function set.
  ///
  class AccelerationObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Mark the acceleration as needing a rebuild.  See @ref rtAccelerationMarkDirty.
    void markDirty();
    /// Query if the acceleration needs a rebuild.  See @ref rtAccelerationIsDirty.
    bool isDirty() const;
    /// @}

    /// @{
    /// Set properties specifying Acceleration builder behavior.
    /// See @ref rtAccelerationSetProperty.
    void        setProperty( const std::string& name, const std::string& value );
    /// Query properties specifying Acceleration builder behavior.
    /// See @ref rtAccelerationGetProperty.
    std::string getProperty( const std::string& name ) const;

    /// Specify the acceleration structure builder.  See @ref rtAccelerationSetBuilder.
    void        setBuilder(const std::string& builder);
    /// Query the acceleration structure builder.  See @ref rtAccelerationGetBuilder.
    std::string getBuilder() const;
    /// <B>Deprecated in OptiX 4.0</B> Specify the acceleration structure traverser.  See @ref rtAccelerationSetTraverser.
    void        setTraverser(const std::string& traverser);
    /// <B>Deprecated in OptiX 4.0</B> Query the acceleration structure traverser.  See @ref rtAccelerationGetTraverser.
    std::string getTraverser() const;
    /// @}

    /// @{
    /// <B>Deprecated in OptiX 4.0</B> Query the size of the marshaled acceleration data.  See @ref rtAccelerationGetDataSize.
    RTsize getDataSize() const;
    /// <B>Deprecated in OptiX 4.0</B> Get the marshaled acceleration data.  See @ref rtAccelerationGetData.
    void   getData( void* data ) const;
    /// <B>Deprecated in OptiX 4.0</B> Specify the acceleration structure via marshaled acceleration data.  See @ref rtAccelerationSetData.
    void   setData( const void* data, RTsize size );
    /// @}

    /// Get the underlying OptiX C API RTacceleration opaque pointer.
    RTacceleration get();

  private:
    typedef RTacceleration api_t;
    virtual ~AccelerationObj() {}
    RTacceleration m_acceleration;
    AccelerationObj(RTacceleration acceleration) : m_acceleration(acceleration) {}
    friend class Handle<AccelerationObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief GeometryInstance wraps the OptiX C API RTgeometryinstance acceleration
  /// opaque type and its associated function set.
  ///
  class GeometryInstanceObj : public ScopedObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the geometry object associated with this instance.  See @ref rtGeometryInstanceSetGeometry.
    void setGeometry(Geometry  geometry);
    /// Get the geometry object associated with this instance.  See @ref rtGeometryInstanceGetGeometry.
    Geometry getGeometry() const;
    /// Set the GeometryTriangles object associated with this instance.  See @ref rtGeometryInstanceSetGeometryTriangles.
    void setGeometryTriangles( GeometryTriangles geometry );
    /// Get the GeometryTriangles object associated with this instance.  See @ref rtGeometryInstanceGetGeometryTriangles.
    GeometryTriangles getGeometryTriangles() const;

    /// Set the number of materials associated with this instance.  See @ref rtGeometryInstanceSetMaterialCount.
    void setMaterialCount(unsigned int  count);
    /// Query the number of materials associated with this instance.  See @ref rtGeometryInstanceGetMaterialCount.
    unsigned int getMaterialCount() const;

    /// Set the material at given index.  See @ref rtGeometryInstanceSetMaterial.
    void setMaterial(unsigned int idx, Material  material);
    /// Get the material at given index.  See @ref rtGeometryInstanceGetMaterial.
    Material getMaterial(unsigned int idx) const;

    /// Adds the provided material and returns the index to newly added material; increases material count by one.
    unsigned int addMaterial(Material material);
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTgeometryinstance opaque pointer.
    RTgeometryinstance get();

  private:
    typedef RTgeometryinstance api_t;
    virtual ~GeometryInstanceObj() {}
    RTgeometryinstance m_geometryinstance;
    GeometryInstanceObj(RTgeometryinstance geometryinstance) : m_geometryinstance(geometryinstance) {}
    friend class Handle<GeometryInstanceObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Geometry wraps the OptiX C API RTgeometry opaque type and its associated function set.
  /// 
  class GeometryObj : public ScopedObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// <B>Deprecated in OptiX 4.0</B>  See @ref rtGeometryMarkDirty.
    void markDirty();
    /// <B>Deprecated in OptiX 4.0</B>  See @ref rtGeometryIsDirty.
    bool isDirty() const;
    /// @}

    /// @{
    /// Set the number of primitives in this geometry object (eg, number of triangles in mesh).
    /// See @ref rtGeometrySetPrimitiveCount
    void setPrimitiveCount(unsigned int  num_primitives);
    /// Query the number of primitives in this geometry object (eg, number of triangles in mesh).
    /// See @ref rtGeometryGetPrimitiveCount
    unsigned int getPrimitiveCount() const;
    /// @}

    /// @{
    /// Set the primitive index offset for this geometry object.
    /// See @ref rtGeometrySetPrimitiveIndexOffset
    void setPrimitiveIndexOffset(unsigned int  index_offset);
    /// Query the primitive index offset for this geometry object.
    /// See @ref rtGeometryGetPrimitiveIndexOffset
    unsigned int getPrimitiveIndexOffset() const;
    /// @}
    
    /// @{
    /// Set motion time range for this geometry object.
    /// See @ref rtGeometrySetMotionRange
    void setMotionRange( float timeBegin, float timeEnd );
    /// Query the motion time range for this geometry object.
    /// See @ref rtGeometryGetMotionRange
    void getMotionRange( float& timeBegin, float& timeEnd );
    /// Set motion border mode for this geometry object.
    /// See @ref rtGeometrySetMotionBorderMode
    void setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode );
    /// Query the motion border mode for this geometry object.
    /// See @ref rtGeometryGetMotionBorderMode
    void getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode );
    /// Set the number of motion steps for this geometry object.
    /// See @ref rtGeometrySetMotionSteps
    void setMotionSteps( unsigned int n );
    /// Query the number of motion steps for this geometry object.
    /// See @ref rtGeometryGetMotionSteps
    unsigned int getMotionSteps();
    /// @}

    /// @{
    /// Set the bounding box program for this geometry.  See @ref rtGeometrySetBoundingBoxProgram.
    void setBoundingBoxProgram(Program  program);
    /// Get the bounding box program for this geometry.  See @ref rtGeometryGetBoundingBoxProgram.
    Program getBoundingBoxProgram() const;

    /// Set the intersection program for this geometry.  See @ref rtGeometrySetIntersectionProgram.
    void setIntersectionProgram(Program  program);
    /// Get the intersection program for this geometry.  See @ref rtGeometryGetIntersectionProgram.
    Program getIntersectionProgram() const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// @{
    /// See @ref rtGeometrySetFlags
    void setFlags( RTgeometryflags flags );
    RTgeometryflags setFlags() const;
    /// @}

    /// Get the underlying OptiX C API RTgeometry opaque pointer.
    RTgeometry get();

  private:
    typedef RTgeometry api_t;
    virtual ~GeometryObj() {}
    RTgeometry m_geometry;
    GeometryObj(RTgeometry geometry) : m_geometry(geometry) {}
    friend class Handle<GeometryObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief GeometryTriangles wraps the OptiX C API RTgeometrytriangles opaque type and its associated function set.
  /// 
  class GeometryTrianglesObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the primitive index offset for this GeometryTriangles object.
    /// See @ref rtGeometryTrianglesSetPrimitiveIndexOffset
    void setPrimitiveIndexOffset( unsigned int index_offset );
    /// Query the primitive index offset for this GeometryTriangles object.
    /// See @ref rtGeometryTrianglesGetPrimitiveIndexOffset
    unsigned int getPrimitiveIndexOffset() const;
    /// Set the number of triangles in this geometry triangles object.
    /// See @ref rtGeometryTrianglesSetPrimitiveCount
    void setPrimitiveCount( unsigned int num_triangles );
    /// Query the number of triangles in this geometry triangles object.
    /// See @ref rtGeometryTrianglesGetPrimitiveCount
    unsigned int getPrimitiveCount() const;
    /// @}

    /// @{
    void setPreTransformMatrix( bool transpose, const float* matrix );
    void getPreTransformMatrix( bool transpose, float* matrix );
    /// @}

    /// @{
    /// See @ref rtGeometryTrianglesSetTriangleIndices
    void setTriangleIndices( Buffer index_buffer, RTformat tri_indices_format );
    void setTriangleIndices( Buffer index_buffer, RTsize index_buffer_byte_offset, RTformat tri_indices_format );
    void setTriangleIndices( Buffer index_buffer, RTsize index_buffer_byte_offset, RTsize tri_indices_byte_stride, RTformat tri_indices_format );
    /// See @ref rtGeometryTrianglesSetVertices
    void setVertices( unsigned int num_vertices, Buffer vertex_buffer, RTformat position_format );
    void setVertices( unsigned int num_vertices, Buffer vertex_buffer, RTsize vertex_buffer_byte_offset, RTformat position_format );
    void setVertices( unsigned int num_vertices, Buffer vertex_buffer, RTsize vertex_buffer_byte_offset, RTsize vertex_byte_stride, RTformat position_format );

    /// Set the attribute program for this GeometryTriangles object.  See @ref rtGeometryTrianglesSetAttributeProgram
    void setAttributeProgram( Program program );
    /// Get the attribute program for this GeometryTriangles object.  See @ref rtGeometryTrianglesGetAttributeProgram
    Program getAttributeProgram() const;

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// @}

    /// @{
    /// Set motion time range for this geometry triangles object.
    /// See @ref rtGeometryTrianglesSetMotionRange
    void setMotionRange( float timeBegin, float timeEnd );
    /// Query the motion time range for this geometry triangles object.
    /// See @ref rtGeometryTrianglesGetMotionRange
    void getMotionRange( float& timeBegin, float& timeEnd ) const;
    /// Set motion border mode for this geometry triangles object.
    /// See @ref rtGeometryTrianglesSetMotionBorderMode
    void setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode );
    /// Query the motion border mode for this geometry triangles object.
    /// See @ref rtGeometryTrianglesGetMotionBorderMode
    void getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode ) const;
    /// Set the number of motion steps for this geometry triangles object.
    /// See @ref rtGeometryTrianglesSetMotionSteps
    void setMotionSteps( unsigned int n );
    /// Query the number of motion steps for this geometry triangles object.
    /// See @ref rtGeometryTrianglesGetMotionSteps
    unsigned int getMotionSteps() const;

    /// See @ref rtGeometryTrianglesSetMotionVertices
    void setMotionVertices( unsigned int num_vertices,
                            Buffer       vertex_buffer,
                            RTsize       vertex_buffer_byte_offset,
                            RTsize       vertex_byte_stride,
                            RTsize       vertex_motion_step_byte_stride,
                            RTformat     position_format );
    /// See @ref rtGeometryTrianglesSetMotionVerticesMultiBuffer
    template <class BufferIterator>
    void setMotionVerticesMultiBuffer( unsigned int   num_vertices,
                                       BufferIterator vertex_buffers_begin,
                                       BufferIterator vertex_buffers_end,
                                       RTsize         vertex_buffer_byte_offset,
                                       RTsize         vertex_byte_stride,
                                       RTformat       position_format );
    /// @}

    /// @{
    /// See @ref rtGeometryTrianglesSetBuildFlags
    void setBuildFlags( RTgeometrybuildflags build_flags );
    /// See @ref rtGeometryTrianglesSetMaterialCount
    void setMaterialCount( unsigned int num_materials );
    /// See @ref rtGeometryTrianglesGetMaterialCount
    unsigned int getMaterialCount() const;
    /// See @ref rtGeometryTrianglesSetMaterialIndices
    void setMaterialIndices( Buffer   material_index_buffer,
                             RTsize   material_index_buffer_byte_offset,
                             RTsize   material_index_byte_stride,
                             RTformat material_index_format );
    /// See @ref rtGeometryTrianglesSetFlagsPerMaterial
    void setFlagsPerMaterial( unsigned int material_index, RTgeometryflags flags );
    RTgeometryflags getFlagsPerMaterial( unsigned int material_index ) const;
    /// @}

    /// Get the underlying OptiX C API RTgeometrytriangles opaque pointer.
    RTgeometrytriangles get();

  private:
    typedef RTgeometrytriangles api_t;
    virtual ~GeometryTrianglesObj() {}
    RTgeometrytriangles m_geometryTriangles;
    GeometryTrianglesObj(RTgeometrytriangles geometrytriangles) : m_geometryTriangles(geometrytriangles) {}
    friend class Handle<GeometryTrianglesObj>;
  };


  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Material wraps the OptiX C API RTmaterial opaque type and its associated function set.
  ///
  class MaterialObj : public ScopedObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set closest hit program for this material at the given \a ray_type index.  See @ref rtMaterialSetClosestHitProgram.
    void setClosestHitProgram(unsigned int ray_type_index, Program  program);
    /// Get closest hit program for this material at the given \a ray_type index.  See @ref rtMaterialGetClosestHitProgram.
    Program getClosestHitProgram(unsigned int ray_type_index) const;

    /// Set any hit program for this material at the given \a ray_type index.  See @ref rtMaterialSetAnyHitProgram.
    void setAnyHitProgram(unsigned int ray_type_index, Program  program);
    /// Get any hit program for this material at the given \a ray_type index.  See @ref rtMaterialGetAnyHitProgram.
    Program getAnyHitProgram(unsigned int ray_type_index) const;
    /// @}

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    void     removeVariable  (Variable v);
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTmaterial opaque pointer.
    RTmaterial get();
  private:
    typedef RTmaterial api_t;
    virtual ~MaterialObj() {}
    RTmaterial m_material;
    MaterialObj(RTmaterial material) : m_material(material) {}
    friend class Handle<MaterialObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief TextureSampler wraps the OptiX C API RTtexturesampler opaque type and its associated function set.
  ///
  class TextureSamplerObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// <B>Deprecated in OptiX 4.0</B> Set the number of mip levels for this sampler.  See @ref rtTextureSamplerSetMipLevelCount.
    void setMipLevelCount (unsigned int  num_mip_levels);
    /// <B>Deprecated in OptiX 4.0</B> Query the number of mip levels for this sampler.  See @ref rtTextureSamplerGetMipLevelCount.
    unsigned int getMipLevelCount () const;

    /// <B>Deprecated in OptiX 4.0</B> Set the texture array size for this sampler.  See @ref rtTextureSamplerSetArraySize
    void setArraySize(unsigned int  num_textures_in_array);
    /// <B>Deprecated in OptiX 4.0</B> Query the texture array size for this sampler.  See @ref rtTextureSamplerGetArraySize
    unsigned int getArraySize() const;

    /// Set the texture wrap mode for this sampler.  See @ref rtTextureSamplerSetWrapMode
    void setWrapMode(unsigned int dim, RTwrapmode wrapmode);
    /// Query the texture wrap mode for this sampler.  See @ref rtTextureSamplerGetWrapMode
    RTwrapmode getWrapMode(unsigned int dim) const;

    /// Set filtering modes for this sampler.  See @ref rtTextureSamplerSetFilteringModes.
    void setFilteringModes(RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode  mipmapping);
    /// Query filtering modes for this sampler.  See @ref rtTextureSamplerGetFilteringModes.
    void getFilteringModes(RTfiltermode& minification, RTfiltermode& magnification, RTfiltermode& mipmapping) const;

    /// Set maximum anisotropy for this sampler.  See @ref rtTextureSamplerSetMaxAnisotropy.
    void setMaxAnisotropy(float value);
    /// Query maximum anisotropy for this sampler.  See @ref rtTextureSamplerGetMaxAnisotropy.
    float getMaxAnisotropy() const;

    /// Set minimum and maximum mipmap levels for this sampler.  See @ref rtTextureSamplerSetMipLevelClamp.
    void setMipLevelClamp(float minLevel, float maxLevel);
    /// Query minimum and maximum mipmap levels for this sampler.  See @ref rtTextureSamplerGetMipLevelClamp.
    void getMipLevelClamp(float &minLevel, float &maxLevel) const;
        
    /// Set mipmap offset for this sampler.  See @ref rtTextureSamplerSetMipLevelBias.
    void setMipLevelBias(float value);
    /// Query mipmap offset for this sampler.  See @ref rtTextureSamplerGetMipLevelBias.
    float getMipLevelBias() const;

    /// Set texture read mode for this sampler.  See @ref rtTextureSamplerSetReadMode.
    void setReadMode(RTtexturereadmode  readmode);
    /// Query texture read mode for this sampler.  See @ref rtTextureSamplerGetReadMode.
    RTtexturereadmode getReadMode() const;

    /// Set texture indexing mode for this sampler.  See @ref rtTextureSamplerSetIndexingMode.
    void setIndexingMode(RTtextureindexmode  indexmode);
    /// Query texture indexing mode for this sampler.  See @ref rtTextureSamplerGetIndexingMode.
    RTtextureindexmode getIndexingMode() const;
    /// @}

    /// @{
    /// Returns the device-side ID of this sampler. See @ref rtTextureSamplerGetId
    int getId() const;
    /// @}

    /// @{
    /// <B>Deprecated in OptiX 4.0</B> Set the underlying buffer used for texture storage. See @ref rtTextureSamplerSetBuffer.
    void setBuffer(unsigned int texture_array_idx, unsigned int mip_level, Buffer buffer);
    /// <B>Deprecated in OptiX 4.0</B> Get the underlying buffer used for texture storage. See @ref rtTextureSamplerGetBuffer.
    Buffer getBuffer(unsigned int texture_array_idx, unsigned int mip_level) const;
    /// Set the underlying buffer used for texture storage. See @ref rtTextureSamplerSetBuffer.
    void setBuffer(Buffer buffer);
    /// Get the underlying buffer used for texture storage. See @ref rtTextureSamplerGetBuffer.
    Buffer getBuffer() const;
    /// @}

    /// Get the underlying OptiX C API RTtexturesampler opaque pointer.
    RTtexturesampler get();

    /// @{
    /// Declare the texture's buffer as immutable and accessible by OptiX.  See @ref rtTextureSamplerGLRegister.
    void registerGLTexture();
    /// Declare the texture's buffer as mutable and inaccessible by OptiX.  See @ref rtTextureSamplerGLUnregister.
    void unregisterGLTexture();
    /// @}

  private:
    typedef RTtexturesampler api_t;
    virtual ~TextureSamplerObj() {}
    RTtexturesampler m_texturesampler;
    TextureSamplerObj(RTtexturesampler texturesampler) : m_texturesampler(texturesampler) {}
    friend class Handle<TextureSamplerObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief Buffer wraps the OptiX C API RTbuffer opaque type and its associated function set.
  ///
  class BufferObj : public DestroyableObj {
  public:
    void destroy();
    void validate();
    Context getContext() const;

    /// @{
    /// Set the data format for the buffer.  See @ref rtBufferSetFormat.
    void setFormat    (RTformat format);
    /// Query the data format for the buffer.  See @ref rtBufferGetFormat.
    RTformat getFormat() const;

    /// Set the data element size for user format buffers.  See @ref rtBufferSetElementSize.
    void setElementSize  (RTsize size_of_element);
    /// Query the data element size for user format buffers.  See @ref rtBufferGetElementSize.
    RTsize getElementSize() const;

    /// Get the pointer to buffer memory on a specific device. See @ref rtBufferGetDevicePointer
    void getDevicePointer( int optix_device_ordinal, void** device_pointer );
    void* getDevicePointer( int optix_device_ordinal );

    /// Set the pointer to buffer memory on a specific device. See @ref rtBufferSetDevicePointer
    void setDevicePointer( int optix_device_ordinal, void* device_pointer );

    /// Mark the buffer dirty
    void markDirty();

    /// Set buffer dimensionality to one and buffer width to specified width.  See @ref rtBufferSetSize1D.
    void setSize(RTsize  width);
    /// Query 1D buffer dimension.  See @ref rtBufferGetSize1D.
    void getSize(RTsize& width) const;
    /// Query 1D buffer dimension of specific MIP level.  See @ref rtBufferGetMipLevelSize1D.
    void getMipLevelSize(unsigned int level, RTsize& width) const;
    /// Set buffer dimensionality to two and buffer dimensions to specified width,height.  See @ref rtBufferSetSize2D.
    void setSize(RTsize  width, RTsize  height);
    /// Query 2D buffer dimension.  See @ref rtBufferGetSize2D.
    void getSize(RTsize& width, RTsize& height) const;
    /// Query 2D buffer dimension of specific MIP level.  See @ref rtBufferGetMipLevelSize2D.
    void getMipLevelSize(unsigned int level, RTsize& width, RTsize& height) const;
    /// Set buffer dimensionality to three and buffer dimensions to specified width,height,depth.
    /// See @ref rtBufferSetSize3D.
    void setSize(RTsize  width, RTsize  height, RTsize  depth);
     /// Query 3D buffer dimension.  See @ref rtBufferGetSize3D.
    void getSize(RTsize& width, RTsize& height, RTsize& depth) const;
    /// Query 3D buffer dimension of specific MIP level.  See @ref rtBufferGetMipLevelSize3D.
    void getMipLevelSize(unsigned int level, RTsize& width, RTsize& height, RTsize& depth) const;

    /// Set buffer dimensionality and dimensions to specified values. See @ref rtBufferSetSizev.
    void setSize(unsigned int dimensionality, const RTsize* dims);
    /// Query dimensions of buffer.  See @ref rtBufferGetSizev.
    void getSize(unsigned int dimensionality,       RTsize* dims) const;

    /// Query dimensionality of buffer.  See @ref rtBufferGetDimensionality.
    unsigned int getDimensionality() const;
   
    /// Set buffer number of MIP levels. See @ref rtBufferSetMipLevelCount.
    void setMipLevelCount(unsigned int levels);
    /// Query number of mipmap levels of buffer.  See @ref rtBufferGetMipLevelCount.
    unsigned int getMipLevelCount() const;
    /// @}

    /// @{
    /// Queries an id suitable for referencing the buffer in an another buffer.  See @ref rtBufferGetId.
    int getId() const;
    /// @}

    /// @{
    /// Queries the OpenGL Buffer Object ID associated with this buffer.  See @ref rtBufferGetGLBOId.
    unsigned int getGLBOId() const;

    /// Declare the buffer as mutable and inaccessible by OptiX.  See @ref rtTextureSamplerGLRegister.
    void registerGLBuffer();
    /// Unregister the buffer, re-enabling OptiX operations.  See @ref rtTextureSamplerGLUnregister.
    void unregisterGLBuffer();
    /// @}

    /// @{
    /// Set a Buffer Attribute. See @ref rtBufferSetAttribute.
    void setAttribute( RTbufferattribute attrib, RTsize size, const void *p );
    /// Get a Buffer Attribute. See @ref rtBufferGetAttribute.
    void getAttribute( RTbufferattribute attrib, RTsize size, void *p );
    /// @}

    /// @{
    /// Maps a buffer object for host access.  See @ref rtBufferMap and @ref rtBufferMapEx.
    void* map( unsigned int level=0, unsigned int map_flags=RT_BUFFER_MAP_READ_WRITE, void* user_owned=0 );

    /// Unmaps a buffer object.  See @ref rtBufferUnmap and @ref rtBufferUnmapEx.
    void unmap( unsigned int level=0 );
    /// @}

    /// @{
    /// Bind a buffer as source for a progressive stream. See @ref rtBufferBindProgressiveStream.
    void bindProgressiveStream( Buffer source );
    /// Query updates from a progressive stream. See @ref rtBufferGetProgressiveUpdateReady.
    void getProgressiveUpdateReady( int* ready, unsigned int* subframe_count, unsigned int* max_subframes );

    /// Query updates from a progressive stream. See @ref rtBufferGetProgressiveUpdateReady.
    bool getProgressiveUpdateReady();

    /// Query updates from a progressive stream. See @ref rtBufferGetProgressiveUpdateReady.
    bool getProgressiveUpdateReady( unsigned int& subframe_count );

    /// Query updates from a progressive stream. See @ref rtBufferGetProgressiveUpdateReady.
    bool getProgressiveUpdateReady( unsigned int& subframe_count, unsigned int& max_subframes );
    /// @}

    /// Get the underlying OptiX C API RTbuffer opaque pointer.
    RTbuffer get();

  private:
    typedef RTbuffer api_t;
    virtual ~BufferObj() {}
    RTbuffer m_buffer;
    BufferObj(RTbuffer buffer) : m_buffer(buffer) {}
    friend class Handle<BufferObj>;
  };

#if !defined(__CUDACC__)
  ///
  /// \brief bufferId is a host version of the device side bufferId.
  ///
  /// Use bufferId to define types that can be included from both the host and device
  /// code.  This class provides a container that can be used to transport the buffer id
  /// back and forth between host and device code.  The bufferId class is useful, because
  /// it can take a buffer id obtained from rtBufferGetId and provide accessors similar to
  /// the buffer class.
  ///
  /// "bindless_type.h" used by both host and device code:
  ///
  /// @code
  /// #include <optix_world.h>
  /// struct BufInfo {
  ///   int val;
  ///   rtBufferId<int, 1> data;
  /// };
  /// @endcode
  ///
  /// Host code:
  ///
  /// @code
  /// #include "bindless_type.h"
  /// BufInfo input_buffer_info;
  /// input_buffer_info.val = 0;
  /// input_buffer_info.data = rtBufferId<int,1>(inputBuffer0->getId());
  /// context["input_buffer_info"]->setUserData(sizeof(BufInfo), &input_buffer_info);
  /// @endcode
  ///
  /// Device code:
  ///
  /// @code
  /// #include "bindless_type.h"
  /// rtBuffer<int,1> result;
  /// rtDeclareVariable(BufInfo, input_buffer_info, ,);
  /// 
  /// RT_PROGRAM void bindless()
  /// {
  ///   int value = input_buffer_info.data[input_buffer_info.val];
  ///   result[0] = value;
  /// }
  /// @endcode
  /// 
  /// 
  template<typename T, int Dim=1>
  struct bufferId {
    bufferId() {}
    bufferId(int id) : m_id(id) {}
    int getId() const { return m_id; }
  private:
    int m_id;
  };
#define rtBufferId optix::bufferId

  ///
  /// \brief callableProgramId is a host version of the device side callableProgramId.
  ///
  /// Use callableProgramId to define types that can be included from both the host and
  /// device code.  This class provides a container that can be used to transport the
  /// program id back and forth between host and device code.  The callableProgramId class
  /// is useful, because it can take a program id obtained from rtProgramGetId and provide
  /// accessors for calling the program corresponding to the program id.
  ///
  /// "bindless_type.h" used by both host and device code:
  ///
  /// @code
  /// #include <optix_world.h>
  /// struct ProgramInfo {
  ///   int val;
  ///   rtProgramId<int(int)> program;
  /// };
  /// @endcode
  ///
  /// Host code:
  ///
  /// @code
  /// #include "bindless_type.h"
  /// ProgramInfo input_program_info;
  /// input_program_info.val = 0;
  /// input_program_info.program = rtCallableProgramId<int(int)>(inputProgram0->getId());
  /// context["input_program_info"]->setUserData(sizeof(ProgramInfo), &input_program_info);
  /// @endcode
  ///
  /// Device code:
  ///
  /// @code
  /// #include "bindless_type.h"
  /// rtBuffer<int,1> result;
  /// rtDeclareVariable(ProgramInfo, input_program_info, ,);
  /// 
  /// RT_PROGRAM void bindless()
  /// {
  ///   int value = input_program_info.program(input_program_info.val);
  ///   result[0] = value;
  /// }
  /// @endcode
  /// 
  ///

#define RT_INTERNAL_CALLABLE_PROGRAM_DEFS()   \
  {                                           \
  public:                                     \
    callableProgramId() {}                    \
    callableProgramId(int id) : m_id(id) {}   \
    int getId() const { return m_id; }        \
  private:                                    \
    int m_id;                                 \
  }
  
  // 
  template<typename T>
  class callableProgramId;
  template<typename ReturnT>
  class callableProgramId<ReturnT()> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T>
  class callableProgramId<ReturnT(Arg0T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T>
  class callableProgramId<ReturnT(Arg0T,Arg1T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T,
           typename Arg4T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T,Arg4T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T,
           typename Arg4T, typename Arg5T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T,Arg4T,Arg5T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T,
           typename Arg4T, typename Arg5T, typename Arg6T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T,Arg4T,Arg5T,Arg6T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T,
           typename Arg4T, typename Arg5T, typename Arg6T, typename Arg7T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T,Arg4T,Arg5T,Arg6T,Arg7T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T,
           typename Arg4T, typename Arg5T, typename Arg6T, typename Arg7T, typename Arg8T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T,Arg4T,Arg5T,Arg6T,Arg7T,Arg8T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();
  template<typename ReturnT, typename Arg0T, typename Arg1T, typename Arg2T, typename Arg3T,
           typename Arg4T, typename Arg5T, typename Arg6T, typename Arg7T, typename Arg8T, typename Arg9T>
  class callableProgramId<ReturnT(Arg0T,Arg1T,Arg2T,Arg3T,Arg4T,Arg5T,Arg6T,Arg7T,Arg8T,Arg9T)> RT_INTERNAL_CALLABLE_PROGRAM_DEFS();

#define rtCallableProgramId    optix::callableProgramId

  template<typename T>
  class markedCallableProgramId
  {
      // This class is not available on the host and will produce
      // a compile error if it is used.
      template <bool>
      struct Fail;
  public:
      markedCallableProgramId() { Fail<true> rtMarkedCallableProgramId_is_only_valid_in_device_code_use_rtCallableProgramId_instead; }
      markedCallableProgramId(int) { Fail<true> rtMarkedCallableProgramId_is_only_valid_in_device_code_use_rtCallableProgramId_instead; }
  };

#define rtMarkedCallableProgramId optix::markedCallableProgramId

#endif


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief PostProcessingStage wraps the OptiX C API RTpostprocessingstage opaque type and its associated function set.
  ///
  class PostprocessingStageObj : public DestroyableObj {
  public:
    void destroy();
    void validate();

    Context getContext() const;

    /// @{
    Variable declareVariable (const std::string& name);
    Variable queryVariable   (const std::string& name) const;
    unsigned int getVariableCount() const;
    Variable getVariable     (unsigned int index) const;
    /// @}

    /// Get the underlying OptiX C API RTpostprocessingstage opaque pointer.
    RTpostprocessingstage get();

  private:
    typedef RTpostprocessingstage api_t;
    virtual ~PostprocessingStageObj() {}
    RTpostprocessingstage m_stage;
    PostprocessingStageObj(RTpostprocessingstage stage) : m_stage(stage) {}
    friend class Handle<PostprocessingStageObj>;
  };


  //----------------------------------------------------------------------------


  /// \ingroup optixpp
  ///
  /// \brief CommandList wraps the OptiX C API RTcommandlist opaque type and its associated function set.
  ///
  class CommandListObj : public DestroyableObj {
  public:
    void destroy();
    void validate();

    Context getContext() const;

    /// @{
    /// Append a postprocessing stage to the command list. See @ref rtCommandListAppendPostprocessingStage.
    void appendPostprocessingStage(PostprocessingStage stage, RTsize launch_width, RTsize launch_height);

    /// Append a 1D launch to the command list. See @ref rtCommandListAppendLaunch1D.
    void appendLaunch(unsigned int entryIndex, RTsize launch_width);

    /// Append a 2D launch to the command list. See @ref rtCommandListAppendLaunch2D.
    void appendLaunch(unsigned int entryIndex, RTsize launch_width, RTsize launch_height);

    /// Append a 3D launch to the command list. See @ref rtCommandListAppendLaunch3D.
    void appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height, RTsize launch_depth );
    /// @}

    /// @{
    /// Finalize the command list so that it can be called, later. See @ref rtCommandListFinalize.
    void finalize();
    /// @}

    /// @{
    /// See @ref rtCommandListSetDevices. Sets the devices to use for this command list.
    template<class Iterator>
    void setDevices( Iterator begin, Iterator end );

    /// See @ref rtContextGetDevices. Returns the list of devices set for this command list.
    std::vector<int> getDevices() const;
    /// @}

    /// @{
    // Excecute the command list. Can only be called after finalizing it. See @ref rtCommandListExecute.
    void execute();
    /// @}

    /// Get the underlying OptiX C API RTcommandlist opaque pointer.
    RTcommandlist get();

    /// @{
    /// Sets the cuda stream for this command list. See @ref rtCommandListSetCudaStream.
    void setCudaStream( void* stream );

    /// Gets the cuda stream set for this command list. See @ref rtCommandListGetCudaStream.
    void getCudaStream( void** stream );
    /// @}

  private:
    typedef RTcommandlist api_t;
    virtual ~CommandListObj() {}
    RTcommandlist m_list;
    CommandListObj(RTcommandlist list) : m_list(list) {}
    friend class Handle<CommandListObj>;
  };


  //----------------------------------------------------------------------------

  inline void APIObj::checkError( RTresult code ) const
  {
    if( code != RT_SUCCESS) {
      RTcontext c = this->getContext()->get();
      throw Exception::makeException( code, c );
    }
  }

  inline void APIObj::checkError( RTresult code, Context context ) const
  {
    if( code != RT_SUCCESS) {
      RTcontext c = context->get();
      throw Exception::makeException( code, c );
    }
  }

  inline void APIObj::checkErrorNoGetContext( RTresult code ) const
  {
    if( code != RT_SUCCESS) {
      throw Exception::makeException( code, 0u );
    }
  }

  inline Context ContextObj::getContext() const
  {
    return Context::take( m_context );
  }

  inline void ContextObj::checkError(RTresult code) const
  {
    if( code != RT_SUCCESS && code != RT_TIMEOUT_CALLBACK )
      throw Exception::makeException( code, m_context );
  }

  inline unsigned int ContextObj::getDeviceCount()
  {
    unsigned int count;
    if( RTresult code = rtDeviceGetDeviceCount(&count) )
      throw Exception::makeException( code, 0 );

    return count;
  }

  inline std::string ContextObj::getDeviceName(int ordinal)
  {
    const RTsize max_string_size = 256;
    char name[max_string_size];
    if( RTresult code = rtDeviceGetAttribute(ordinal, RT_DEVICE_ATTRIBUTE_NAME,
                                             max_string_size, name) )
      throw Exception::makeException( code, 0 );
    return std::string(name);
  }

  inline std::string ContextObj::getDevicePCIBusId(int ordinal)
  {
    const RTsize max_string_size = 16;  // at least 13, e.g., "0000:01:00.0" with NULL-terminator
    char pciBusId[max_string_size];
    if( RTresult code = rtDeviceGetAttribute(ordinal, RT_DEVICE_ATTRIBUTE_PCI_BUS_ID,
                                             max_string_size, pciBusId) )
      throw Exception::makeException( code, 0 );
    return std::string(pciBusId);
  }

  inline std::vector<int> ContextObj::getCompatibleDevices(int ordinal)
  {
    std::vector<int> compatibleDevices;
    compatibleDevices.resize( ContextObj::getDeviceCount() + 1 );
    if( RTresult code = rtDeviceGetAttribute( ordinal, RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES,
                                              compatibleDevices.size() * sizeof( int ), &compatibleDevices[0] ) )
        throw Exception::makeException( code, 0 );
    std::vector<int>::const_iterator firstOrdinal = compatibleDevices.begin() + 1;
    return std::vector<int>( firstOrdinal, firstOrdinal + compatibleDevices[0] );
  }

  inline void ContextObj::getDeviceAttribute(int ordinal, RTdeviceattribute attrib, RTsize size, void* p)
  {
    if( RTresult code = rtDeviceGetAttribute(ordinal, attrib, size, p) )
      throw Exception::makeException( code, 0 );
  }

  inline Context ContextObj::create()
  {
    RTcontext c;
    if( RTresult code = rtContextCreate(&c) )
      throw Exception::makeException( code, 0 );

    return Context::take(c);
  }

  inline void ContextObj::destroy()
  {
    checkErrorNoGetContext( rtContextDestroy( m_context ) );
    m_context = 0;
  }

  inline void ContextObj::validate()
  {
    checkError( rtContextValidate( m_context ) );
  }

  inline Acceleration ContextObj::createAcceleration(const std::string& builder, const std::string& /*traverser*/)
  {
    RTacceleration acceleration;
    checkError( rtAccelerationCreate( m_context, &acceleration ) );
    checkError( rtAccelerationSetBuilder( acceleration, builder.c_str() ) );
    return Acceleration::take(acceleration);
  }


  inline Buffer ContextObj::createBuffer(unsigned int type)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize1D( buffer, width ) );
    return Buffer::take(buffer);
  }
    
  inline Buffer ContextObj::createMipmappedBuffer(unsigned int type, RTformat format, RTsize width, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize1D( buffer, width ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize2D( buffer, width, height ) );
    return Buffer::take(buffer);
  }
    
  inline Buffer ContextObj::createMipmappedBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize2D( buffer, width, height ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize3D( buffer, width, height, depth ) );
    return Buffer::take(buffer);
  }
    
  inline Buffer ContextObj::createMipmappedBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize3D( buffer, width, height, depth ) );
    return Buffer::take(buffer);
  }
   
  inline Buffer ContextObj::create1DLayeredBuffer(unsigned int type, RTformat format, RTsize width, RTsize layers, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type | RT_BUFFER_LAYERED, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize3D( buffer, width, 1, layers ) );
    return Buffer::take(buffer);
  }
    
  inline Buffer ContextObj::create2DLayeredBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize layers, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type | RT_BUFFER_LAYERED, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize3D( buffer, width, height, layers ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createCubeBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type | RT_BUFFER_CUBEMAP, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize3D( buffer, width, height, 6 ) );
    return Buffer::take(buffer);
  }
    
  inline Buffer ContextObj::createCubeLayeredBuffer(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize faces, unsigned int levels)
  {
    RTbuffer buffer;
    checkError( rtBufferCreate( m_context, type | RT_BUFFER_CUBEMAP | RT_BUFFER_LAYERED, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetMipLevelCount( buffer, levels ) );
    checkError( rtBufferSetSize3D( buffer, width, height, faces ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format, RTsize width)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize1D( buffer, width ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize2D( buffer, width, height ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferForCUDA(unsigned int type, RTformat format, RTsize width, RTsize height, RTsize depth)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateForCUDA( m_context, type, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize3D( buffer, width, height, depth ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromGLBO(unsigned int type, unsigned int vbo)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromGLBO( m_context, type, vbo, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromCallback( m_context, type, callback, callbackData, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromCallback( m_context, type, callback, callbackData, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format, RTsize width)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromCallback( m_context, type, callback, callbackData, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize1D( buffer, width ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format, RTsize width, RTsize height)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromCallback( m_context, type, callback, callbackData, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize2D( buffer, width, height ) );
    return Buffer::take(buffer);
  }

  inline Buffer ContextObj::createBufferFromCallback(unsigned int type, RTbuffercallback callback, void* callbackData, RTformat format, RTsize width, RTsize height, RTsize depth)
  {
    RTbuffer buffer;
    checkError( rtBufferCreateFromCallback( m_context, type, callback, callbackData, &buffer ) );
    checkError( rtBufferSetFormat( buffer, format ) );
    checkError( rtBufferSetSize3D( buffer, width, height, depth ) );
    return Buffer::take(buffer);
  }

  inline TextureSampler ContextObj::createTextureSamplerFromGLImage(unsigned int id, RTgltarget target)
  {
    RTtexturesampler textureSampler;
    checkError( rtTextureSamplerCreateFromGLImage(m_context, id, target, &textureSampler));
    return TextureSampler::take(textureSampler);
  }

  inline Buffer ContextObj::getBufferFromId(int buffer_id)
  {
    RTbuffer buffer;
    checkError( rtContextGetBufferFromId( m_context, buffer_id, &buffer ) );
    return Buffer::take(buffer);
  }

  inline Program ContextObj::getProgramFromId(int program_id)
  {
    RTprogram program;
    checkError( rtContextGetProgramFromId( m_context, program_id, &program ) );
    return Program::take(program);
  }

  inline TextureSampler ContextObj::getTextureSamplerFromId(int sampler_id)
  {
    RTtexturesampler sampler;
    checkError( rtContextGetTextureSamplerFromId( m_context, sampler_id, &sampler ) );
    return TextureSampler::take(sampler);
  }

  inline Geometry ContextObj::createGeometry()
  {
    RTgeometry geometry;
    checkError( rtGeometryCreate( m_context, &geometry ) );
    return Geometry::take(geometry);
  }

  inline GeometryTriangles ContextObj::createGeometryTriangles()
  {
    RTgeometrytriangles geom_tris;
    checkError( rtGeometryTrianglesCreate( m_context, &geom_tris ) );
    return GeometryTriangles::take(geom_tris);
  }

  inline GeometryInstance ContextObj::createGeometryInstance()
  {
    RTgeometryinstance geometryinstance;
    checkError( rtGeometryInstanceCreate( m_context, &geometryinstance ) );
    return GeometryInstance::take(geometryinstance);
  }

  template<class Iterator>
  GeometryInstance ContextObj::createGeometryInstance( Geometry geometry, Iterator matlbegin, Iterator matlend)
  {
    GeometryInstance result = createGeometryInstance();
    result->setGeometry( geometry );
    unsigned int count = 0;
    for( Iterator iter = matlbegin; iter != matlend; ++iter )
      ++count;
    result->setMaterialCount( count );
    unsigned int index = 0;
    for(Iterator iter = matlbegin; iter != matlend; ++iter, ++index )
      result->setMaterial( index, *iter );
    return result;
  }

  template<class Iterator>
  GeometryInstance ContextObj::createGeometryInstance( GeometryTriangles geometrytriangles, Iterator matlbegin, Iterator matlend)
  {
    GeometryInstance result = createGeometryInstance();
    result->setGeometryTriangles( geometrytriangles );
    unsigned int count = 0;
    for( Iterator iter = matlbegin; iter != matlend; ++iter )
      ++count;
    result->setMaterialCount( count );
    unsigned int index = 0;
    for(Iterator iter = matlbegin; iter != matlend; ++iter, ++index )
      result->setMaterial( index, *iter );
    return result;
  }

  inline GeometryInstance ContextObj::createGeometryInstance( GeometryTriangles geometrytriangles, Material mat )
  {
    GeometryInstance result = createGeometryInstance();
    result->setGeometryTriangles( geometrytriangles );
    result->setMaterialCount( 1 );
    result->setMaterial( 0, mat );
    return result;
  }

  inline Group ContextObj::createGroup()
  {
    RTgroup group;
    checkError( rtGroupCreate( m_context, &group ) );
    return Group::take(group);
  }

  template<class Iterator>
    inline Group ContextObj::createGroup( Iterator childbegin, Iterator childend )
  {
    Group result = createGroup();
    unsigned int count = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter )
      ++count;
    result->setChildCount( count );
    unsigned int index = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter, ++index )
      result->setChild( index, *iter );
    return result;
  }

  inline GeometryGroup ContextObj::createGeometryGroup()
  {
    RTgeometrygroup gg;
    checkError( rtGeometryGroupCreate( m_context, &gg ) );
    return GeometryGroup::take( gg );
  }

  template<class Iterator>
  inline GeometryGroup ContextObj::createGeometryGroup( Iterator childbegin, Iterator childend )
  {
    GeometryGroup result = createGeometryGroup();
    unsigned int count = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter )
      ++count;
    result->setChildCount( count );
    unsigned int index = 0;
    for(Iterator iter = childbegin; iter != childend; ++iter, ++index )
      result->setChild( index, *iter );
    return result;
  }

  inline Transform ContextObj::createTransform()
  {
    RTtransform t;
    checkError( rtTransformCreate( m_context, &t ) );
    return Transform::take( t );
  }

  inline Material ContextObj::createMaterial()
  {
    RTmaterial material;
    checkError( rtMaterialCreate( m_context, &material ) );
    return Material::take(material);
  }

  inline Program ContextObj::createProgramFromPTXFile( const std::string& filename, const std::string& program_name )
  {
    RTprogram program;
    checkError( rtProgramCreateFromPTXFile( m_context, filename.c_str(), program_name.c_str(), &program ) );
    return Program::take(program);
  }

  inline Program ContextObj::createProgramFromPTXFiles( const std::vector<std::string>& filenames, const std::string& program_name )
  {
    std::vector<const char*> cstrings( filenames.size() );
    for ( size_t i = 0; i < filenames.size(); ++i )
    {
        cstrings[i] = filenames[i].c_str();
    }
    return createProgramFromPTXFiles( cstrings, program_name );
  }

  inline Program ContextObj::createProgramFromPTXFiles( const std::vector<const char*>& filenames, const std::string& program_name )
  {
    RTprogram program;
    unsigned int n = static_cast<unsigned int>(filenames.size());
    checkError( rtProgramCreateFromPTXFiles( m_context, n, const_cast<const char**>(&filenames[0]), program_name.c_str(), &program ) );
    return Program::take( program );
  }

  inline Program ContextObj::createProgramFromPTXString( const std::string& ptx, const std::string& program_name )
  {
    RTprogram program;
    checkError( rtProgramCreateFromPTXString( m_context, ptx.c_str(), program_name.c_str(), &program ) );
    return Program::take(program);
  }

  inline Program ContextObj::createProgramFromPTXStrings( const std::vector<std::string>& ptxStrings, const std::string& program_name )
  {
    std::vector<const char*> cstrings( ptxStrings.size() );
    for ( size_t i = 0; i < ptxStrings.size(); ++i )
    {
        cstrings[i] = ptxStrings[i].c_str();
    }
    return createProgramFromPTXStrings( cstrings, program_name );
  }

  inline Program ContextObj::createProgramFromPTXStrings( const std::vector<const char*>& ptxStrings, const std::string& program_name )
  {
    RTprogram program;
    unsigned int n = static_cast<unsigned int>(ptxStrings.size());
    checkError( rtProgramCreateFromPTXStrings( m_context, n, const_cast<const char**>(&ptxStrings[0]), program_name.c_str(), &program ) );
    return Program::take( program );
  }

  inline Program ContextObj::createProgramFromProgram( Program program_in )
  {
    RTprogram program;
    checkError( rtProgramCreateFromProgram( m_context, program_in->get(), &program ) );
    return Program::take( program );
  }

  inline Selector ContextObj::createSelector()
  {
    RTselector selector;
    checkError( rtSelectorCreate( m_context, &selector ) );
    return Selector::take(selector);
  }

  inline TextureSampler ContextObj::createTextureSampler()
  {
    RTtexturesampler texturesampler;
    checkError( rtTextureSamplerCreate( m_context, &texturesampler ) );
    return TextureSampler::take(texturesampler);
  }

  inline PostprocessingStage ContextObj::createBuiltinPostProcessingStage(const std::string & builtin_name)
  {
      RTpostprocessingstage stage;
      checkError( rtPostProcessingStageCreateBuiltin(m_context, builtin_name.c_str(), &stage) );
      return PostprocessingStage::take(stage);
  }

  inline CommandList ContextObj::createCommandList()
  {
    RTcommandlist cl;
    checkError( rtCommandListCreate(m_context, &cl) );
    return CommandList::take(cl);
  }

  inline std::string ContextObj::getErrorString( RTresult code ) const
  {
    const char* str;
    rtContextGetErrorString( m_context, code, &str);
    return std::string(str);
  }

  template<class Iterator> inline
    void ContextObj::setDevices(Iterator begin, Iterator end)
  {
    std::vector<int> devices( begin, end );
    checkError( rtContextSetDevices( m_context, static_cast<unsigned int>(devices.size()), &devices[0]) );
  }

  inline std::vector<int> ContextObj::getEnabledDevices() const
  {
    // Initialize with the number of enabled devices
    std::vector<int> devices(getEnabledDeviceCount());
    checkError( rtContextGetDevices( m_context, &devices[0] ) );
    return devices;
  }

  inline unsigned int ContextObj::getEnabledDeviceCount() const
  {
    unsigned int num;
    checkError( rtContextGetDeviceCount( m_context, &num ) );
    return num;
  }
  
  inline int ContextObj::getMaxTextureCount() const
  {
    int tex_count;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_MAX_TEXTURE_COUNT, sizeof(tex_count), &tex_count) );
    return tex_count;
  }

  inline int ContextObj::getCPUNumThreads() const
  {
    int cpu_num_threads;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(cpu_num_threads), &cpu_num_threads) );
    return cpu_num_threads;
  }

  inline RTsize ContextObj::getUsedHostMemory() const
  {
    RTsize used_mem;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_USED_HOST_MEMORY, sizeof(used_mem), &used_mem) );
    return used_mem;
  }

  inline bool ContextObj::getPreferFastRecompiles() const
  {
    int enabled;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES, sizeof(enabled), &enabled) );
    return enabled != 0;
  }

  inline bool ContextObj::getForceInlineUserFunctions() const
  {
    int enabled;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS, sizeof(enabled), &enabled) );
    return enabled != 0;
  }

  inline int ContextObj::getGPUPagingActive() const
  {
    int gpu_paging_active;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_ACTIVE, sizeof(gpu_paging_active), &gpu_paging_active) );
    return gpu_paging_active;
  }

  inline int ContextObj::getGPUPagingForcedOff() const
  {
    int gpu_paging_forced_off;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(gpu_paging_forced_off), &gpu_paging_forced_off) );
    return gpu_paging_forced_off;
  }

  inline RTsize ContextObj::getAvailableDeviceMemory(int ordinal) const
  {
    RTsize free_mem;
    checkError( rtContextGetAttribute( m_context,
                                       static_cast<RTcontextattribute>(RT_CONTEXT_ATTRIBUTE_AVAILABLE_DEVICE_MEMORY + ordinal),
                                       sizeof(free_mem), &free_mem) );
    return free_mem;
  }

  inline void ContextObj::setCPUNumThreads(int cpu_num_threads)
  {
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(cpu_num_threads), &cpu_num_threads) );
  }

  inline void ContextObj::setPreferFastRecompiles( bool enabled )
  {
    int value = enabled;
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_PREFER_FAST_RECOMPILES, sizeof(value), &value ) );
  }

  inline void ContextObj::setForceInlineUserFunctions( bool enabled )
  {
    int value = enabled;
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_FORCE_INLINE_USER_FUNCTIONS, sizeof(value), &value ) );
  }

  inline void ContextObj::setDiskCacheLocation( const std::string& path )
  {
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION, sizeof(path.c_str()), path.c_str() ) );
  } 

  inline std::string ContextObj::getDiskCacheLocation()
  {
    char* str = 0;
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_LOCATION, sizeof(char**), &str ) );
    return std::string( str );
  }

  inline void ContextObj::setDiskCacheMemoryLimits( RTsize lowWaterMark, RTsize highWaterMark )
  {
    RTsize limits[2] = { lowWaterMark, highWaterMark };
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS,
                                       sizeof( limits ), limits ) );
  }

  inline void ContextObj::getDiskCacheMemoryLimits( RTsize& lowWaterMark, RTsize& highWaterMark )
  {
    RTsize limits[2] = { 0 };
    checkError( rtContextGetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_DISK_CACHE_MEMORY_LIMITS,
                                       sizeof( limits ), limits ) );
    lowWaterMark  = limits[0];
    highWaterMark = limits[1];
  }

  inline void ContextObj::setGPUPagingForcedOff(int gpu_paging_forced_off)
  {
    checkError( rtContextSetAttribute( m_context, RT_CONTEXT_ATTRIBUTE_GPU_PAGING_FORCED_OFF, sizeof(gpu_paging_forced_off), &gpu_paging_forced_off) );
  }

  template<class T>
  inline void ContextObj::setAttribute(RTcontextattribute attribute, const T& val)
  {
    checkError( rtContextSetAttribute( m_context, attribute, sizeof(T), &val) );
  }

  inline void ContextObj::setStackSize(RTsize  stack_size_bytes)
  {
    checkError(rtContextSetStackSize(m_context, stack_size_bytes) );
  }

  inline RTsize ContextObj::getStackSize() const
  {
    RTsize result;
    checkError( rtContextGetStackSize( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setMaxCallableProgramDepth(unsigned int  max_depth)
  {
    checkError(rtContextSetMaxCallableProgramDepth( m_context, max_depth ) );
  }

  inline unsigned int ContextObj::getMaxCallableProgramDepth() const
  {
    unsigned int result;
    checkError( rtContextGetMaxCallableProgramDepth( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setMaxTraceDepth(unsigned int  max_depth)
  {
    checkError( rtContextSetMaxTraceDepth( m_context, max_depth ) );
  }

  inline unsigned int ContextObj::getMaxTraceDepth() const
  {
    unsigned int result;
    checkError( rtContextGetMaxTraceDepth( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setTimeoutCallback(RTtimeoutcallback callback, double min_polling_seconds)
  {
    checkError( rtContextSetTimeoutCallback( m_context, callback, min_polling_seconds ) );
  }
    
  inline void ContextObj::setUsageReportCallback(RTusagereportcallback callback, int verbosity, void* cbdata)
  {
    checkError( rtContextSetUsageReportCallback( m_context, callback, verbosity, cbdata) );
  }

  inline void ContextObj::setEntryPointCount(unsigned int  num_entry_points)
  {
    checkError( rtContextSetEntryPointCount( m_context, num_entry_points ) );
  }

  inline unsigned int ContextObj::getEntryPointCount() const
  {
    unsigned int result;
    checkError( rtContextGetEntryPointCount( m_context, &result ) );
    return result;
  }


  inline void ContextObj::setRayGenerationProgram(unsigned int entry_point_index, Program  program)
  {
    checkError( rtContextSetRayGenerationProgram( m_context, entry_point_index, program ? program->get() : 0 ) );
  }

  inline Program ContextObj::getRayGenerationProgram(unsigned int entry_point_index) const
  {
    RTprogram result;
    checkError( rtContextGetRayGenerationProgram( m_context, entry_point_index, &result ) );
    return Program::take( result );
  }


  inline void ContextObj::setExceptionProgram(unsigned int entry_point_index, Program  program)
  {
    checkError( rtContextSetExceptionProgram( m_context, entry_point_index, program ? program->get() : 0 ) );
  }

  inline Program ContextObj::getExceptionProgram(unsigned int entry_point_index) const
  {
    RTprogram result;
    checkError( rtContextGetExceptionProgram( m_context, entry_point_index, &result ) );
    return Program::take( result );
  }


  inline void ContextObj::setExceptionEnabled( RTexception exception, bool enabled )
  {
    checkError( rtContextSetExceptionEnabled( m_context, exception, enabled ) );
  }

  inline bool ContextObj::getExceptionEnabled( RTexception exception ) const
  {
    int enabled;
    checkError( rtContextGetExceptionEnabled( m_context, exception, &enabled ) );
    return enabled != 0;
  }


  inline void ContextObj::setRayTypeCount(unsigned int  num_ray_types)
  {
    checkError( rtContextSetRayTypeCount( m_context, num_ray_types ) );
  }

  inline unsigned int ContextObj::getRayTypeCount() const
  {
    unsigned int result;
    checkError( rtContextGetRayTypeCount( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setMissProgram(unsigned int ray_type_index, Program  program)
  {
    checkError( rtContextSetMissProgram( m_context, ray_type_index, program ? program->get() : 0 ) );
  }

  inline Program ContextObj::getMissProgram(unsigned int ray_type_index) const
  {
    RTprogram result;
    checkError( rtContextGetMissProgram( m_context, ray_type_index, &result ) );
    return Program::take( result );
  }

  inline void ContextObj::compile()
  {
    checkError( rtContextCompile( m_context ) );
  }

  inline void ContextObj::launch(unsigned int entry_point_index, RTsize image_width)
  {
    checkError( rtContextLaunch1D( m_context, entry_point_index, image_width ) );
  }

  inline void ContextObj::launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height)
  {
    checkError( rtContextLaunch2D( m_context, entry_point_index, image_width, image_height ) );
  }

  inline void ContextObj::launch(unsigned int entry_point_index, RTsize image_width, RTsize image_height, RTsize image_depth)
  {
    checkError( rtContextLaunch3D( m_context, entry_point_index, image_width, image_height, image_depth ) );
  }


  // Progressive API
  inline void ContextObj::launchProgressive(unsigned int entry_point_index, RTsize image_width, RTsize image_height, unsigned int max_subframes)
  {
    checkError( rtContextLaunchProgressive2D( m_context, entry_point_index, image_width, image_height, max_subframes ) );
  }

  inline void ContextObj::stopProgressive()
  {
    checkError( rtContextStopProgressive( m_context ) );
  }

  inline int ContextObj::getRunningState() const
  {
    int result;
    checkError( rtContextGetRunningState( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setPrintEnabled(bool enabled)
  {
    checkError( rtContextSetPrintEnabled( m_context, enabled ) );
  }

  inline bool ContextObj::getPrintEnabled() const
  {
    int enabled;
    checkError( rtContextGetPrintEnabled( m_context, &enabled ) );
    return enabled != 0;
  }

  inline void ContextObj::setPrintBufferSize(RTsize buffer_size_bytes)
  {
    checkError( rtContextSetPrintBufferSize( m_context, buffer_size_bytes ) );
  }

  inline RTsize ContextObj::getPrintBufferSize() const
  {
    RTsize result;
    checkError( rtContextGetPrintBufferSize( m_context, &result ) );
    return result;
  }

  inline void ContextObj::setPrintLaunchIndex(int x, int y, int z)
  {
    checkError( rtContextSetPrintLaunchIndex( m_context, x, y, z ) );
  }

  inline optix::int3 ContextObj::getPrintLaunchIndex() const
  {
    optix::int3 result;
    checkError( rtContextGetPrintLaunchIndex( m_context, &result.x, &result.y, &result.z ) );
    return result;
  }

  inline Variable ContextObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtContextDeclareVariable( m_context, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable ContextObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtContextQueryVariable( m_context, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void ContextObj::removeVariable(Variable v)
  {
    checkError( rtContextRemoveVariable( m_context, v->get() ) );
  }

  inline unsigned int ContextObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtContextGetVariableCount( m_context, &result ) );
    return result;
  }

  inline Variable ContextObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtContextGetVariable( m_context, index, &v ) );
    return Variable::take( v );
  }


  inline RTcontext ContextObj::get()
  {
    return m_context;
  }

  inline void ProgramObj::destroy()
  {
    Context context = getContext();
    checkError( rtProgramDestroy( m_program ), context );
    m_program = 0;
  }

  inline void ProgramObj::validate()
  {
    checkError( rtProgramValidate( m_program ) );
  }

  inline Context ProgramObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtProgramGetContext( m_program, &c ) );
    return Context::take( c );
  }

  inline Variable ProgramObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtProgramDeclareVariable( m_program, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable ProgramObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtProgramQueryVariable( m_program, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void ProgramObj::removeVariable(Variable v)
  {
    checkError( rtProgramRemoveVariable( m_program, v->get() ) );
  }

  inline unsigned int ProgramObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtProgramGetVariableCount( m_program, &result ) );
    return result;
  }

  inline Variable ProgramObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtProgramGetVariable( m_program, index, &v ) );
    return Variable::take(v);
  }

  inline int ProgramObj::getId() const
  {
    int result;
    checkError( rtProgramGetId( m_program, &result ) );
    return result;
  }

  inline void ProgramObj::setCallsitePotentialCallees( const std::string& callSiteName, const std::vector<int>& calleeIds )
  {
      if( !calleeIds.empty() )
          checkError( rtProgramCallsiteSetPotentialCallees( m_program, callSiteName.c_str(), &calleeIds[0],
                                                            static_cast<int>( calleeIds.size() ) ) );
      else
          // reset potential callees of this call site
          checkError( rtProgramCallsiteSetPotentialCallees( m_program, callSiteName.c_str(), 0, 0 ) );
  }

  inline RTprogram ProgramObj::get()
  {
    return m_program;
  }

  inline void GroupObj::destroy()
  {
    Context context = getContext();
    checkError( rtGroupDestroy( m_group ), context );
    m_group = 0;
  }

  inline void GroupObj::validate()
  {
    checkError( rtGroupValidate( m_group ) );
  }

  inline Context GroupObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGroupGetContext( m_group, &c) );
    return Context::take(c);
  }

  inline void SelectorObj::destroy()
  {
    Context context = getContext();
    checkError( rtSelectorDestroy( m_selector ), context );
    m_selector = 0;
  }

  inline void SelectorObj::validate()
  {
    checkError( rtSelectorValidate( m_selector ) );
  }

  inline Context SelectorObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtSelectorGetContext( m_selector, &c ) );
    return Context::take( c );
  }

  inline void SelectorObj::setVisitProgram(Program program)
  {
    checkError( rtSelectorSetVisitProgram( m_selector, program ? program->get() : 0 ) );
  }

  inline Program SelectorObj::getVisitProgram() const
  {
    RTprogram result;
    checkError( rtSelectorGetVisitProgram( m_selector, &result ) );
    return Program::take( result );
  }

  inline void SelectorObj::setChildCount(unsigned int count)
  {
    checkError( rtSelectorSetChildCount( m_selector, count) );
  }

  inline unsigned int SelectorObj::getChildCount() const
  {
    unsigned int result;
    checkError( rtSelectorGetChildCount( m_selector, &result ) );
    return result;
  }

  template< typename T >
  inline void SelectorObj::setChild(unsigned int index, T child)
  {
    checkError( rtSelectorSetChild( m_selector, index, child->get() ) );
  }

  template< typename T >
  inline T SelectorObj::getChild(unsigned int index) const
  {
    RTobject result;
    checkError( rtSelectorGetChild( m_selector, index, &result ) );
    return T::take( result );
  }
  
  inline RTobjecttype SelectorObj::getChildType(unsigned int index) const
  {
     RTobjecttype type;
     checkError( rtSelectorGetChildType( m_selector, index, &type) );
     return type;
  }

  template< typename T >
  inline unsigned int SelectorObj::addChild(T child)
  {
    unsigned int index;
    checkError( rtSelectorGetChildCount( m_selector, &index ) );
    checkError( rtSelectorSetChildCount( m_selector, index+1 ) );
    checkError( rtSelectorSetChild( m_selector, index, child->get() ) );
    return index;
  }

  template< typename T >
  inline unsigned int SelectorObj::removeChild(T child)
  {
    unsigned int index = getChildIndex( child );
    removeChild( index );
    return index;
  }

  inline void SelectorObj::removeChild(int index)
  {
    removeChild(static_cast<unsigned int>(index));
  }

  inline void SelectorObj::removeChild(unsigned int index)
  {
    // Shift down all elements in O(n)
    unsigned int count;
    RTobject temp;
    checkError( rtSelectorGetChildCount( m_selector, &count ) );
    if(index >= count) {
      RTcontext c = this->getContext()->get();
      throw Exception::makeException( RT_ERROR_INVALID_VALUE, c );
    }
    for(unsigned int i=index+1; i<count; i++) {
      checkError( rtSelectorGetChild( m_selector, i, &temp ) );
      checkError( rtSelectorSetChild( m_selector, i-1, temp ) );
    }
    checkError( rtSelectorSetChildCount( m_selector, count-1 ) );
  }

  template< typename T >
  inline unsigned int SelectorObj::getChildIndex(T child) const
  {
    unsigned int count;
    RTobject temp;
    checkError( rtSelectorGetChildCount( m_selector, &count ) );
    for( unsigned int index = 0; index < count; index++ ) {
      checkError( rtSelectorGetChild( m_selector, index, &temp ) );
      if( child->get() == temp ) return index; // Found
    }
    RTcontext c = this->getContext()->get();
    throw Exception::makeException( RT_ERROR_INVALID_VALUE, c );
  }

  inline Variable SelectorObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtSelectorDeclareVariable( m_selector, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable SelectorObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtSelectorQueryVariable( m_selector, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void SelectorObj::removeVariable(Variable v)
  {
    checkError( rtSelectorRemoveVariable( m_selector, v->get() ) );
  }

  inline unsigned int SelectorObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtSelectorGetVariableCount( m_selector, &result ) );
    return result;
  }

  inline Variable SelectorObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtSelectorGetVariable( m_selector, index, &v ) );
    return Variable::take( v );
  }

  inline RTselector SelectorObj::get()
  {
    return m_selector;
  }

  inline void GroupObj::setAcceleration(Acceleration acceleration)
  {
    checkError( rtGroupSetAcceleration( m_group, acceleration->get() ) );
  }

  inline Acceleration GroupObj::getAcceleration() const
  {
    RTacceleration result;
    checkError( rtGroupGetAcceleration( m_group, &result ) );
    return Acceleration::take( result );
  }

  inline void GroupObj::setChildCount(unsigned int  count)
  {
    checkError( rtGroupSetChildCount( m_group, count ) );
  }

  inline unsigned int GroupObj::getChildCount() const
  {
    unsigned int result;
    checkError( rtGroupGetChildCount( m_group, &result ) );
    return result;
  }

  template< typename T >
  inline void GroupObj::setChild(unsigned int index, T child)
  {
    checkError( rtGroupSetChild( m_group, index, child->get() ) );
  }

  template< typename T >
  inline T GroupObj::getChild(unsigned int index) const
  {
    RTobject result;
    checkError( rtGroupGetChild( m_group, index, &result) );
    return T::take( result );
  }

  inline RTobjecttype GroupObj::getChildType(unsigned int index) const
  {
     RTobjecttype type;
     checkError( rtGroupGetChildType( m_group, index, &type) );
     return type;
  }

  template< typename T >
  inline unsigned int GroupObj::addChild(T child)
  {
    unsigned int index;
    checkError( rtGroupGetChildCount( m_group, &index ) );
    checkError( rtGroupSetChildCount( m_group, index+1 ) );
    checkError( rtGroupSetChild( m_group, index, child->get() ) );
    return index;
  }

  template< typename T >
  inline unsigned int GroupObj::removeChild(T child)
  {
    unsigned int index = getChildIndex( child );
    removeChild( index );
    return index;
  }

  inline void GroupObj::removeChild(int index)
  {
    removeChild(static_cast<unsigned int>(index));
  }
  
  inline void GroupObj::removeChild(unsigned int index)
  {
    unsigned int count;
    checkError( rtGroupGetChildCount( m_group, &count ) );
    if(index >= count) {
      RTcontext c = this->getContext()->get();
      throw Exception::makeException( RT_ERROR_INVALID_VALUE, c );
    }

    // Replace to-be-removed child with last child. 
    RTobject temp;
    checkError( rtGroupGetChild( m_group, count-1, &temp ) );
    checkError( rtGroupSetChild( m_group, index, temp ) );
    checkError( rtGroupSetChildCount( m_group, count-1 ) );
  }

  inline void GroupObj::setVisibilityMask( RTvisibilitymask mask )
  {
    checkError( rtGroupSetVisibilityMask(m_group, mask ) );
  }

  inline RTvisibilitymask GroupObj::getVisibilityMask() const
  {
    RTvisibilitymask mask = ~RTvisibilitymask(0);
    checkError( rtGroupGetVisibilityMask(m_group, &mask ) );
    return mask;
  }

  template< typename T >
  inline unsigned int GroupObj::getChildIndex(T child) const
  {
    unsigned int count;
    RTobject temp;
    checkError( rtGroupGetChildCount( m_group, &count ) );
    for( unsigned int index = 0; index < count; index++ ) {
      checkError( rtGroupGetChild( m_group, index, &temp ) );
      if( child->get() == temp ) return index; // Found
    }
    RTcontext c = this->getContext()->get();
    throw Exception::makeException( RT_ERROR_INVALID_VALUE, c );
  }

  inline RTgroup GroupObj::get()
  {
    return m_group;
  }

  inline void GeometryGroupObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryGroupDestroy( m_geometrygroup ), context );
    m_geometrygroup = 0;
  }

  inline void GeometryGroupObj::validate()
  {
    checkError( rtGeometryGroupValidate( m_geometrygroup ) );
  }

  inline Context GeometryGroupObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryGroupGetContext( m_geometrygroup, &c) );
    return Context::take(c);
  }

  inline void GeometryGroupObj::setAcceleration(Acceleration acceleration)
  {
    checkError( rtGeometryGroupSetAcceleration( m_geometrygroup, acceleration->get() ) );
  }

  inline Acceleration GeometryGroupObj::getAcceleration() const
  {
    RTacceleration result;
    checkError( rtGeometryGroupGetAcceleration( m_geometrygroup, &result ) );
    return Acceleration::take( result );
  }

  inline void GeometryGroupObj::setChildCount(unsigned int  count)
  {
    checkError( rtGeometryGroupSetChildCount( m_geometrygroup, count ) );
  }

  inline unsigned int GeometryGroupObj::getChildCount() const
  {
    unsigned int result;
    checkError( rtGeometryGroupGetChildCount( m_geometrygroup, &result ) );
    return result;
  }

  inline void GeometryGroupObj::setChild(unsigned int index, GeometryInstance child)
  {
    checkError( rtGeometryGroupSetChild( m_geometrygroup, index, child->get() ) );
  }

  inline GeometryInstance GeometryGroupObj::getChild(unsigned int index) const
  {
    RTgeometryinstance result;
    checkError( rtGeometryGroupGetChild( m_geometrygroup, index, &result) );
    return GeometryInstance::take( result );
  }

  inline unsigned int GeometryGroupObj::addChild(GeometryInstance child)
  {
    unsigned int index;
    checkError( rtGeometryGroupGetChildCount( m_geometrygroup, &index ) );
    checkError( rtGeometryGroupSetChildCount( m_geometrygroup, index+1 ) );
    checkError( rtGeometryGroupSetChild( m_geometrygroup, index, child->get() ) );
    return index;
  }

  inline unsigned int GeometryGroupObj::removeChild(GeometryInstance child)
  {
    unsigned int index = getChildIndex( child );
    removeChild( index );
    return index;
  }

  inline void GeometryGroupObj::removeChild(int index)
  {
    removeChild(static_cast<unsigned int>(index));
  }
  
  inline void GeometryGroupObj::removeChild(unsigned int index)
  {
    unsigned int count;
    checkError( rtGeometryGroupGetChildCount( m_geometrygroup, &count ) );
    if(index >= count) {
      RTcontext c = this->getContext()->get();
      throw Exception::makeException( RT_ERROR_INVALID_VALUE, c );
    }
    
    // Replace to-be-removed child with last child. 
    RTgeometryinstance temp;
    checkError( rtGeometryGroupGetChild( m_geometrygroup, count-1, &temp ) );
    checkError( rtGeometryGroupSetChild( m_geometrygroup, index, temp ) );
    checkError( rtGeometryGroupSetChildCount( m_geometrygroup, count-1 ) );
  }

  inline unsigned int GeometryGroupObj::getChildIndex(GeometryInstance child) const
  {
    unsigned int count;
    RTgeometryinstance temp;
    checkError( rtGeometryGroupGetChildCount( m_geometrygroup, &count ) );
    for( unsigned int index = 0; index < count; index++ ) {
      checkError( rtGeometryGroupGetChild( m_geometrygroup, index, &temp ) );
      if( child->get() == temp ) return index; // Found
    }
    RTcontext c = this->getContext()->get();
    throw Exception::makeException( RT_ERROR_INVALID_VALUE, c );
  }

  inline void GeometryGroupObj::setFlags( RTinstanceflags flags )
  {
    checkError( rtGeometryGroupSetFlags( m_geometrygroup, flags ) );
  }

  inline RTinstanceflags GeometryGroupObj::getFlags() const
  {
    RTinstanceflags flags = RT_INSTANCE_FLAG_NONE;
    checkError( rtGeometryGroupGetFlags( m_geometrygroup, &flags ) );
    return flags;
  }

  inline void GeometryGroupObj::setVisibilityMask( RTvisibilitymask mask )
  {
    checkError( rtGeometryGroupSetVisibilityMask( m_geometrygroup, mask ) );
  }

  inline RTvisibilitymask GeometryGroupObj::getVisibilityMask() const
  {
    RTvisibilitymask mask = ~RTvisibilitymask( 0 );
    checkError( rtGeometryGroupGetVisibilityMask( m_geometrygroup, &mask ) );
    return mask;
  }

  inline RTgeometrygroup GeometryGroupObj::get()
  {
    return m_geometrygroup;
  }

  inline void TransformObj::destroy()
  {
    Context context = getContext();
    checkError( rtTransformDestroy( m_transform ), context );
    m_transform = 0;
  }

  inline void TransformObj::validate()
  {
    checkError( rtTransformValidate( m_transform ) );
  }

  inline Context TransformObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtTransformGetContext( m_transform, &c) );
    return Context::take(c);
  }

  template< typename T >
  inline void TransformObj::setChild(T child)
  {
    checkError( rtTransformSetChild( m_transform, child->get() ) );
  }

  template< typename T >
  inline T TransformObj::getChild() const
  {
    RTobject result;
    checkError( rtTransformGetChild( m_transform, &result) );
    return T::take( result );
  }
  
  inline RTobjecttype TransformObj::getChildType() const
  {

     RTobjecttype type;
     checkError( rtTransformGetChildType( m_transform, &type) );
     return type;
  }

  inline void TransformObj::setMatrix(bool transpose, const float* matrix, const float* inverse_matrix)
  {
    checkError( rtTransformSetMatrix( m_transform, transpose, matrix, inverse_matrix ) );
  }

  inline void TransformObj::getMatrix(bool transpose, float* matrix, float* inverse_matrix) const
  {
    checkError( rtTransformGetMatrix( m_transform, transpose, matrix, inverse_matrix ) );
  }
    
  inline void TransformObj::setMotionRange( float timeBegin, float timeEnd )
  {
    checkError( rtTransformSetMotionRange( m_transform, timeBegin, timeEnd ) );
  }

  inline void TransformObj::getMotionRange( float& timeBegin, float& timeEnd )
  {
    checkError( rtTransformGetMotionRange( m_transform, &timeBegin, &timeEnd ) );
  }

  inline void TransformObj::setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode )
  {
    checkError( rtTransformSetMotionBorderMode( m_transform, beginMode, endMode ) );
  }

  inline void TransformObj::getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode )
  {
    checkError( rtTransformGetMotionBorderMode( m_transform, &beginMode, &endMode ) );
  }

  inline void TransformObj::setMotionKeys( unsigned int n, RTmotionkeytype type, const float* keys )
  {
    checkError( rtTransformSetMotionKeys( m_transform, n, type, keys ) );
  }

  inline unsigned int TransformObj::getMotionKeyCount()
  {
    unsigned int n;
    checkError( rtTransformGetMotionKeyCount( m_transform, &n ) );
    return n;
  }

  inline RTmotionkeytype TransformObj::getMotionKeyType()
  {
    RTmotionkeytype type;
    checkError( rtTransformGetMotionKeyType( m_transform, &type ) );
    return type;
  }

  inline void TransformObj::getMotionKeys( float* keys )
  {
    checkError( rtTransformGetMotionKeys( m_transform, keys ) );
  }

  inline RTtransform TransformObj::get()
  {
    return m_transform;
  }

  inline void AccelerationObj::destroy()
  {
    Context context = getContext();
    checkError( rtAccelerationDestroy(m_acceleration), context );
    m_acceleration = 0;
  }

  inline void AccelerationObj::validate()
  {
    checkError( rtAccelerationValidate(m_acceleration) );
  }

  inline Context AccelerationObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtAccelerationGetContext(m_acceleration, &c ) );
    return Context::take( c );
  }

  inline void AccelerationObj::markDirty()
  {
    checkError( rtAccelerationMarkDirty(m_acceleration) );
  }

  inline bool AccelerationObj::isDirty() const
  {
    int dirty;
    checkError( rtAccelerationIsDirty(m_acceleration,&dirty) );
    return dirty != 0;
  }

  inline void AccelerationObj::setProperty( const std::string& name, const std::string& value )
  {
    checkError( rtAccelerationSetProperty(m_acceleration, name.c_str(), value.c_str() ) );
  }

  inline std::string AccelerationObj::getProperty( const std::string& name ) const
  {
    const char* s;
    checkError( rtAccelerationGetProperty(m_acceleration, name.c_str(), &s ) );
    return std::string( s );
  }

  inline void AccelerationObj::setBuilder(const std::string& builder)
  {
    checkError( rtAccelerationSetBuilder(m_acceleration, builder.c_str() ) );
  }

  inline std::string AccelerationObj::getBuilder() const
  {
    const char* s;
    checkError( rtAccelerationGetBuilder(m_acceleration, &s ) );
    return std::string( s );
  }

  inline void AccelerationObj::setTraverser(const std::string& traverser)
  {
    checkError( rtAccelerationSetTraverser(m_acceleration, traverser.c_str() ) );
  }

  inline std::string AccelerationObj::getTraverser() const
  {
    const char* s;
    checkError( rtAccelerationGetTraverser(m_acceleration, &s ) );
    return std::string( s );
  }

  inline RTsize AccelerationObj::getDataSize() const
  {
    RTsize sz;
    checkError( rtAccelerationGetDataSize(m_acceleration, &sz) );
    return sz;
  }

  inline void AccelerationObj::getData( void* data ) const
  {
    checkError( rtAccelerationGetData(m_acceleration,data) );
  }

  inline void AccelerationObj::setData( const void* data, RTsize size )
  {
    checkError( rtAccelerationSetData(m_acceleration,data,size) );
  }

  inline RTacceleration AccelerationObj::get()
  {
    return m_acceleration;
  }

  inline void GeometryInstanceObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryInstanceDestroy( m_geometryinstance ), context );
    m_geometryinstance = 0;
  }

  inline void GeometryInstanceObj::validate()
  {
    checkError( rtGeometryInstanceValidate( m_geometryinstance ) );
  }

  inline Context GeometryInstanceObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryInstanceGetContext( m_geometryinstance, &c ) );
    return Context::take( c );
  }

  inline void GeometryInstanceObj::setGeometry(Geometry geometry)
  {
    checkError( rtGeometryInstanceSetGeometry( m_geometryinstance, geometry->get() ) );
  }

  inline void GeometryInstanceObj::setGeometryTriangles(GeometryTriangles geom_tris)
  {
    checkError( rtGeometryInstanceSetGeometryTriangles( m_geometryinstance, geom_tris->get() ) );
  }

  inline Geometry GeometryInstanceObj::getGeometry() const
  {
    RTgeometry result;
    checkError( rtGeometryInstanceGetGeometry( m_geometryinstance, &result ) );
    return Geometry::take( result );
  }

  inline GeometryTriangles GeometryInstanceObj::getGeometryTriangles() const
  {
    RTgeometrytriangles result;
    checkError( rtGeometryInstanceGetGeometryTriangles( m_geometryinstance, &result ) );
    return GeometryTriangles::take( result );
  }

  inline void GeometryInstanceObj::setMaterialCount(unsigned int  count)
  {
    checkError( rtGeometryInstanceSetMaterialCount( m_geometryinstance, count ) );
  }

  inline unsigned int GeometryInstanceObj::getMaterialCount() const
  {
    unsigned int result;
    checkError( rtGeometryInstanceGetMaterialCount( m_geometryinstance, &result ) );
    return result;
  }

  inline void GeometryInstanceObj::setMaterial(unsigned int idx, Material  material)
  {
    checkError( rtGeometryInstanceSetMaterial( m_geometryinstance, idx, material->get()) );
  }

  inline Material GeometryInstanceObj::getMaterial(unsigned int idx) const
  {
    RTmaterial result;
    checkError( rtGeometryInstanceGetMaterial( m_geometryinstance, idx, &result ) );
    return Material::take( result );
  }

  // Adds the material and returns the index to the added material.
  inline unsigned int GeometryInstanceObj::addMaterial(Material material)
  {
    unsigned int old_count = getMaterialCount();
    setMaterialCount(old_count+1);
    setMaterial(old_count, material);
    return old_count;
  }

  inline Variable GeometryInstanceObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtGeometryInstanceDeclareVariable( m_geometryinstance, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable GeometryInstanceObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtGeometryInstanceQueryVariable( m_geometryinstance, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void GeometryInstanceObj::removeVariable(Variable v)
  {
    checkError( rtGeometryInstanceRemoveVariable( m_geometryinstance, v->get() ) );
  }

  inline unsigned int GeometryInstanceObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtGeometryInstanceGetVariableCount( m_geometryinstance, &result ) );
    return result;
  }

  inline Variable GeometryInstanceObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtGeometryInstanceGetVariable( m_geometryinstance, index, &v ) );
    return Variable::take( v );
  }

  inline RTgeometryinstance GeometryInstanceObj::get()
  {
    return m_geometryinstance;
  }

  inline void GeometryObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryDestroy( m_geometry ), context );
    m_geometry = 0;
  }

  inline void GeometryObj::validate()
  {
    checkError( rtGeometryValidate( m_geometry ) );
  }

  inline Context GeometryObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryGetContext( m_geometry, &c ) );
    return Context::take( c );
  }

  inline void GeometryObj::setPrimitiveCount(unsigned int num_primitives)
  {
    checkError( rtGeometrySetPrimitiveCount( m_geometry, num_primitives ) );
  }

  inline unsigned int GeometryObj::getPrimitiveCount() const
  {
    unsigned int result;
    checkError( rtGeometryGetPrimitiveCount( m_geometry, &result ) );
    return result;
  }

  inline void GeometryObj::setPrimitiveIndexOffset(unsigned int index_offset)
  {
    checkError( rtGeometrySetPrimitiveIndexOffset( m_geometry, index_offset) );
  }

  inline unsigned int GeometryObj::getPrimitiveIndexOffset() const
  {
    unsigned int result;
    checkError( rtGeometryGetPrimitiveIndexOffset( m_geometry, &result ) );
    return result;
  }

  inline void GeometryObj::setMotionRange( float timeBegin, float timeEnd )
  {
    checkError( rtGeometrySetMotionRange( m_geometry, timeBegin, timeEnd ) );
  }

  inline void GeometryObj::getMotionRange( float& timeBegin, float& timeEnd )
  {

    checkError( rtGeometryGetMotionRange( m_geometry, &timeBegin, &timeEnd ) );
  }

  inline void GeometryObj::setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode )
  {
    checkError( rtGeometrySetMotionBorderMode( m_geometry, beginMode, endMode ) );
  }

  inline void GeometryObj::getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode )
  {
    checkError( rtGeometryGetMotionBorderMode( m_geometry, &beginMode, &endMode ) );
  }

  inline void GeometryObj::setMotionSteps( unsigned int n )
  {
    checkError( rtGeometrySetMotionSteps( m_geometry, n ) );
  }

  inline unsigned int GeometryObj::getMotionSteps()
  {
    unsigned int n;
    checkError( rtGeometryGetMotionSteps( m_geometry, &n ) );
    return n;
  }

  inline void GeometryObj::setBoundingBoxProgram(Program  program)
  {
    checkError( rtGeometrySetBoundingBoxProgram( m_geometry, program ? program->get() : 0 ) );
  }

  inline Program GeometryObj::getBoundingBoxProgram() const
  {
    RTprogram result;
    checkError( rtGeometryGetBoundingBoxProgram( m_geometry, &result ) );
    return Program::take( result );
  }

  inline void GeometryObj::setIntersectionProgram(Program  program)
  {
    checkError( rtGeometrySetIntersectionProgram( m_geometry, program ? program->get() : 0 ) );
  }

  inline Program GeometryObj::getIntersectionProgram() const
  {
    RTprogram result;
    checkError( rtGeometryGetIntersectionProgram( m_geometry, &result ) );
    return Program::take( result );
  }

  inline Variable GeometryObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtGeometryDeclareVariable( m_geometry, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable GeometryObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtGeometryQueryVariable( m_geometry, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void GeometryObj::removeVariable(Variable v)
  {
    checkError( rtGeometryRemoveVariable( m_geometry, v->get() ) );
  }

  inline unsigned int GeometryObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtGeometryGetVariableCount( m_geometry, &result ) );
    return result;
  }

  inline Variable GeometryObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtGeometryGetVariable( m_geometry, index, &v ) );
    return Variable::take( v );
  }

  inline void GeometryObj::setFlags( RTgeometryflags flags )
  {
    checkError( rtGeometrySetFlags( m_geometry, flags ) );
  }

  inline RTgeometryflags GeometryObj::setFlags() const
  {
    RTgeometryflags flags = RT_GEOMETRY_FLAG_NONE;
    checkError( rtGeometryGetFlags( m_geometry, &flags ) );
    return flags;
  }

  inline void GeometryObj::markDirty()
  {
    checkError( rtGeometryMarkDirty(m_geometry) );
  }

  inline bool GeometryObj::isDirty() const
  {
    int dirty;
    checkError( rtGeometryIsDirty(m_geometry,&dirty) );
    return dirty != 0;
  }

  inline RTgeometry GeometryObj::get()
  {
    return m_geometry;
  }

  inline void GeometryTrianglesObj::destroy()
  {
    Context context = getContext();
    checkError( rtGeometryTrianglesDestroy( m_geometryTriangles ), context );
    m_geometryTriangles = 0;
  }

  inline void GeometryTrianglesObj::validate()
  {
    checkError( rtGeometryTrianglesValidate( m_geometryTriangles ) );
  }

  inline Context GeometryTrianglesObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtGeometryTrianglesGetContext( m_geometryTriangles, &c ) );
    return Context::take( c );
  }

  inline void GeometryTrianglesObj::setPrimitiveIndexOffset( unsigned int index_offset )
  {
      checkError( rtGeometryTrianglesSetPrimitiveIndexOffset( m_geometryTriangles, index_offset ) );
  }

  inline unsigned int GeometryTrianglesObj::getPrimitiveIndexOffset() const
  {
      unsigned int index_offset = 0;
      checkError( rtGeometryTrianglesGetPrimitiveIndexOffset( m_geometryTriangles, &index_offset ) );
      return index_offset;
  }

  inline void GeometryTrianglesObj::setPrimitiveCount( unsigned int num_triangles )
  {
      checkError( rtGeometryTrianglesSetPrimitiveCount( m_geometryTriangles, num_triangles ) );
  }

  inline unsigned int GeometryTrianglesObj::getPrimitiveCount() const
  {
      unsigned int result = 0;
      checkError( rtGeometryTrianglesGetPrimitiveCount( m_geometryTriangles, &result ) );
      return result;
  }

  inline void GeometryTrianglesObj::setPreTransformMatrix( bool transpose, const float* matrix )
  {
      checkError( rtGeometryTrianglesSetPreTransformMatrix( m_geometryTriangles, transpose ? 1 : 0, matrix ) );
  }

  inline void GeometryTrianglesObj::getPreTransformMatrix( bool transpose, float* matrix )
  {
      checkError( rtGeometryTrianglesGetPreTransformMatrix( m_geometryTriangles, transpose ? 1 : 0, matrix ) );
  }

  inline void GeometryTrianglesObj::setTriangleIndices( Buffer index_buffer, RTformat tri_indices_format )
  {
      setTriangleIndices( index_buffer, 0, tri_indices_format );
  }

  inline void GeometryTrianglesObj::setTriangleIndices( Buffer index_buffer, RTsize index_buffer_byte_offset, RTformat tri_indices_format )
  {
      setTriangleIndices( index_buffer, index_buffer_byte_offset, index_buffer->getElementSize(), tri_indices_format );
  }

  inline void GeometryTrianglesObj::setTriangleIndices( Buffer   index_buffer,
                                                         RTsize   index_buffer_byte_offset,
                                                         RTsize   tri_indices_byte_stride,
                                                         RTformat tri_indices_format )
  {
      checkError( rtGeometryTrianglesSetTriangleIndices( m_geometryTriangles, index_buffer->get(), index_buffer_byte_offset,
                                                          tri_indices_byte_stride, tri_indices_format ) );
  }

  inline void GeometryTrianglesObj::setVertices( unsigned int num_vertices, Buffer vertex_buffer, RTformat position_format )
  {
      setVertices( num_vertices, vertex_buffer, 0, position_format );
  }

  inline void GeometryTrianglesObj::setVertices( unsigned int num_vertices, Buffer vertex_buffer, RTsize vertex_buffer_byte_offset, RTformat position_format )
  {
      setVertices( num_vertices, vertex_buffer, vertex_buffer_byte_offset, vertex_buffer->getElementSize(), position_format );
  }

  inline void GeometryTrianglesObj::setVertices( unsigned int num_vertices,
                                                 Buffer       vertex_buffer,
                                                 RTsize       vertex_buffer_byte_offset,
                                                 RTsize       vertex_byte_stride,
                                                 RTformat     position_format )
  {
      checkError( rtGeometryTrianglesSetVertices( m_geometryTriangles, num_vertices, vertex_buffer->get(),
                                                  vertex_buffer_byte_offset, vertex_byte_stride, position_format ) );
  }

  inline void GeometryTrianglesObj::setMotionVertices( unsigned int num_vertices,
                                                       Buffer       vertex_buffer,
                                                       RTsize       vertex_buffer_byte_offset,
                                                       RTsize       vertex_byte_stride,
                                                       RTsize       vertex_motion_step_byte_stride,
                                                       RTformat     position_format )
  {
      checkError( rtGeometryTrianglesSetMotionVertices( m_geometryTriangles, num_vertices, vertex_buffer->get(), vertex_buffer_byte_offset,
                                                        vertex_byte_stride, vertex_motion_step_byte_stride, position_format ) );
  }

  template <class BufferIterator>
  inline void GeometryTrianglesObj::setMotionVerticesMultiBuffer( unsigned int   num_vertices,
                                                                  BufferIterator vertex_buffers_begin,
                                                                  BufferIterator vertex_buffers_end,
                                                                  RTsize         vertex_buffer_byte_offset,
                                                                  RTsize         vertex_byte_stride,
                                                                  RTformat       position_format )
  {
      // cannot use motion step count, because we don't know the order of the user calls
      const typename std::iterator_traits<BufferIterator>::difference_type count =
          std::distance( vertex_buffers_begin, vertex_buffers_end );

      std::vector<RTbuffer> buffers;
      buffers.reserve( count );
      for( BufferIterator iter = vertex_buffers_begin; iter != vertex_buffers_end; ++iter )
      {
          // untangle Buffer* (Handle<BufferObj>*) to underlying RTbuffer
          buffers.push_back( iter->get()->get() );
      }

      checkError( rtGeometryTrianglesSetMotionVerticesMultiBuffer( m_geometryTriangles, num_vertices,
                                                                   count > 0 ? &buffers[0] : 0, (unsigned int)count, vertex_buffer_byte_offset,
                                                                   vertex_byte_stride, position_format ) );
  }

  inline void GeometryTrianglesObj::setMotionSteps( unsigned int num_motion_steps )
  {
      checkError( rtGeometryTrianglesSetMotionSteps( m_geometryTriangles, num_motion_steps ) );
  }

  inline unsigned int GeometryTrianglesObj::getMotionSteps() const
  {
      unsigned int n;
      checkError( rtGeometryTrianglesGetMotionSteps( m_geometryTriangles, &n ) );
      return n;
  }

  inline void GeometryTrianglesObj::setMotionRange( float timeBegin, float timeEnd )
  {
      checkError( rtGeometryTrianglesSetMotionRange( m_geometryTriangles, timeBegin, timeEnd ) );
  }

  inline void GeometryTrianglesObj::getMotionRange( float& timeBegin, float& timeEnd ) const
  {

      checkError( rtGeometryTrianglesGetMotionRange( m_geometryTriangles, &timeBegin, &timeEnd ) );
  }

  inline void GeometryTrianglesObj::setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode )
  {
      checkError( rtGeometryTrianglesSetMotionBorderMode( m_geometryTriangles, beginMode, endMode ) );
  }

  inline void GeometryTrianglesObj::getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode ) const
  {
      checkError( rtGeometryTrianglesGetMotionBorderMode( m_geometryTriangles, &beginMode, &endMode ) );
  }

  inline void GeometryTrianglesObj::setBuildFlags( RTgeometrybuildflags build_flags )
  {
      checkError( rtGeometryTrianglesSetBuildFlags( m_geometryTriangles, build_flags ) );
  }

  inline void GeometryTrianglesObj::setMaterialCount( unsigned int num_materials )
  {
      checkError( rtGeometryTrianglesSetMaterialCount( m_geometryTriangles, num_materials ) );
  }

  inline unsigned int GeometryTrianglesObj::getMaterialCount() const
  {
      unsigned int n;
      checkError( rtGeometryTrianglesGetMaterialCount( m_geometryTriangles, &n ) );
      return n;
  }

  inline void GeometryTrianglesObj::setMaterialIndices( Buffer   material_index_buffer,
                                                        RTsize   material_index_buffer_byte_offset,
                                                        RTsize   material_index_byte_stride,
                                                        RTformat material_index_format )
  {
      checkError( rtGeometryTrianglesSetMaterialIndices( m_geometryTriangles, material_index_buffer->get(), material_index_buffer_byte_offset,
                                                         material_index_byte_stride, material_index_format ) );
  }

  inline void GeometryTrianglesObj::setFlagsPerMaterial( unsigned int material_index, RTgeometryflags flags )
  {
      checkError( rtGeometryTrianglesSetFlagsPerMaterial( m_geometryTriangles, material_index, flags ) );
  }

  inline RTgeometryflags GeometryTrianglesObj::getFlagsPerMaterial( unsigned int material_index ) const
  {
      RTgeometryflags flags = RT_GEOMETRY_FLAG_NONE;
      checkError( rtGeometryTrianglesGetFlagsPerMaterial( m_geometryTriangles, material_index, &flags ) );
      return flags;
  }

  inline void GeometryTrianglesObj::setAttributeProgram( Program program )
  {
      checkError( rtGeometryTrianglesSetAttributeProgram( m_geometryTriangles, program ? program->get() : 0 ) );
  }

  inline Program GeometryTrianglesObj::getAttributeProgram() const
  {
      RTprogram result;
      checkError( rtGeometryTrianglesGetAttributeProgram( m_geometryTriangles, &result ) );
      return Program::take( result );
  }

  inline Variable GeometryTrianglesObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtGeometryTrianglesDeclareVariable( m_geometryTriangles, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable GeometryTrianglesObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtGeometryTrianglesQueryVariable( m_geometryTriangles, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline void GeometryTrianglesObj::removeVariable(Variable v)
  {
    checkError( rtGeometryTrianglesRemoveVariable( m_geometryTriangles, v->get() ) );
  }

  inline unsigned int GeometryTrianglesObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtGeometryTrianglesGetVariableCount( m_geometryTriangles, &result ) );
    return result;
  }

  inline Variable GeometryTrianglesObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtGeometryTrianglesGetVariable( m_geometryTriangles, index, &v ) );
    return Variable::take( v );
  }

  inline RTgeometrytriangles GeometryTrianglesObj::get()
  {
      return m_geometryTriangles;
  }

  inline void MaterialObj::destroy()
  {
    Context context = getContext();
    checkError( rtMaterialDestroy( m_material ), context );
    m_material = 0;
  }

  inline void MaterialObj::validate()
  {
    checkError( rtMaterialValidate( m_material ) );
  }

  inline Context MaterialObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtMaterialGetContext( m_material, &c ) );
    return Context::take( c );
  }

  inline void MaterialObj::setClosestHitProgram(unsigned int ray_type_index, Program  program)
  {
    checkError( rtMaterialSetClosestHitProgram( m_material, ray_type_index, program ? program->get() : 0 ) );
  }

  inline Program MaterialObj::getClosestHitProgram(unsigned int ray_type_index) const
  {
    RTprogram result;
    checkError( rtMaterialGetClosestHitProgram( m_material, ray_type_index, &result ) );
    return Program::take( result );
  }

  inline void MaterialObj::setAnyHitProgram(unsigned int ray_type_index, Program  program)
  {
    checkError( rtMaterialSetAnyHitProgram( m_material, ray_type_index, program ? program->get() : 0 ) );
  }

  inline Program MaterialObj::getAnyHitProgram(unsigned int ray_type_index) const
  {
    RTprogram result;
    checkError( rtMaterialGetAnyHitProgram( m_material, ray_type_index, &result ) );
    return Program::take( result );
  }

  inline Variable MaterialObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtMaterialDeclareVariable( m_material, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable MaterialObj::queryVariable(const std::string& name) const
  {
    RTvariable v;
    checkError( rtMaterialQueryVariable( m_material, name.c_str(), &v) );
    return Variable::take( v );
  }

  inline void MaterialObj::removeVariable(Variable v)
  {
    checkError( rtMaterialRemoveVariable( m_material, v->get() ) );
  }

  inline unsigned int MaterialObj::getVariableCount() const
  {
    unsigned int result;
    checkError( rtMaterialGetVariableCount( m_material, &result ) );
    return result;
  }

  inline Variable MaterialObj::getVariable(unsigned int index) const
  {
    RTvariable v;
    checkError( rtMaterialGetVariable( m_material, index, &v) );
    return Variable::take( v );
  }

  inline RTmaterial MaterialObj::get()
  {
    return m_material;
  }

  inline void PostprocessingStageObj::destroy()
  {
      Context context = getContext();
      checkError(rtPostProcessingStageDestroy(m_stage), context);
  }

  inline void PostprocessingStageObj::validate()
  {
  }

  inline Context PostprocessingStageObj::getContext() const
  {
      RTcontext c;
      checkErrorNoGetContext(rtPostProcessingStageGetContext(m_stage, &c));
      return Context::take(c);
  }

  inline unsigned int PostprocessingStageObj::getVariableCount() const
  {
      unsigned int result;
      checkError(rtPostProcessingStageGetVariableCount(m_stage, &result));
      return result;
  }

  inline Variable PostprocessingStageObj::getVariable(unsigned int index) const
  {
      RTvariable v;
      checkError(rtPostProcessingStageGetVariable(m_stage, index, &v));
      return Variable::take(v);
  }

  inline Variable PostprocessingStageObj::declareVariable(const std::string& name)
  {
    RTvariable v;
    checkError( rtPostProcessingStageDeclareVariable( m_stage, name.c_str(), &v ) );
    return Variable::take( v );
  }

  inline Variable PostprocessingStageObj::queryVariable(const std::string& name) const
  {
      RTvariable v;
      checkError(rtPostProcessingStageQueryVariable(m_stage, name.c_str(), &v));
      return Variable::take(v);
  }

  inline RTpostprocessingstage PostprocessingStageObj::get()
  {
      return m_stage;
  }

  inline void CommandListObj::destroy()
  {
    Context context = getContext();
    checkError(rtCommandListDestroy(m_list), context);
  }

  inline void CommandListObj::validate()
  {
  }

  inline Context CommandListObj::getContext() const
  {
    RTcontext c = 0;
    checkErrorNoGetContext(rtCommandListGetContext(m_list, &c));
    return Context::take(c);
  }

  inline void CommandListObj::appendPostprocessingStage(PostprocessingStage stage, RTsize launch_width, RTsize launch_height)
  {
    Context context = getContext();
    checkError(rtCommandListAppendPostprocessingStage(m_list, stage->get(), launch_width, launch_height), context);
  }

  inline void CommandListObj::appendLaunch(unsigned int entryIndex, RTsize launch_width)
  {
    Context context = getContext();
    checkError(rtCommandListAppendLaunch1D(m_list, entryIndex, launch_width), context);
  }

  inline void CommandListObj::appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height )
  {
      Context context = getContext();
      checkError( rtCommandListAppendLaunch2D( m_list, entryIndex, launch_width, launch_height ), context );
  }

  inline void CommandListObj::appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height, RTsize launch_depth )
  {
      Context context = getContext();
      checkError( rtCommandListAppendLaunch3D( m_list, entryIndex, launch_width, launch_height, launch_depth ), context );
  }

  template<class Iterator> inline
      void CommandListObj::setDevices( Iterator begin, Iterator end )
  {
      std::vector<int> devices( begin, end );
      checkError( rtCommandListSetDevices( m_list, static_cast<unsigned int>(devices.size()), &devices[0] ) );
  }

  inline std::vector<int> CommandListObj::getDevices() const
  {
      // Initialize with the number of enabled devices
      unsigned int count = 0;
      rtCommandListGetDeviceCount( m_list, &count );
      std::vector<int> devices( count );
      if( count > 0)
          checkError( rtCommandListGetDevices( m_list, &devices[0] ) );
      return devices;
  }

  inline void CommandListObj::finalize()
  {
    Context context = getContext();
    checkError(rtCommandListFinalize(m_list), context);
  }

  inline void CommandListObj::execute()
  {
    Context context = getContext();
    checkError(rtCommandListExecute(m_list), context);
  }

  inline void CommandListObj::setCudaStream( void* stream )
  {
    Context context = getContext();
    checkError( rtCommandListSetCudaStream( m_list, stream ) );
  }

  inline void CommandListObj::getCudaStream( void** stream )
  {
      Context context = getContext();
      checkError( rtCommandListGetCudaStream( m_list, stream ) );
  }

  inline RTcommandlist CommandListObj::get()
  {
    return m_list;
  }

  inline void TextureSamplerObj::destroy()
  {
    Context context = getContext();
    checkError( rtTextureSamplerDestroy( m_texturesampler ), context );
    m_texturesampler = 0;
  }

  inline void TextureSamplerObj::validate()
  {
    checkError( rtTextureSamplerValidate( m_texturesampler ) );
  }

  inline Context TextureSamplerObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtTextureSamplerGetContext( m_texturesampler, &c ) );
    return Context::take( c );
  }

  inline void TextureSamplerObj::setMipLevelCount(unsigned int  num_mip_levels)
  {
    checkError( rtTextureSamplerSetMipLevelCount(m_texturesampler, num_mip_levels ) );
  }

  inline unsigned int TextureSamplerObj::getMipLevelCount() const
  {
    unsigned int result;
    checkError( rtTextureSamplerGetMipLevelCount( m_texturesampler, &result ) );
    return result;
  }

  inline void TextureSamplerObj::setArraySize(unsigned int  num_textures_in_array)
  {
    checkError( rtTextureSamplerSetArraySize( m_texturesampler, num_textures_in_array ) );
  }

  inline unsigned int TextureSamplerObj::getArraySize() const
  {
    unsigned int result;
    checkError( rtTextureSamplerGetArraySize( m_texturesampler, &result ) );
    return result;
  }

  inline void TextureSamplerObj::setWrapMode(unsigned int dim, RTwrapmode wrapmode)
  {
    checkError( rtTextureSamplerSetWrapMode( m_texturesampler, dim, wrapmode ) );
  }

  inline RTwrapmode TextureSamplerObj::getWrapMode(unsigned int dim) const
  {
    RTwrapmode wrapmode;
    checkError( rtTextureSamplerGetWrapMode( m_texturesampler, dim, &wrapmode ) );
    return wrapmode;
  }

  inline void TextureSamplerObj::setFilteringModes(RTfiltermode  minification, RTfiltermode  magnification, RTfiltermode  mipmapping)
  {
    checkError( rtTextureSamplerSetFilteringModes( m_texturesampler, minification, magnification, mipmapping ) );
  }

  inline void TextureSamplerObj::getFilteringModes(RTfiltermode& minification, RTfiltermode& magnification, RTfiltermode& mipmapping) const
  {
    checkError( rtTextureSamplerGetFilteringModes( m_texturesampler, &minification, &magnification, &mipmapping ) );
  }

  inline void TextureSamplerObj::setMaxAnisotropy(float value)
  {
    checkError( rtTextureSamplerSetMaxAnisotropy(m_texturesampler, value ) );
  }

  inline float TextureSamplerObj::getMaxAnisotropy() const
  {
    float result;
    checkError( rtTextureSamplerGetMaxAnisotropy( m_texturesampler, &result) );
    return result;
  }

  inline void TextureSamplerObj::setMipLevelClamp(float minLevel, float maxLevel)
  {
    checkError( rtTextureSamplerSetMipLevelClamp(m_texturesampler, minLevel, maxLevel ) );
  }

  inline void TextureSamplerObj::getMipLevelClamp(float &minLevel, float &maxLevel) const
  {
    checkError( rtTextureSamplerGetMipLevelClamp( m_texturesampler, &minLevel, &maxLevel) );
  }

  inline void TextureSamplerObj::setMipLevelBias(float value)
  {
    checkError( rtTextureSamplerSetMipLevelBias(m_texturesampler, value ) );
  }

  inline float TextureSamplerObj::getMipLevelBias() const
  {
    float result;
    checkError( rtTextureSamplerGetMipLevelBias( m_texturesampler, &result) );
    return result;
  }

  inline int TextureSamplerObj::getId() const
  {
    int result;
    checkError( rtTextureSamplerGetId( m_texturesampler, &result) );
    return result;
  }

  inline void TextureSamplerObj::setReadMode(RTtexturereadmode  readmode)
  {
    checkError( rtTextureSamplerSetReadMode( m_texturesampler, readmode ) );
  }

  inline RTtexturereadmode TextureSamplerObj::getReadMode() const
  {
    RTtexturereadmode result;
    checkError( rtTextureSamplerGetReadMode( m_texturesampler, &result) );
    return result;
  }

  inline void TextureSamplerObj::setIndexingMode(RTtextureindexmode  indexmode)
  {
    checkError( rtTextureSamplerSetIndexingMode( m_texturesampler, indexmode ) );
  }

  inline RTtextureindexmode TextureSamplerObj::getIndexingMode() const
  {
    RTtextureindexmode result;
    checkError( rtTextureSamplerGetIndexingMode( m_texturesampler, &result ) );
    return result;
  }

  inline void TextureSamplerObj::setBuffer(unsigned int texture_array_idx, unsigned int mip_level, Buffer buffer)
  {
    checkError( rtTextureSamplerSetBuffer( m_texturesampler, texture_array_idx, mip_level, buffer->get() ) );
  }

  inline Buffer TextureSamplerObj::getBuffer(unsigned int texture_array_idx, unsigned int mip_level) const
  {
    RTbuffer result;
    checkError( rtTextureSamplerGetBuffer(m_texturesampler, texture_array_idx, mip_level, &result ) );
    return Buffer::take(result);
  }
   
  inline void TextureSamplerObj::setBuffer(Buffer buffer)
  {
    checkError( rtTextureSamplerSetBuffer( m_texturesampler, 0, 0, buffer->get() ) );
  }

  inline Buffer TextureSamplerObj::getBuffer() const
  {
    RTbuffer result;
    checkError( rtTextureSamplerGetBuffer(m_texturesampler, 0, 0, &result ) );
    return Buffer::take(result);
  }

  inline RTtexturesampler TextureSamplerObj::get()
  {
    return m_texturesampler;
  }

  inline void TextureSamplerObj::registerGLTexture()
  {
    checkError( rtTextureSamplerGLRegister( m_texturesampler ) );
  }

  inline void TextureSamplerObj::unregisterGLTexture()
  {
    checkError( rtTextureSamplerGLUnregister( m_texturesampler ) );
  }

  inline void BufferObj::destroy()
  {
    Context context = getContext();
    checkError( rtBufferDestroy( m_buffer ), context );
    m_buffer = 0;
  }

  inline void BufferObj::validate()
  {
    checkError( rtBufferValidate( m_buffer ) );
  }

  inline Context BufferObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtBufferGetContext( m_buffer, &c ) );
    return Context::take( c );
  }

  inline void BufferObj::setFormat(RTformat format)
  {
    checkError( rtBufferSetFormat( m_buffer, format ) );
  }

  inline RTformat BufferObj::getFormat() const
  {
    RTformat result;
    checkError( rtBufferGetFormat( m_buffer, &result ) );
    return result;
  }

  inline void BufferObj::setElementSize(RTsize size_of_element)
  {
    checkError( rtBufferSetElementSize ( m_buffer, size_of_element ) );
  }

  inline RTsize BufferObj::getElementSize() const
  {
    RTsize result;
    checkError( rtBufferGetElementSize ( m_buffer, &result) );
    return result;
  }

  inline void BufferObj::getDevicePointer(int optix_device_ordinal, void** device_pointer)
  {
    checkError( rtBufferGetDevicePointer( m_buffer, optix_device_ordinal, device_pointer ) );
  }

  inline void* BufferObj::getDevicePointer(int optix_device_ordinal)
  {
    void* dptr;
    getDevicePointer( optix_device_ordinal, &dptr );
    return dptr;
  }

  inline void BufferObj::setDevicePointer(int optix_device_ordinal, void* device_pointer)
  {
    checkError( rtBufferSetDevicePointer( m_buffer, optix_device_ordinal, device_pointer ) );
  }

  inline void BufferObj::markDirty()
  {
    checkError( rtBufferMarkDirty( m_buffer ) );
  }

  inline void BufferObj::setSize(RTsize width)
  {
    checkError( rtBufferSetSize1D( m_buffer, width ) );
  }

  inline void BufferObj::getSize(RTsize& width) const
  {
    checkError( rtBufferGetSize1D( m_buffer, &width ) );
  }
   
  inline void BufferObj::getMipLevelSize(unsigned int level, RTsize& width) const
  {
    checkError( rtBufferGetMipLevelSize1D( m_buffer, level, &width ) );
  }

  inline void BufferObj::setSize(RTsize width, RTsize height)
  {
    checkError( rtBufferSetSize2D( m_buffer, width, height ) );
  }

  inline void BufferObj::getSize(RTsize& width, RTsize& height) const
  {
    checkError( rtBufferGetSize2D( m_buffer, &width, &height ) );
  }

  inline void BufferObj::getMipLevelSize(unsigned int level, RTsize& width, RTsize& height) const
  {
    checkError( rtBufferGetMipLevelSize2D( m_buffer, level, &width, &height ) );
  }

  inline void BufferObj::setSize(RTsize width, RTsize height, RTsize depth)
  {
    checkError( rtBufferSetSize3D( m_buffer, width, height, depth ) );
  }

  inline void BufferObj::getSize(RTsize& width, RTsize& height, RTsize& depth) const
  {
    checkError( rtBufferGetSize3D( m_buffer, &width, &height, &depth ) );
  }

  inline void BufferObj::getMipLevelSize(unsigned int level, RTsize& width, RTsize& height, RTsize& depth) const
  {
    checkError( rtBufferGetMipLevelSize3D( m_buffer, level,  &width, &height, &depth ) );
  }

  inline void BufferObj::setSize(unsigned int dimensionality, const RTsize* dims)
  {
    checkError( rtBufferSetSizev( m_buffer, dimensionality, dims ) );
  }

  inline void BufferObj::getSize(unsigned int dimensionality, RTsize* dims) const
  {
    checkError( rtBufferGetSizev( m_buffer, dimensionality, dims ) );
  }

  inline unsigned int BufferObj::getDimensionality() const
  {
    unsigned int result;
    checkError( rtBufferGetDimensionality( m_buffer, &result ) );
    return result;
  }
   
  inline void BufferObj::setMipLevelCount(unsigned int levels)
  {
    checkError( rtBufferSetMipLevelCount( m_buffer, levels ) );
  }

  inline unsigned int BufferObj::getMipLevelCount() const
  {
    unsigned int result;
    checkError( rtBufferGetMipLevelCount( m_buffer, &result ) );
    return result;
  }
      
  inline int BufferObj::getId() const
  {
    int result;
    checkError( rtBufferGetId( m_buffer, &result ) );
    return result;
  }

  inline unsigned int BufferObj::getGLBOId() const
  {
    unsigned int result;
    checkError( rtBufferGetGLBOId( m_buffer, &result ) );
    return result;
  }

  inline void BufferObj::registerGLBuffer()
  {
    checkError( rtBufferGLRegister( m_buffer ) );
  }

  inline void BufferObj::unregisterGLBuffer()
  {
    checkError( rtBufferGLUnregister( m_buffer ) );
  }

  inline void* BufferObj::map( unsigned int level, unsigned int map_flags, void* user_owned )
  {
    // Note: the order of the 'level' and 'map_flags' argument flips here for compatibility with older versions of this wrapper
    void* result;
    checkError( rtBufferMapEx( m_buffer, map_flags, level, user_owned, &result ) );
    return result;
  }

  inline void BufferObj::unmap( unsigned int level )
  {
    checkError( rtBufferUnmapEx( m_buffer, level ) );
  }

  inline void BufferObj::bindProgressiveStream( Buffer source )
  {
    checkError( rtBufferBindProgressiveStream( m_buffer, source->get() ) );
  }

  inline void BufferObj::getProgressiveUpdateReady( int* ready, unsigned int* subframe_count, unsigned int* max_subframes )
  {
    checkError( rtBufferGetProgressiveUpdateReady( m_buffer, ready, subframe_count, max_subframes ) );
  }

  inline bool BufferObj::getProgressiveUpdateReady()
  {

    int ready = 0;
    checkError( rtBufferGetProgressiveUpdateReady( m_buffer, &ready, 0, 0 ) );
    return ( ready != 0 );
  }

  inline bool BufferObj::getProgressiveUpdateReady( unsigned int& subframe_count )
  {

    int ready = 0;
    checkError( rtBufferGetProgressiveUpdateReady( m_buffer, &ready, &subframe_count, 0 ) );
    return ( ready != 0 );
  }

  inline bool BufferObj::getProgressiveUpdateReady( unsigned int& subframe_count, unsigned int& max_subframes )
  {

    int ready = 0;
    checkError( rtBufferGetProgressiveUpdateReady( m_buffer, &ready, &subframe_count, &max_subframes ) );
    return ( ready != 0 );
  }

  inline RTbuffer BufferObj::get()
  {
    return m_buffer;
  }

  inline void BufferObj::setAttribute( RTbufferattribute attrib, RTsize size, const void *p )
  {
    checkError( rtBufferSetAttribute( m_buffer, attrib, size, p ) );
  }

  inline void BufferObj::getAttribute( RTbufferattribute attrib, RTsize size, void *p )
  {
    checkError( rtBufferGetAttribute( m_buffer, attrib, size, p ) );
  }

  inline Context VariableObj::getContext() const
  {
    RTcontext c;
    checkErrorNoGetContext( rtVariableGetContext( m_variable, &c ) );
    return Context::take( c );
  }

  inline void VariableObj::setUint(unsigned int u1)
  {
    checkError( rtVariableSet1ui( m_variable, u1 ) );
  }

  inline void VariableObj::setUint(unsigned int u1, unsigned int u2)
  {
    checkError( rtVariableSet2ui( m_variable, u1, u2 ) );
  }

  inline void VariableObj::setUint(unsigned int u1, unsigned int u2, unsigned int u3)
  {
    checkError( rtVariableSet3ui( m_variable, u1, u2, u3 ) );
  }

  inline void VariableObj::setUint(unsigned int u1, unsigned int u2, unsigned int u3, unsigned int u4)
  {
    checkError( rtVariableSet4ui( m_variable, u1, u2, u3, u4 ) );
  }

  inline void VariableObj::setUint(optix::uint2 u)
  {
    checkError( rtVariableSet2uiv( m_variable, &u.x ) );
  }

  inline void VariableObj::setUint(optix::uint3 u)
  {
    checkError( rtVariableSet3uiv( m_variable, &u.x ) );
  }

  inline void VariableObj::setUint(optix::uint4 u)
  {
    checkError( rtVariableSet4uiv( m_variable, &u.x ) );
  }

  inline void VariableObj::set1uiv(const unsigned int* u)
  {
    checkError( rtVariableSet1uiv( m_variable, u ) );
  }

  inline void VariableObj::set2uiv(const unsigned int* u)
  {
    checkError( rtVariableSet2uiv( m_variable, u ) );
  }

  inline void VariableObj::set3uiv(const unsigned int* u)
  {
    checkError( rtVariableSet3uiv( m_variable, u ) );
  }

  inline void VariableObj::set4uiv(const unsigned int* u)
  {
    checkError( rtVariableSet4uiv( m_variable, u ) );
  }

  inline void VariableObj::setMatrix2x2fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix2x2fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix2x3fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix2x3fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix2x4fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix2x4fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix3x2fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix3x2fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix3x3fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix3x3fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix3x4fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix3x4fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix4x2fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix4x2fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix4x3fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix4x3fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setMatrix4x4fv(bool transpose, const float* m)
  {
    checkError( rtVariableSetMatrix4x4fv( m_variable, (int)transpose, m ) );
  }

  inline void VariableObj::setFloat(float f1)
  {
    checkError( rtVariableSet1f( m_variable, f1 ) );
  }

  inline void VariableObj::setFloat(optix::float2 f)
  {
    checkError( rtVariableSet2fv( m_variable, &f.x ) );
  }

  inline void VariableObj::setFloat(float f1, float f2)
  {
    checkError( rtVariableSet2f( m_variable, f1, f2 ) );
  }

  inline void VariableObj::setFloat(optix::float3 f)
  {
    checkError( rtVariableSet3fv( m_variable, &f.x ) );
  }

  inline void VariableObj::setFloat(float f1, float f2, float f3)
  {
    checkError( rtVariableSet3f( m_variable, f1, f2, f3 ) );
  }

  inline void VariableObj::setFloat(optix::float4 f)
  {
    checkError( rtVariableSet4fv( m_variable, &f.x ) );
  }

  inline void VariableObj::setFloat(float f1, float f2, float f3, float f4)
  {
    checkError( rtVariableSet4f( m_variable, f1, f2, f3, f4 ) );
  }

  inline void VariableObj::set1fv(const float* f)
  {
    checkError( rtVariableSet1fv( m_variable, f ) );
  }

  inline void VariableObj::set2fv(const float* f)
  {
    checkError( rtVariableSet2fv( m_variable, f ) );
  }

  inline void VariableObj::set3fv(const float* f)
  {
    checkError( rtVariableSet3fv( m_variable, f ) );
  }

  inline void VariableObj::set4fv(const float* f)
  {
    checkError( rtVariableSet4fv( m_variable, f ) );
  }

  ///////
  inline void VariableObj::setInt(int i1)
  {
    checkError( rtVariableSet1i( m_variable, i1 ) );
  }

  inline void VariableObj::setInt(optix::int2 i)
  {
    checkError( rtVariableSet2iv( m_variable, &i.x ) );
  }

  inline void VariableObj::setInt(int i1, int i2)
  {
    checkError( rtVariableSet2i( m_variable, i1, i2 ) );
  }

  inline void VariableObj::setInt(optix::int3 i)
  {
    checkError( rtVariableSet3iv( m_variable, &i.x ) );
  }

  inline void VariableObj::setInt(int i1, int i2, int i3)
  {
    checkError( rtVariableSet3i( m_variable, i1, i2, i3 ) );
  }

  inline void VariableObj::setInt(optix::int4 i)
  {
    checkError( rtVariableSet4iv( m_variable, &i.x ) );
  }

  inline void VariableObj::setInt(int i1, int i2, int i3, int i4)
  {
    checkError( rtVariableSet4i( m_variable, i1, i2, i3, i4 ) );
  }

  inline void VariableObj::set1iv( const int* i )
  {
    checkError( rtVariableSet1iv( m_variable, i ) );
  }

  inline void VariableObj::set2iv( const int* i )
  {
    checkError( rtVariableSet2iv( m_variable, i ) );
  }

  inline void VariableObj::set3iv( const int* i )
  {
    checkError( rtVariableSet3iv( m_variable, i ) );
  }

  inline void VariableObj::set4iv( const int* i )
  {
    checkError( rtVariableSet4iv( m_variable, i ) );
  }

  ///////
  inline void VariableObj::setLongLong(long long i1)
  {
      checkError(rtVariableSet1ll(m_variable, i1));
  }

  inline void VariableObj::setLongLong(optix::longlong2 i)
  {
      checkError(rtVariableSet2llv(m_variable, &i.x));
  }

  inline void VariableObj::setLongLong(long long i1, long long i2)
  {
      checkError(rtVariableSet2ll(m_variable, i1, i2));
  }

  inline void VariableObj::setLongLong(optix::longlong3 i)
  {
      checkError(rtVariableSet3llv(m_variable, &i.x));
  }

  inline void VariableObj::setLongLong(long long i1, long long i2, long long i3)
  {
      checkError(rtVariableSet3ll(m_variable, i1, i2, i3));
  }

  inline void VariableObj::setLongLong(optix::longlong4 i)
  {
      checkError(rtVariableSet4llv(m_variable, &i.x));
  }

  inline void VariableObj::setLongLong(long long i1, long long i2, long long i3, long long i4)
  {
      checkError(rtVariableSet4ll(m_variable, i1, i2, i3, i4));
  }

  inline void VariableObj::set1llv(const long long* i)
  {
      checkError(rtVariableSet1llv(m_variable, i));
  }

  inline void VariableObj::set2llv(const long long* i)
  {
      checkError(rtVariableSet2llv(m_variable, i));
  }

  inline void VariableObj::set3llv(const long long* i)
  {
      checkError(rtVariableSet3llv(m_variable, i));
  }

  inline void VariableObj::set4llv(const long long* i)
  {
      checkError(rtVariableSet4llv(m_variable, i));
  }

  ///////
  inline void VariableObj::setULongLong(unsigned long long i1)
  {
      checkError(rtVariableSet1ull(m_variable, i1));
  }

  inline void VariableObj::setULongLong(optix::ulonglong2 i)
  {
      checkError(rtVariableSet2ullv(m_variable, &i.x));
  }

  inline void VariableObj::setULongLong(unsigned long long i1, unsigned long long i2)
  {
      checkError(rtVariableSet2ull(m_variable, i1, i2));
  }

  inline void VariableObj::setULongLong(optix::ulonglong3 i)
  {
      checkError(rtVariableSet3ullv(m_variable, &i.x));
  }

  inline void VariableObj::setULongLong(unsigned long long i1, unsigned long long i2, unsigned long long i3)
  {
      checkError(rtVariableSet3ull(m_variable, i1, i2, i3));
  }

  inline void VariableObj::setULongLong(optix::ulonglong4 i)
  {
      checkError(rtVariableSet4ullv(m_variable, &i.x));
  }

  inline void VariableObj::setULongLong(unsigned long long i1, unsigned long long i2, unsigned long long i3, unsigned long long i4)
  {
      checkError(rtVariableSet4ull(m_variable, i1, i2, i3, i4));
  }

  inline void VariableObj::set1ullv(const unsigned long long* i)
  {
      checkError(rtVariableSet1ullv(m_variable, i));
  }

  inline void VariableObj::set2ullv(const unsigned long long* i)
  {
      checkError(rtVariableSet2ullv(m_variable, i));
  }

  inline void VariableObj::set3ullv(const unsigned long long* i)
  {
      checkError(rtVariableSet3ullv(m_variable, i));
  }

  inline void VariableObj::set4ullv(const unsigned long long* i)
  {
      checkError(rtVariableSet4ullv(m_variable, i));
  }

  inline void VariableObj::setBuffer(Buffer buffer)
  {
    checkError( rtVariableSetObject( m_variable, buffer->get() ) );
  }

  inline void VariableObj::set(Buffer buffer)
  {
    checkError( rtVariableSetObject( m_variable, buffer->get() ) );
  }

  inline void VariableObj::setUserData(RTsize size, const void* ptr)
  {
    checkError( rtVariableSetUserData( m_variable, size, ptr ) );
  }

  inline void VariableObj::getUserData(RTsize size,       void* ptr) const
  {
    checkError( rtVariableGetUserData( m_variable, size, ptr ) );
  }

  inline void VariableObj::setTextureSampler(TextureSampler texturesampler)
  {
    checkError( rtVariableSetObject( m_variable, texturesampler->get() ) );
  }

  inline void VariableObj::set(TextureSampler texturesampler)
  {
    checkError( rtVariableSetObject( m_variable, texturesampler->get() ) );
  }

  inline void VariableObj::set(GeometryGroup group)
  {
    checkError( rtVariableSetObject( m_variable, group->get() ) );
  }

  inline void VariableObj::set(Group group)
  {
    checkError( rtVariableSetObject( m_variable, group->get() ) );
  }

  inline void VariableObj::set(Program program)
  {
    checkError( rtVariableSetObject( m_variable, program->get() ) );
  }

  inline void VariableObj::setProgramId(Program program)
  {
    int id = program->getId();
    setInt(id);
  }
  
  inline void VariableObj::set(Selector sel)
  {
    checkError( rtVariableSetObject( m_variable, sel->get() ) );
  }

  inline void VariableObj::set(Transform tran)
  {
    checkError( rtVariableSetObject( m_variable, tran->get() ) );
  }

  inline Buffer VariableObj::getBuffer() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTbuffer buffer = reinterpret_cast<RTbuffer>(temp);
    return Buffer::take(buffer);
  }

  inline std::string VariableObj::getName() const
  {
    const char* name;
    checkError( rtVariableGetName( m_variable, &name ) );
    return std::string(name);
  }

  inline std::string VariableObj::getAnnotation() const
  {
    const char* annotation;
    checkError( rtVariableGetAnnotation( m_variable, &annotation ) );
    return std::string(annotation);
  }

  inline RTobjecttype VariableObj::getType() const
  {
    RTobjecttype type;
    checkError( rtVariableGetType( m_variable, &type ) );
    return type;
  }

  inline RTvariable VariableObj::get()
  {
    return m_variable;
  }

  inline RTsize VariableObj::getSize() const
  {
    RTsize size;
    checkError( rtVariableGetSize( m_variable, &size ) );
    return size;
  }

  inline optix::GeometryGroup VariableObj::getGeometryGroup() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTgeometrygroup geometrygroup = reinterpret_cast<RTgeometrygroup>(temp);
    return GeometryGroup::take( geometrygroup );
  }

  inline optix::GeometryInstance VariableObj::getGeometryInstance() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTgeometryinstance geometryinstance = 
      reinterpret_cast<RTgeometryinstance>(temp);
    return GeometryInstance::take( geometryinstance );
  }

  inline optix::Group VariableObj::getGroup() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTgroup group = reinterpret_cast<RTgroup>(temp);
    return Group::take( group );
  }

  inline optix::Program VariableObj::getProgram() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTprogram program = reinterpret_cast<RTprogram>(temp);
    return Program::take(program);
  }

  inline optix::Selector VariableObj::getSelector() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTselector selector = reinterpret_cast<RTselector>(temp);
    return Selector::take( selector );
  }

  inline optix::TextureSampler VariableObj::getTextureSampler() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTtexturesampler sampler = reinterpret_cast<RTtexturesampler>(temp);
    return TextureSampler::take(sampler);
  }

  inline optix::Transform VariableObj::getTransform() const
  {
    RTobject temp;
    checkError( rtVariableGetObject( m_variable, &temp ) );
    RTtransform transform = reinterpret_cast<RTtransform>(temp);
    return Transform::take( transform );
  }

  inline float VariableObj::getFloat() const
  {
    float f;
    checkError( rtVariableGet1f( m_variable, &f ) );
    return f;
  }

  inline optix::float2 VariableObj::getFloat2() const
  {
      optix::float2 f;
      checkError( rtVariableGet2f( m_variable, &f.x, &f.y ) );
      return f;

  }

  inline optix::float3 VariableObj::getFloat3() const
  {
      optix::float3 f;
      checkError( rtVariableGet3f( m_variable, &f.x, &f.y, &f.z ) );
      return f;
  }

  inline optix::float4 VariableObj::getFloat4() const
  {
      optix::float4 f;
      checkError( rtVariableGet4f( m_variable, &f.x, &f.y, &f.z, &f.w ) );
      return f;
  }

  inline void VariableObj::getFloat(float& f1) const
  {
    checkError( rtVariableGet1f( m_variable, &f1 ) );
  }

  inline void VariableObj::getFloat(float& f1, float& f2) const
  {
      checkError( rtVariableGet2f( m_variable, &f1, &f2 ) );
  }

  inline void VariableObj::getFloat(float& f1, float& f2, float& f3) const
  {
      checkError( rtVariableGet3f( m_variable, &f1, &f2, &f3 ) );
  }

  inline void VariableObj::getFloat(float& f1, float& f2, float& f3,
                                 float& f4) const
  {
      checkError( rtVariableGet4f( m_variable, &f1, &f2, &f3, &f4 ) );
  }


  inline unsigned VariableObj::getUint()  const
  {
    unsigned u;
    checkError( rtVariableGet1ui( m_variable, &u ) );
    return u;
  }

  inline optix::uint2 VariableObj::getUint2() const
  {
    optix::uint2 u;
    checkError( rtVariableGet2ui( m_variable, &u.x, &u.y ) );
    return u;
  }

  inline optix::uint3 VariableObj::getUint3() const
  {
    optix::uint3 u;
    checkError( rtVariableGet3ui( m_variable, &u.x, &u.y, &u.z ) );
    return u;
  }

  inline optix::uint4 VariableObj::getUint4() const
  {
    optix::uint4 u;
    checkError( rtVariableGet4ui( m_variable, &u.x, &u.y, &u.z, &u.w ) );
    return u;
  }

  inline void VariableObj::getUint(unsigned& u1) const
  {
    checkError( rtVariableGet1ui( m_variable, &u1 ) );
  }

  inline void VariableObj::getUint(unsigned& u1, unsigned& u2) const
  {
    checkError( rtVariableGet2ui( m_variable, &u1, &u2 ) );
  }

  inline void VariableObj::getUint(unsigned& u1, unsigned& u2, unsigned& u3) const
  {
    checkError( rtVariableGet3ui( m_variable, &u1, &u2, &u3 ) );
  }

  inline void VariableObj::getUint(unsigned& u1, unsigned& u2, unsigned& u3,
                                unsigned& u4) const
  {
    checkError( rtVariableGet4ui( m_variable, &u1, &u2, &u3, &u4 ) );
  }

  inline int VariableObj::getInt()  const
  {
    int i;
    checkError( rtVariableGet1i( m_variable, &i ) );
    return i;
  }

  inline optix::int2 VariableObj::getInt2() const
  {
    optix::int2 i;
    checkError( rtVariableGet2i( m_variable, &i.x, &i.y ) );
    return i;
  }

  inline optix::int3 VariableObj::getInt3() const
  {
    optix::int3 i;
    checkError( rtVariableGet3i( m_variable, &i.x, &i.y, &i.z ) );
    return i;
  }

  inline optix::int4 VariableObj::getInt4() const
  {
    optix::int4 i;
    checkError( rtVariableGet4i( m_variable, &i.x, &i.y, &i.z, &i.w ) );
    return i;
  }

  inline void VariableObj::getInt(int& i1) const
  {
    checkError( rtVariableGet1i( m_variable, &i1 ) );
  }

  inline void VariableObj::getInt(int& i1, int& i2) const
  {
    checkError( rtVariableGet2i( m_variable, &i1, &i2 ) );
  }

  inline void VariableObj::getInt(int& i1, int& i2, int& i3) const
  {
    checkError( rtVariableGet3i( m_variable, &i1, &i2, &i3 ) );
  }

  inline void VariableObj::getInt(int& i1, int& i2, int& i3, int& i4) const
  {
    checkError( rtVariableGet4i( m_variable, &i1, &i2, &i3, &i4 ) );
  }

  inline unsigned long long VariableObj::getULongLong()  const
  {
      unsigned long long llu;
      checkError( rtVariableGet1ull( m_variable, &llu ) );
      return llu;
  }

  inline optix::ulonglong2 VariableObj::getULongLong2() const
  {
      optix::ulonglong2 llu;
      checkError( rtVariableGet2ull( m_variable, &llu.x, &llu.y ) );
      return llu;
  }

  inline optix::ulonglong3 VariableObj::getULongLong3() const
  {
      optix::ulonglong3 ull;
      checkError( rtVariableGet3ull( m_variable, &ull.x, &ull.y, &ull.z ) );
      return ull;
  }

  inline optix::ulonglong4 VariableObj::getULongLong4() const
  {
      optix::ulonglong4 ull;
      checkError( rtVariableGet4ull( m_variable, &ull.x, &ull.y, &ull.z, &ull.w ) );
      return ull;
  }

  inline void VariableObj::getULongLong(unsigned long long& ull1) const
  {
      checkError( rtVariableGet1ull( m_variable, &ull1 ) );
  }

  inline void VariableObj::getULongLong(unsigned long long& ull1, unsigned long long& ull2) const
  {
      checkError( rtVariableGet2ull( m_variable, &ull1, &ull2 ) );
  }

  inline void VariableObj::getULongLong(unsigned long long& ull1, unsigned long long& ull2, 
      unsigned long long& ull3) const
  {
      checkError( rtVariableGet3ull( m_variable, &ull1, &ull2, &ull3 ) );
  }

  inline void VariableObj::getULongLong(unsigned long long& ull1, unsigned long long& ull2, 
      unsigned long long& ull3, unsigned long long& ull4) const
  {
      checkError( rtVariableGet4ull( m_variable, &ull1, &ull2, &ull3, &ull4 ) );
  }

  inline long long VariableObj::getLongLong()  const
  {
      long long ll;
      checkError( rtVariableGet1ll( m_variable, &ll ) );
      return ll;
  }

  inline optix::longlong2 VariableObj::getLongLong2() const
  {
      optix::longlong2 ll;
      checkError( rtVariableGet2ll( m_variable, &ll.x, &ll.y ) );
      return ll;
  }

  inline optix::longlong3 VariableObj::getLongLong3() const
  {
      optix::longlong3 ll;
      checkError( rtVariableGet3ll( m_variable, &ll.x, &ll.y, &ll.z ) );
      return ll;
  }

  inline optix::longlong4 VariableObj::getLongLong4() const
  {
      optix::longlong4 ll;
      checkError( rtVariableGet4ll( m_variable, &ll.x, &ll.y, &ll.z, &ll.w ) );
      return ll;
  }

  inline void VariableObj::getLongLong(long long& ll1) const
  {
      checkError( rtVariableGet1ll( m_variable, &ll1 ) );
  }

  inline void VariableObj::getLongLong(long long& ll1, long long& ll2) const
  {
      checkError( rtVariableGet2ll( m_variable, &ll1, &ll2 ) );
  }

  inline void VariableObj::getLongLong(long long& ll1, long long& ll2, long long& ll3) const
  {
      checkError( rtVariableGet3ll( m_variable, &ll1, &ll2, &ll3 ) );
  }

  inline void VariableObj::getLongLong(long long& ll1, long long& ll2, long long& ll3, long long& ll4) const
  {
      checkError( rtVariableGet4ll( m_variable, &ll1, &ll2, &ll3, &ll4 ) );
  }

  inline void VariableObj::getMatrix2x2(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix2x2fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix2x3(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix2x3fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix2x4(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix2x4fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix3x2(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix3x2fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix3x3(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix3x3fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix3x4(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix3x4fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix4x2(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix4x2fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix4x3(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix4x3fv( m_variable, transpose, m ) );
  }

  inline void VariableObj::getMatrix4x4(bool transpose, float* m) const
  {
    checkError( rtVariableGetMatrix4x4fv( m_variable, transpose, m ) );
  }
}

#endif /* __optixu_optixpp_namespace_h__ */
