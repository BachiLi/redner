#include "load_serialized.h"

#include <fstream>
#include <iostream>

#include "miniz.h"

// http://stackoverflow.com/questions/348833/how-to-know-the-exact-line-of-code-where-where-an-exception-has-been-caused
class fl_exception : public std::runtime_error {
    std::string msg;

    public:
    fl_exception(const std::string &arg, const char *file, int line) : std::runtime_error(arg) {
        std::ostringstream o;
        o << file << ":" << line << ": " << arg;
        msg = o.str();
    }
    ~fl_exception() throw() {
    }
    const char *what() const throw() {
        return msg.c_str();
    }
};

#define Error(arg) throw fl_exception(arg, __FILE__, __LINE__);

#define MTS_FILEFORMAT_VERSION_V3 0x0003
#define MTS_FILEFORMAT_VERSION_V4 0x0004

#define ZSTREAM_BUFSIZE 32768

namespace py = pybind11;

enum ETriMeshFlags {
    EHasNormals = 0x0001,
    EHasTexcoords = 0x0002,
    EHasTangents = 0x0004,  // unused
    EHasColors = 0x0008,
    EFaceNormals = 0x0010,
    ESinglePrecision = 0x1000,
    EDoublePrecision = 0x2000
};

class ZStream {
    public:
    /// Create a new compression stream
    ZStream(std::fstream &fs);
    void read(void *ptr, size_t size);
    virtual ~ZStream();

    private:
    std::fstream &fs;
    size_t fsize;
    z_stream m_inflateStream;
    uint8_t m_inflateBuffer[ZSTREAM_BUFSIZE];
};

ZStream::ZStream(std::fstream &fs) : fs(fs) {
    std::streampos pos = fs.tellg();
    fs.seekg(0, fs.end);
    fsize = (size_t)fs.tellg();
    fs.seekg(pos, fs.beg);

    int windowBits = 15;
    m_inflateStream.zalloc = Z_NULL;
    m_inflateStream.zfree = Z_NULL;
    m_inflateStream.opaque = Z_NULL;
    m_inflateStream.avail_in = 0;
    m_inflateStream.next_in = Z_NULL;

    int retval = inflateInit2(&m_inflateStream, windowBits);
    if (retval != Z_OK) {
        Error("Could not initialize ZLIB");
    }
}

void ZStream::read(void *ptr, size_t size) {
    uint8_t *targetPtr = (uint8_t *)ptr;
    while (size > 0) {
        if (m_inflateStream.avail_in == 0) {
            size_t remaining = fsize - fs.tellg();
            m_inflateStream.next_in = m_inflateBuffer;
            m_inflateStream.avail_in = (uInt)std::min(remaining, sizeof(m_inflateBuffer));
            if (m_inflateStream.avail_in == 0) {
                Error("Read less data than expected");
            }

            fs.read((char *)m_inflateBuffer, m_inflateStream.avail_in);
        }

        m_inflateStream.avail_out = (uInt)size;
        m_inflateStream.next_out = targetPtr;

        int retval = inflate(&m_inflateStream, Z_NO_FLUSH);
        switch (retval) {
            case Z_STREAM_ERROR: {
                Error("inflate(): stream error!");
            }
            case Z_NEED_DICT: {
                Error("inflate(): need dictionary!");
            }
            case Z_DATA_ERROR: {
                Error("inflate(): data error!");
            }
            case Z_MEM_ERROR: {
                Error("inflate(): memory error!");
            }
        };

        size_t outputSize = size - (size_t)m_inflateStream.avail_out;
        targetPtr += outputSize;
        size -= outputSize;

        if (size > 0 && retval == Z_STREAM_END) {
            Error("inflate(): attempting to read past the end of the stream!");
        }
    }
}

ZStream::~ZStream() {
    inflateEnd(&m_inflateStream);
}

void skip_to_idx(std::fstream &fs, const short version, const size_t idx) {
    // Go to the end of the file to see how many components are there
    fs.seekg(-sizeof(uint32_t), fs.end);
    uint32_t count = 0;
    fs.read((char *)&count, sizeof(uint32_t));
    size_t offset = 0;
    if (version == MTS_FILEFORMAT_VERSION_V4) {
        fs.seekg(-sizeof(uint64_t) * (count - idx) - sizeof(uint32_t), fs.end);
        fs.read((char *)&offset, sizeof(size_t));
    } else {  // V3
        fs.seekg(-sizeof(uint32_t) * (count - idx + 1), fs.end);
        uint32_t upos = 0;
        fs.read((char *)&upos, sizeof(unsigned int));
        offset = upos;
    }
    fs.seekg(offset, fs.beg);
    // Skip the header
    fs.ignore(sizeof(short) * 2);
}

template <typename Precision>
void load_position(ZStream &zs,
                   py::array_t<float> &vertices) {
    assert(vertices.ndim() == 2);
    auto v_acc = vertices.mutable_unchecked<2>();
    for (int i = 0; i < vertices.shape()[0]; i++) {
        Precision x, y, z;
        zs.read(&x, sizeof(Precision));
        zs.read(&y, sizeof(Precision));
        zs.read(&z, sizeof(Precision));
        v_acc(i, 0) = (float)x;
        v_acc(i, 1) = (float)y;
        v_acc(i, 2) = (float)z;
    }
}

template <typename Precision>
void load_normal(ZStream &zs,
                 py::array_t<float> &normals) {
    assert(normals.ndim() == 2);
    auto n_acc = normals.mutable_unchecked<2>();
    for (int i = 0; i < normals.shape()[0]; i++) {
        Precision x, y, z;
        zs.read(&x, sizeof(Precision));
        zs.read(&y, sizeof(Precision));
        zs.read(&z, sizeof(Precision));
        n_acc(i, 0) = (float)x;
        n_acc(i, 1) = (float)y;
        n_acc(i, 2) = (float)z;
    }
}

template <typename Precision>
void load_uv(ZStream &zs,
             py::array_t<float> &uvs) {
    assert(uvs.ndim() == 2);
    auto uv_acc = uvs.mutable_unchecked<2>();
    for (int i = 0; i < uvs.shape()[0]; i++) {
        Precision u, v;
        zs.read(&u, sizeof(Precision));
        zs.read(&v, sizeof(Precision));
        uv_acc(i, 0) = (float)u;
        uv_acc(i, 1) = (float)v;
    }
}

template <typename Precision>
void load_color(ZStream &zs,
                py::array_t<float> &colors) {
    assert(colors.ndim() == 2);
    auto color_acc = colors.mutable_unchecked<2>();
    for (int i = 0; i < colors.shape()[0]; i++) {
        Precision r, g, b;
        zs.read(&r, sizeof(Precision));
        zs.read(&g, sizeof(Precision));
        zs.read(&b, sizeof(Precision));
        color_acc(i, 0) = (float)r;
        color_acc(i, 1) = (float)g;
        color_acc(i, 2) = (float)b;
    }
}

MitsubaTriMesh load_serialized(const std::string &filename, int idx) {
    std::fstream fs(filename.c_str(), std::fstream::in | std::fstream::binary);
    // Format magic number, ignore it
    fs.ignore(sizeof(short));
    // Version number
    short version = 0;
    fs.read((char *)&version, sizeof(short));
    if (idx > 0) {
        skip_to_idx(fs, version, idx);
    }
    ZStream zs(fs);

    uint32_t flags;
    zs.read((char *)&flags, sizeof(uint32_t));
    std::string name;
    if (version == MTS_FILEFORMAT_VERSION_V4) {
        char c;
        while (true) {
            zs.read((char *)&c, sizeof(char));
            if (c == '\0')
                break;
            name.push_back(c);
        }
    }
    size_t vertex_count = 0;
    zs.read((char *)&vertex_count, sizeof(size_t));
    size_t triangle_count = 0;
    zs.read((char *)&triangle_count, sizeof(size_t));

    bool file_double_precision = flags & EDoublePrecision;
    // bool face_normals = flags & EFaceNormals;

    auto vertices = py::array_t<float>({(int)vertex_count, 3});
    auto normals = py::array_t<float>();
    auto uvs = py::array_t<float>();
    auto colors = py::array_t<float>();
    if (file_double_precision) {
        load_position<double>(zs, vertices);
    } else {
        load_position<float>(zs, vertices);
    }

    if (flags & EHasNormals) {
        normals = py::array_t<float>({(int)vertex_count, 3});
        if (file_double_precision) {
            load_normal<double>(zs, normals);
        } else {
            load_normal<float>(zs, normals);
        }
    }

    if (flags & EHasTexcoords) {
        uvs = py::array_t<float>({(int)vertex_count, 2});
        if (file_double_precision) {
            load_uv<double>(zs, uvs);
        } else {
            load_uv<float>(zs, uvs);
        }
    }

    if (flags & EHasColors) {
        colors = py::array_t<float>({(int)vertex_count, 3});
        if (file_double_precision) {
            load_color<double>(zs, colors);
        } else {
            load_color<float>(zs, colors);
        }
    }

    auto indices = py::array_t<int>({(int)triangle_count, 3});
    auto indices_acc = indices.mutable_unchecked<2>();
    for (int i = 0; i < (int)indices.shape()[0]; i++) {
        int i0, i1, i2;
        zs.read(&i0, sizeof(int));
        zs.read(&i1, sizeof(int));
        zs.read(&i2, sizeof(int));
        indices_acc(i, 0) = i0;
        indices_acc(i, 1) = i1;
        indices_acc(i, 2) = i2;
    }

    return MitsubaTriMesh{vertices, indices, uvs, normals};
}
