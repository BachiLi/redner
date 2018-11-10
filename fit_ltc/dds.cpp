#include "dds.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#pragma pack( push, 1 )
struct DDS_PIXELFORMAT
{
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
};

struct DDS_HEADER
{
    uint32_t        dwSize;
    uint32_t        dwFlags;
    uint32_t        dwHeight;
    uint32_t        dwWidth;
    uint32_t        dwPitchOrLinearSize;
    uint32_t        dwDepth;
    uint32_t        dwMipMapCount;
    uint32_t        dwReserved1[ 11 ];
    DDS_PIXELFORMAT ddspf;
    uint32_t        dwCaps;
    uint32_t        dwCaps2;
    uint32_t        dwCaps3;
    uint32_t        dwCaps4;
    uint32_t        dwReserved2;
};

struct DDS_HEADER_DXT10
{
    uint32_t dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag;
    uint32_t arraySize;
    uint32_t miscFlags2;
};
#pragma pack( pop )

DDS_PIXELFORMAT const DDSPF_DX10 = { sizeof(DDS_PIXELFORMAT), 0x00000004, 0x30315844, 0, 0, 0, 0, 0 };

uint32_t const DDS_MAGIC                        = 0x20534444; // "DDS "
uint32_t const DDS_HEADER_FLAGS_TEXTURE         = 0x00001007; // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT 
uint32_t const DDS_HEADER_FLAGS_PITCH           = 0x00000008;
uint32_t const DDS_SURFACE_FLAGS_TEXTURE        = 0x00001000; // DDSCAPS_TEXTURE
uint32_t const DDS_RESOURCE_DIMENSION_TEXTURE2D = 3;

bool SaveDDS( char const* path, unsigned format, unsigned texelSizeInBytes, unsigned width, unsigned height, void const* data )
{
    FILE* f = fopen( path, "wb" );
    if ( !f )
    {
        return false;
    }


    fwrite( &DDS_MAGIC, sizeof( DDS_MAGIC ), 1, f );

    DDS_HEADER hdr;
    memset( &hdr, 0, sizeof( hdr ) );
    hdr.dwSize              = sizeof( hdr );
    hdr.dwFlags             = DDS_HEADER_FLAGS_TEXTURE | DDS_HEADER_FLAGS_PITCH;
    hdr.dwHeight            = height;
    hdr.dwWidth             = width;
    hdr.dwDepth             = 1;
    hdr.dwMipMapCount       = 1;
    hdr.dwPitchOrLinearSize = width * texelSizeInBytes;
    hdr.ddspf               = DDSPF_DX10;
    hdr.dwCaps              = DDS_SURFACE_FLAGS_TEXTURE;
    fwrite( &hdr, sizeof( hdr ), 1, f );

    DDS_HEADER_DXT10 hdrDX10;
    memset( &hdrDX10, 0, sizeof( hdrDX10 ) );
    hdrDX10.dxgiFormat          = format;
    hdrDX10.resourceDimension   = DDS_RESOURCE_DIMENSION_TEXTURE2D;
    hdrDX10.arraySize           = 1;
    fwrite( &hdrDX10, sizeof( hdrDX10 ), 1, f );

    fwrite( data, width * height * texelSizeInBytes, 1, f );

    fclose( f );
    return true;
}