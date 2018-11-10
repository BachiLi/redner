#pragma once

unsigned const DDS_FORMAT_R32G32B32A32_FLOAT = 2;
unsigned const DDS_FORMAT_R32G32_FLOAT       = 16;
unsigned const DDS_FORMAT_R16G16_FLOAT       = 34;
unsigned const DDS_FORMAT_R32_FLOAT          = 41;

bool SaveDDS( char const* path, unsigned format, unsigned texelSizeInBytes, unsigned width, unsigned height, void const* data );