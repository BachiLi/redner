
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
 * @file   optix.h
 * @author NVIDIA Corporation
 * @brief  OptiX public API header
 *
 * Includes the host api if compiling host code, includes the cuda api if compiling device code.
 * For the math library routines include optix_math.h
 */

/******************************************************************************\
 *
 * Primary OptiX include file -- includes the host api if compiling host code,
 *                               includes the cuda api if compiling device code
 *
 * For the math library routines include optix_math.h.
 *
\******************************************************************************/

#ifndef __optix_optix_h__
#define __optix_optix_h__

#define OPTIX_VERSION 60500  /* major =  OPTIX_VERSION/10000,        *
                              * minor = (OPTIX_VERSION%10000)/100,   *
                              * micro =  OPTIX_VERSION%100           */


#ifdef __CUDACC__
#  include "optix_device.h"
#else
#  include "optix_host.h"
#endif


#endif /* __optix_optix_h__ */
