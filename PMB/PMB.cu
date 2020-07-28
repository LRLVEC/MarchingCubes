#include <_Time.h>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>
#include <curand_kernel.h>
//1024: {16, 16, 64}
//2048: {16, 16, 64}

unsigned int edgeTable[256]
{
	0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
	0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
	0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
	0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
	0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
	0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
	0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
	0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
	0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
	0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
	0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
	0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
	0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
	0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
	0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
	0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
	0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
	0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
	0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
	0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
	0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
	0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
	0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
	0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
	0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
	0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
	0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
	0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
	0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
	0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
	0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
	0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
};

unsigned short distinctEdgesTable[256]
{
	0x0000, 0xE001, 0x8001, 0x6002, 0x0001, 0xE002, 0x8002, 0x6003,
	0x4001, 0xA002, 0xC002, 0x2003, 0x4002, 0xA003, 0xC003, 0x2002,
	0x2001, 0xC002, 0xA002, 0x4003, 0x2002, 0xC003, 0xA003, 0x4004,
	0x6002, 0x8003, 0xE003, 0x0004, 0x6003, 0x8004, 0xE004, 0x0003,
	0x0001, 0xE002, 0x8002, 0x6003, 0x0002, 0xE003, 0x8003, 0x6004,
	0x4002, 0xA003, 0xC003, 0x2004, 0x4003, 0xA004, 0xC004, 0x2003,
	0x2002, 0xC003, 0xA003, 0x4002, 0x2003, 0xC004, 0xA004, 0x4003,
	0x6003, 0x8004, 0xE004, 0x0003, 0x6004, 0x8005, 0xE005, 0x0002,
	0x0001, 0xE002, 0x8002, 0x6003, 0x0002, 0xE003, 0x8003, 0x6004,
	0x4002, 0xA003, 0xC003, 0x2004, 0x4003, 0xA004, 0xC004, 0x2003,
	0x2002, 0xC003, 0xA003, 0x4004, 0x2003, 0xC004, 0xA004, 0x4005,
	0x6003, 0x8004, 0xE004, 0x0005, 0x6004, 0x8005, 0xE005, 0x0004,
	0x0002, 0xE003, 0x8003, 0x6004, 0x0003, 0xE004, 0x8002, 0x6003,
	0x4003, 0xA004, 0xC004, 0x2005, 0x4004, 0xA005, 0xC003, 0x2002,
	0x2003, 0xC004, 0xA004, 0x4003, 0x2004, 0xC005, 0xA003, 0x4002,
	0x6004, 0x8005, 0xE005, 0x0004, 0x6005, 0x8002, 0xE004, 0x0001,
	0x0001, 0xE002, 0x8002, 0x6003, 0x0002, 0xE003, 0x8003, 0x6004,
	0x4002, 0xA003, 0xC003, 0x2004, 0x4003, 0xA004, 0xC004, 0x2003,
	0x2002, 0xC003, 0xA003, 0x4004, 0x2003, 0xC004, 0xA004, 0x4005,
	0x6003, 0x8002, 0xE004, 0x0003, 0x6004, 0x8003, 0xE005, 0x0002,
	0x0002, 0xE003, 0x8003, 0x6004, 0x0003, 0xE004, 0x8004, 0x6005,
	0x4003, 0xA004, 0xC004, 0x2005, 0x4004, 0xA005, 0xC005, 0x2004,
	0x2003, 0xC004, 0xA004, 0x4003, 0x2004, 0xC005, 0xA005, 0x4004,
	0x6004, 0x8003, 0xE005, 0x0002, 0x6005, 0x8004, 0xE002, 0x0001,
	0x0002, 0xE003, 0x8003, 0x6004, 0x0003, 0xE004, 0x8004, 0x6005,
	0x4003, 0xA004, 0xC004, 0x2005, 0x4002, 0xA003, 0xC003, 0x2002,
	0x2003, 0xC004, 0xA004, 0x4005, 0x2004, 0xC005, 0xA005, 0x4002,
	0x6004, 0x8003, 0xE005, 0x0004, 0x6003, 0x8002, 0xE004, 0x0001,
	0x0003, 0xE004, 0x8004, 0x6005, 0x0004, 0xE005, 0x8003, 0x6004,
	0x4004, 0xA005, 0xC005, 0x2002, 0x4003, 0xA004, 0xC002, 0x2001,
	0x2002, 0xC003, 0xA003, 0x4002, 0x2003, 0xC004, 0xA002, 0x4001,
	0x6003, 0x8002, 0xE004, 0x0001, 0x6002, 0x8001, 0xE001, 0x0000
};

int triTable[256][16]
{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
	{3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
	{1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
	{4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
	{9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
	{2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
	{10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
	{5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
	{5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
	{8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
	{2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
	{7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
	{11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
	{5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
	{11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
	{11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
	{9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
	{6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
	{6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
	{6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
	{8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
	{7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
	{3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
	{5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
	{0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
	{9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
	{8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
	{5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
	{0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
	{6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
	{10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
	{10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
	{0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
	{3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
	{6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
	{9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
	{8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
	{3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
	{6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
	{10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
	{10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
	{7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
	{7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
	{2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
	{1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
	{11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
	{8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
	{0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
	{7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
	{7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
	{1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
	{10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
	{0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
	{7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
	{6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
	{9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
	{6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
	{4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
	{10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
	{8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
	{1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
	{8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
	{10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
	{4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
	{10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
	{9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
	{6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
	{7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
	{3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
	{7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
	{3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
	{6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
	{9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
	{1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
	{4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
	{7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
	{6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
	{0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
	{6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
	{0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
	{11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
	{6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
	{5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
	{9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
	{1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
	{1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
	{10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
	{0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
	{5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
	{10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
	{11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
	{9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
	{7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
	{2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
	{9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
	{9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
	{1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
	{9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
	{9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
	{0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
	{10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
	{2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
	{0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
	{0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
	{9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
	{5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
	{3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
	{5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
	{0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
	{9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
	{1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
	{3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
	{4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
	{9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
	{11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
	{11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
	{2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
	{9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
	{3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
	{1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
	{4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
	{4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
	{0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
	{3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
	{3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
	{0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
	{9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
	{1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

uchar4 edgeIDTable[12]
{
	{0, 0, 0, 0},
	{1, 0, 0, 1},
	{0, 1, 0, 0},
	{0, 0, 0, 1},
	{0, 0, 1, 0},
	{1, 0, 1, 1},
	{0, 1, 1, 0},
	{0, 0, 1, 1},
	{0, 0, 0, 2},
	{1, 0, 0, 2},
	{1, 1, 0, 2},
	{0, 1, 0, 2}
};

constexpr unsigned int N(1024);
constexpr unsigned int Nd2(N / 2);
constexpr unsigned int voxelXLv1(16);
constexpr unsigned int voxelYLv1(16);
constexpr unsigned int voxelZLv1(64);
constexpr unsigned int gridXLv1((N - 1) / (voxelXLv1 - 1));
constexpr unsigned int gridYLv1((N - 1) / (voxelYLv1 - 1));
constexpr unsigned int gridZLv1((N - 1) / (voxelZLv1 - 1));
const dim3 BlockSizeLv1{ voxelXLv1, voxelYLv1, 1 };
const dim3 GridSizeLv1{ gridXLv1, gridYLv1, gridZLv1 };
constexpr unsigned int blockNum(gridXLv1* gridYLv1* gridZLv1);

constexpr unsigned int countingThreadNumLv1(128);
constexpr unsigned int countingBlockNumLv1(blockNum / countingThreadNumLv1);

constexpr unsigned int voxelXLv2(4);
constexpr unsigned int voxelYLv2(4);
constexpr unsigned int voxelZLv2(8);
constexpr unsigned int blockXLv2(5);
constexpr unsigned int blockYLv2(5);
constexpr unsigned int blockZLv2(9);
const dim3 BlockSizeLv2{ voxelXLv2 * voxelYLv2, blockXLv2 * blockYLv2, 1 };
constexpr unsigned int voxelNumLv2(blockXLv2* blockYLv2* blockZLv2);

constexpr unsigned int countingThreadNumLv2(1024);
constexpr unsigned int gridXLv2(gridXLv1* blockXLv2);
constexpr unsigned int gridYLv2(gridYLv1* blockYLv2);
constexpr unsigned int gridZLv2(gridZLv1* blockZLv2);

const dim3 BlockSizeGenerating{ voxelXLv2, voxelYLv2, voxelZLv2 };

//A implementation of Parallel Marching Blocks algorithm
__inline__ __device__ float f(unsigned int x, unsigned int y, unsigned int z)
{
	constexpr float d(2.0f / N);
	float xf((int(x - Nd2)) * d);//[-1, 1)
	float yf((int(z - Nd2)) * d);
	float zf((int(z - Nd2)) * d);
	//if (x)return 1;
	//else return -1;
	return 1.f - 16.f * xf * yf * zf - 4.f * (xf * xf + yf * yf + zf * zf);
	//return xf * xf + yf * yf + zf * zf - 1.0f;
}

__inline__ __device__ float zeroPoint(unsigned int x, float v0, float v1)
{
	constexpr float d(2.0f / N);
	return ((x * v1 - (x + 1) * v0) / (v1 - v0) - Nd2) * d;
}

__global__ void computeMinMaxLv1(/*float* data, */float* minMax)
{
	unsigned int laneid;
	asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
	constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
	constexpr unsigned int warpNum(threadNum / 32);
	__shared__ float sminMax[64];
	unsigned int x(blockIdx.x * (voxelXLv1 - 1) + threadIdx.x);
	unsigned int y(blockIdx.y * (voxelYLv1 - 1) + threadIdx.y);
	unsigned int z(blockIdx.z * (voxelZLv1 - 1));
	unsigned int tid(threadIdx.x + voxelXLv1 * threadIdx.y);
	unsigned int blockid(blockIdx.x + gridXLv1 * (blockIdx.y + gridYLv1 * blockIdx.z));
	unsigned int warpid(tid >> 5);
	//float v(data[tid + threadNum * blockIdx.x]);
	float v(f(x, y, z));
	float minV(v), maxV(v);
	for (int c0(1); c0 < voxelZLv1; ++c0)
	{
		v = f(x, y, z + c0);
		if (v < minV)minV = v;
		if (v > maxV)maxV = v;
	}
#pragma unroll
	for (int c0(16); c0 > 0; c0 /= 2)
	{
		float t0, t1;
		t0 = __shfl_down_sync(0xffffffffu, minV, c0);
		t1 = __shfl_down_sync(0xffffffffu, maxV, c0);
		if (t0 < minV)minV = t0;
		if (t1 > maxV)maxV = t1;
	}
	if (laneid == 0)
	{
		sminMax[warpid] = minV;
		sminMax[warpid + warpNum] = maxV;
	}
	__syncthreads();
	if (warpid == 0)
	{
		minV = sminMax[laneid];
		maxV = sminMax[laneid + warpNum];
#pragma unroll
		for (int c0(warpNum / 2); c0 > 0; c0 /= 2)
		{
			float t0, t1;
			t0 = __shfl_down_sync(0xffffffffu, minV, c0);
			t1 = __shfl_down_sync(0xffffffffu, maxV, c0);
			if (t0 < minV)minV = t0;
			if (t1 > maxV)maxV = t1;
		}
		if (laneid == 0)
		{
			minMax[blockid * 2] = minV;
			minMax[blockid * 2 + 1] = maxV;
		}
	}
}

__global__ void compatingLv1(float isoValue, float* minMax, unsigned int* blockIndices, unsigned int* countedBlockNum)
{
	unsigned int laneid;
	asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
	constexpr unsigned int warpNum(countingThreadNumLv1 / 32);
	__shared__ unsigned int sums[32];
	unsigned int tid(threadIdx.x);
	unsigned int bIdx(blockIdx.x * countingThreadNumLv1 + tid);
	unsigned int warpid(tid >> 5);
	unsigned int test;
	if (minMax[2 * bIdx] <= isoValue && minMax[2 * bIdx + 1] >= isoValue)test = 1;
	else test = 0;
	unsigned int testSum(test);
#pragma unroll
	for (int c0(1); c0 < 32; c0 *= 2)
	{
		unsigned int tp(__shfl_up_sync(0xffffffffu, testSum, c0));
		if (laneid >= c0)testSum += tp;
	}
	if (laneid == 31)sums[warpid] = testSum;
	__syncthreads();
	if (warpid == 0)
	{
		unsigned warpSum = sums[laneid];
#pragma unroll
		for (int c0(1); c0 < warpNum; c0 *= 2)
		{
			unsigned int tp(__shfl_up_sync(0xffffffffu, warpSum, c0));
			if (laneid >= c0)warpSum += tp;
		}
		sums[laneid] = warpSum;
	}
	__syncthreads();
	if (warpid != 0)testSum += sums[warpid - 1];
	if (tid == countingThreadNumLv1 - 1 && testSum != 0)
		sums[31] = atomicAdd(countedBlockNum, testSum);
	__syncthreads();
	if (test)blockIndices[testSum + sums[31] - 1] = bIdx;
}

__global__ void computeMinMaxLv2(unsigned int* blockIndicesLv1, float* minMax)
{
	unsigned int tid(threadIdx.x);
	unsigned int voxelOffset(threadIdx.y);
	unsigned int blockIndex(blockIndicesLv1[blockIdx.x]);
	unsigned int tp(blockIndex);
	unsigned int x((blockIndex % gridXLv1) * (voxelXLv1 - 1) + (voxelOffset % 5) * (voxelXLv2 - 1) + (tid & 3));
	tp /= gridXLv1;
	unsigned int y((tp % gridYLv1) * (voxelYLv1 - 1) + (voxelOffset / 5) * (voxelYLv2 - 1) + (tid >> 2));
	tp /= gridYLv1;
	unsigned int z(tp * (voxelZLv1 - 1));
	float v(f(x, y, z));
	float minV(v), maxV(v);
	unsigned int idx(2 * (voxelOffset + voxelNumLv2 * blockIdx.x));
	for (int c0(0); c0 < blockZLv2; ++c0)
	{
		for (int c1(1); c1 < voxelZLv2; ++c1)
		{
			v = f(x, y, z + c1);
			if (v < minV)minV = v;
			if (v > maxV)maxV = v;
		}
		z += voxelZLv2 - 1;
#pragma unroll
		for (int c1(8); c1 > 0; c1 /= 2)
		{
			float t0, t1;
			t0 = __shfl_down_sync(0xffffffffu, minV, c1);
			t1 = __shfl_down_sync(0xffffffffu, maxV, c1);
			if (t0 < minV)minV = t0;
			if (t1 > maxV)maxV = t1;
		}
		if (tid == 0)
		{
			minMax[idx] = minV;
			minMax[idx + 1] = maxV;
			constexpr unsigned int offsetSize(2 * blockXLv2 * blockYLv2);
			idx += offsetSize;
		}
		minV = v;
		maxV = v;
	}
}

__global__ void compatingLv2(float isoValue, float* minMax,
	unsigned int* blockIndicesLv1, unsigned int* blockIndicesLv2,
	unsigned int counterBlockNumLv1, unsigned int* countedBlockNumLv2)
{
	unsigned int laneid;
	asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
	constexpr unsigned int warpNum(countingThreadNumLv2 / 32);
	__shared__ unsigned int sums[32];
	unsigned int tid(threadIdx.x);
	unsigned int warpid(tid >> 5);
	unsigned int id0(tid + blockIdx.x * countingThreadNumLv2);
	unsigned int id1(id0 / voxelNumLv2);
	unsigned int test;
	if (id1 < counterBlockNumLv1)
	{
		if (minMax[2 * id0] <= isoValue && minMax[2 * id0 + 1] >= isoValue)test = 1;
		else test = 0;
	}
	else test = 0;
	unsigned int testSum(test);
#pragma unroll
	for (int c0(1); c0 < 32; c0 *= 2)
	{
		unsigned int tp(__shfl_up_sync(0xffffffffu, testSum, c0));
		if (laneid >= c0)testSum += tp;
	}
	if (laneid == 31)sums[warpid] = testSum;
	__syncthreads();
	if (warpid == 0)
	{
		unsigned warpSum = sums[laneid];
#pragma unroll
		for (int c0(1); c0 < warpNum; c0 *= 2)
		{
			unsigned int tp(__shfl_up_sync(0xffffffffu, warpSum, c0));
			if (laneid >= c0)warpSum += tp;
		}
		sums[laneid] = warpSum;
	}
	__syncthreads();
	if (warpid != 0)testSum += sums[warpid - 1];
	if (tid == countingThreadNumLv2 - 1)
		sums[31] = atomicAdd(countedBlockNumLv2, testSum);
	__syncthreads();
	if (test)
	{
		unsigned int bIdx1(blockIndicesLv1[id1]);
		unsigned int bIdx2;
		unsigned int x1, y1, z1;
		unsigned int x2, y2, z2;
		unsigned int tp1(bIdx1);
		unsigned int tp2((tid + blockIdx.x * countingThreadNumLv2) % voxelNumLv2);
		x1 = tp1 % gridXLv1;
		x2 = tp2 % blockXLv2;
		tp1 /= gridXLv1;
		tp2 /= blockXLv2;
		y1 = tp1 % gridYLv1;
		y2 = tp2 % blockYLv2;
		z1 = tp1 / gridYLv1;
		z2 = tp2 / blockYLv2;
		bIdx2 = x2 + blockXLv2 * (x1 + gridXLv1 * (y2 + blockYLv2 * (y1 + gridYLv1 * (z1 * blockZLv2 + z2))));
		blockIndicesLv2[testSum + sums[31] - 1] = bIdx2;
	}
}

__global__ void generatingTriangles(
	float isoValue, unsigned int* blockIndicesLv2,
	unsigned short const* distinctEdgesTable, int const* triTable, uchar4 const* edgeIDTable,
	unsigned int* countedVerticesNum, unsigned int* countedTrianglesNum, float* vertices, unsigned int* triangles)
{
	unsigned int blockId(blockIndicesLv2[blockIdx.x]);
	unsigned int tp(blockId);
	unsigned int x((tp % gridXLv2) * (voxelXLv2 - 1) + threadIdx.x);
	tp /= gridXLv2;
	unsigned int y((tp % gridYLv2) * (voxelYLv2 - 1) + threadIdx.y);
	unsigned int z((tp / gridYLv2) * (voxelZLv2 - 1) + threadIdx.z);
	__shared__ unsigned short vertexIndices[voxelZLv2][voxelYLv2][voxelXLv2];
	__shared__ float value[voxelZLv2 + 1][voxelYLv2 + 1][voxelXLv2 + 1];
	unsigned int eds(7);
	float v(value[threadIdx.z][threadIdx.y][threadIdx.x] = f(x, y, z));
	if (threadIdx.x == voxelXLv2 - 1)
	{
		eds &= 6;
		value[threadIdx.z][threadIdx.y][voxelXLv2] = f(x + 1, y, z);
		if (threadIdx.y == voxelYLv2 - 1)
			value[threadIdx.z][voxelYLv2][voxelXLv2] = f(x + 1, y + 1, z);
	}
	if (threadIdx.y == voxelYLv2 - 1)
	{
		eds &= 5;
		value[threadIdx.z][voxelYLv2][threadIdx.x] = f(x, y + 1, z);
		if (threadIdx.z == voxelZLv2 - 1)
			value[voxelZLv2][voxelYLv2][threadIdx.x] = f(x, y + 1, z + 1);
	}
	if (threadIdx.z == voxelZLv2 - 1)
	{
		eds &= 3;
		value[voxelZLv2][threadIdx.y][threadIdx.x] = f(x, y, z + 1);
		if (threadIdx.x == voxelXLv2 - 1)
			value[voxelZLv2][threadIdx.y][voxelXLv2] = f(x + 1, y, z + 1);
	}
	eds <<= 13;
	__syncthreads();
	unsigned int cubeCase(0);
	if (value[threadIdx.z][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 1;
	if (value[threadIdx.z][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 2;
	if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 4;
	if (value[threadIdx.z][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 8;
	if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x] < isoValue) cubeCase |= 16;
	if (value[threadIdx.z + 1][threadIdx.y][threadIdx.x + 1] < isoValue) cubeCase |= 32;
	if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1] < isoValue) cubeCase |= 64;
	if (value[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x] < isoValue) cubeCase |= 128;
	unsigned int distinctEdges(eds ? distinctEdgesTable[cubeCase] : 0);
	unsigned int numTriangles(eds != 0xe000 ? 0 : distinctEdges & 7);
	unsigned int numVertices(__popc(distinctEdges &= eds));
	unsigned int laneid;
	asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
	unsigned warpid((threadIdx.x + voxelXLv2 * (threadIdx.y + voxelYLv2 * threadIdx.z)) >> 5);
	constexpr unsigned int threadNum(voxelXLv2 * voxelYLv2 * voxelZLv2);
	constexpr unsigned int warpNum(threadNum / 32);
	unsigned int sumVertices(numVertices);
	unsigned int sumTriangles(numTriangles);
	__shared__ unsigned int sumsVertices[32];
	__shared__ unsigned int sumsTriangles[32];
#pragma unroll
	for (int c0(1); c0 < 32; c0 *= 2)
	{
		unsigned int tp0(__shfl_up_sync(0xffffffffu, sumVertices, c0));
		unsigned int tp1(__shfl_up_sync(0xffffffffu, sumTriangles, c0));
		if (laneid >= c0)
		{
			sumVertices += tp0;
			sumTriangles += tp1;
		}
	}
	if (laneid == 31)
	{
		sumsVertices[warpid] = sumVertices;
		sumsTriangles[warpid] = sumTriangles;
	}
	__syncthreads();
	if (warpid == 0)
	{
		unsigned warpSumVertices = sumsVertices[laneid];
		unsigned warpSumTriangles = sumsTriangles[laneid];
#pragma unroll
		for (int c0(1); c0 < warpNum; c0 *= 2)
		{
			unsigned int tp0(__shfl_up_sync(0xffffffffu, warpSumVertices, c0));
			unsigned int tp1(__shfl_up_sync(0xffffffffu, warpSumTriangles, c0));
			if (laneid >= c0)
			{
				warpSumVertices += tp0;
				warpSumTriangles += tp1;
			}
		}
		sumsVertices[laneid] = warpSumVertices;
		sumsTriangles[laneid] = warpSumTriangles;
	}
	__syncthreads();
	if (warpid != 0)
	{
		sumVertices += sumsVertices[warpid - 1];
		sumTriangles += sumsTriangles[warpid - 1];
	}
	if (eds == 0)
	{
		sumsVertices[31] = atomicAdd(countedVerticesNum, sumVertices);
		sumsTriangles[31] = atomicAdd(countedTrianglesNum, sumTriangles);
	}
	/*__syncthreads();
	unsigned int interOffsetVertices(sumVertices - numVertices);
	sumVertices = interOffsetVertices + sumsVertices[31];//exclusive offset
	sumTriangles = sumTriangles + sumsTriangles[31] - numTriangles;//exclusive offset
	vertexIndices[threadIdx.z][threadIdx.y][threadIdx.x] = interOffsetVertices | distinctEdges;
	__syncthreads();
	for (unsigned int c0(0); c0 < numTriangles; ++c0)
	{
#pragma unroll
		for (unsigned int c1(0); c1 < 3; ++c1)
		{
			int edgeID(triTable[16 * cubeCase + 3 * c0 + c1]);
			uchar4 edgePos(edgeIDTable[edgeID]);
			unsigned short vertexIndex(vertexIndices[threadIdx.z + edgePos.z][threadIdx.y + edgePos.y][threadIdx.x + edgePos.x]);
			vertexIndex = __popc(vertexIndex >> (13 + edgePos.w)) + (vertexIndex & 0x1fff);
			triangles[3 * (sumTriangles + c0) + c1] = sumsVertices[31] + vertexIndex;
		}
	}
	if (distinctEdges & (1 << 15))
		vertices[sumVertices++] = zeroPoint(x, v, value[threadIdx.z][threadIdx.y][threadIdx.x + 1]);
	if (distinctEdges & (1 << 14))
		vertices[sumVertices++] = zeroPoint(y, v, value[threadIdx.z][threadIdx.y + 1][threadIdx.x]);
	if (distinctEdges & (1 << 13))
		vertices[sumVertices] = zeroPoint(z, v, value[threadIdx.z + 1][threadIdx.y][threadIdx.x]);*/
}

void convertToDistinctEdges()
{
	for (unsigned int c0(0); c0 < 256; ++c0)
	{
		unsigned int distinctEdges((edgeTable[c0] & 1) << 2);
		unsigned int n(0);
		while (triTable[c0][3 * n] >= 0)++n;
		distinctEdges |= (edgeTable[c0] >> 2) & (1 << 1);
		distinctEdges |= (edgeTable[c0] >> 8) & 1;
		distinctEdges <<= 13;
		distinctEdges |= n;
		::printf("0x%p,", distinctEdges);
		if ((c0 + 1 & 7) == 0)::printf("\n");
		else ::printf(" ");
	}
}

int main()
{
	std::uniform_real_distribution<float>rd(0, 1);
	std::mt19937 mt(time(nullptr));
	constexpr unsigned int threadNum(voxelXLv1 * voxelYLv1);
	constexpr size_t dataSize(threadNum * blockNum * sizeof(float));
	constexpr size_t minMaxSize(2 * blockNum * sizeof(float));
	//float* data((float*)::malloc(dataSize));
	//float* dataDebug((float*)::malloc(dataSize));
	//float* dataDevice;
	//float minV, maxV;
	//float* minMax((float*)::malloc(minMaxSize));
	//float* minMaxGPU((float*)::malloc(minMaxSize));
	float* minMaxLv1Device;
	float* minMaxLv2Device;
	unsigned int* blockIndicesLv1Device;
	unsigned int* blockIndicesLv2Device;
	unsigned int* countedBlockNumLv1Device;
	unsigned int* countedBlockNumLv2Device;
	unsigned short* distinctEdgesTableDevice;
	int* triTableDevice;
	uchar4* edgeIDTableDevice;
	unsigned int* countedVerticesNumDevice;
	unsigned int* countedTrianglesNumDevice;
	float* vertices;
	unsigned int* triangles;
	//cudaMalloc(&dataDevice, dataSize);
	cudaMalloc(&minMaxLv1Device, blockNum * 2 * sizeof(float));
	cudaMalloc(&blockIndicesLv1Device, blockNum * sizeof(unsigned int));
	cudaMalloc(&countedBlockNumLv1Device, sizeof(unsigned int));
	cudaMalloc(&countedBlockNumLv2Device, sizeof(unsigned int));
	cudaMalloc(&distinctEdgesTableDevice, sizeof(distinctEdgesTable));
	cudaMalloc(&triTableDevice, sizeof(triTable));
	cudaMalloc(&edgeIDTableDevice, sizeof(edgeIDTable));
	cudaMalloc(&countedVerticesNumDevice, sizeof(unsigned int));
	cudaMalloc(&countedTrianglesNumDevice, sizeof(unsigned int));
	cudaMemcpy(distinctEdgesTableDevice, distinctEdgesTable, sizeof(distinctEdgesTable), cudaMemcpyHostToDevice);
	cudaMemcpy(triTableDevice, triTable, sizeof(triTable), cudaMemcpyHostToDevice);
	cudaMemcpy(edgeIDTableDevice, edgeIDTable, sizeof(edgeIDTable), cudaMemcpyHostToDevice);
	//convertToDistinctEdges();
	for (unsigned int c0(0); c0 < 1; ++c0)
	{
		//for (unsigned int c1(0); c1 < blockNum; ++c1)
		//{
		//	minV = 1;
		//	maxV = 0;
		//	for (unsigned int c2(0); c2 < threadNum; ++c2)
		//	{
		//		float tp(rd(mt));
		//		data[c1 * threadNum + c2] = tp;
		//		if (tp < minV)minV = tp;
		//		if (tp > maxV)maxV = tp;
		//	}
		//	minMax[2 * c1] = minV;
		//	minMax[2 * c1 + 1] = maxV;
		//}
		//cudaMemcpy(dataDevice, data, dataSize, cudaMemcpyHostToDevice);
		Timer timer;
		cudaDeviceSynchronize();
		timer.begin();
		float isoValue(-0.9f);
		unsigned int countedBlockNumLv1(0);
		unsigned int countedBlockNumLv2(0);
		unsigned int countedVerticesNum(0);
		unsigned int countedTrianglesNum(0);
		cudaMemcpy(countedBlockNumLv1Device, &countedBlockNumLv1, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(countedBlockNumLv2Device, &countedBlockNumLv2, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(countedVerticesNumDevice, &countedVerticesNum, sizeof(unsigned int), cudaMemcpyHostToDevice);
		cudaMemcpy(countedTrianglesNumDevice, &countedTrianglesNum, sizeof(unsigned int), cudaMemcpyHostToDevice);

		computeMinMaxLv1 << <GridSizeLv1, BlockSizeLv1 >> > (/*dataDevice, */minMaxLv1Device);
		compatingLv1 << <countingBlockNumLv1, countingThreadNumLv1 >> > (isoValue, minMaxLv1Device, blockIndicesLv1Device, countedBlockNumLv1Device);

		cudaMemcpy(&countedBlockNumLv1, countedBlockNumLv1Device, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMalloc(&minMaxLv2Device, countedBlockNumLv1 * voxelNumLv2 * 2 * sizeof(float));

		computeMinMaxLv2 << < countedBlockNumLv1, BlockSizeLv2 >> > (blockIndicesLv1Device, minMaxLv2Device);

		cudaMalloc(&blockIndicesLv2Device, countedBlockNumLv1 * voxelNumLv2 * sizeof(unsigned int));
		unsigned int countingBlockNumLv2((countedBlockNumLv1 * voxelNumLv2 + countingThreadNumLv2 - 1) / countingThreadNumLv2);

		compatingLv2 << <countingBlockNumLv2, countingThreadNumLv2 >> > (isoValue, minMaxLv2Device, blockIndicesLv1Device, blockIndicesLv2Device, countedBlockNumLv1, countedBlockNumLv2Device);

		cudaMemcpy(&countedBlockNumLv2, countedBlockNumLv2Device, sizeof(unsigned int), cudaMemcpyDeviceToHost);

		//cudaMalloc(&vertices, countedBlockNumLv2 * 304 * sizeof(float));
		//cudaMalloc(&triangles, countedBlockNumLv2 * 315 * 3 * sizeof(unsigned int));
		generatingTriangles << <countedBlockNumLv2, BlockSizeGenerating >> > (
			isoValue, blockIndicesLv2Device,
			distinctEdgesTableDevice, triTableDevice, edgeIDTableDevice,
			countedVerticesNumDevice, countedTrianglesNumDevice, vertices, triangles);
		cudaDeviceSynchronize();
		timer.end();
		timer.print();
		::printf("Block Lv1: %u\nBlock Lv2: %u\n", countedBlockNumLv1, countedBlockNumLv2);
		::printf("Vertices Size: %u\n", countedBlockNumLv2 * 304 * sizeof(unsigned int));
		::printf("Triangles Size: %u\n", countedBlockNumLv2 * 315 * 3 * sizeof(unsigned int));
		cudaMemcpy(&countedVerticesNum, countedVerticesNumDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&countedTrianglesNum, countedTrianglesNumDevice, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		::printf("Vertices: %u\nTriangles: %u\n", countedVerticesNum, countedTrianglesNum);
		unsigned int* debugBuffer((unsigned int*)::malloc(10 * sizeof(unsigned int)));
		cudaMemcpy(debugBuffer, blockIndicesLv2Device, 10 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		for (int c1(0); c1 < 10; ++c1)
			::printf("%u\n", debugBuffer[c1]);
		//::printf("%u %u\n", countedBlockNumLv1, countedBlockNumLv2);
		free(debugBuffer);
		cudaFree(minMaxLv2Device);
		cudaFree(blockIndicesLv2Device);
		//cudaFree(vertices);
		//cudaFree(triangles);

		//cudaMemcpy(minMaxGPU, minMaxDevice, blockNum * 2 * sizeof(float), cudaMemcpyDeviceToHost);
		//for (unsigned int c1(0); c1 < blockNum; ++c1)
		//	if (minMax[2 * c1] != minMaxGPU[2 * c1] || minMax[2 * c1 + 1] != minMaxGPU[2 * c1 + 1])
		//		::printf("%d\n", c1);
	}
	//cudaFree(dataDevice);
	cudaFree(minMaxLv1Device);
	cudaFree(blockIndicesLv1Device);
	cudaFree(countedBlockNumLv1Device);
	cudaFree(countedBlockNumLv2Device);
	cudaFree(distinctEdgesTableDevice);
	cudaFree(triTableDevice);
	cudaFree(edgeIDTableDevice);
	cudaFree(countedVerticesNumDevice);
	cudaFree(countedTrianglesNumDevice);
}