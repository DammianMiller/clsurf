
#ifndef __OPENCL_VERSION__
//! This is a silly hack to make Eclipse's syntax highlighting work with OpenCL kernels
#define __kernel
#define __global
#define __local
#endif

#pragma OPENCL EXTENSION cl_amd_printf:ENABLE

//float read_img(	__global float * img, int W, int H, 
//				int xid, int yid)
//{
//	return img[xid];
//}

__kernel
void compare(
			__global float * previous, __global float * next,
			__global float * result)
{
	
	result[get_global_id(0)] = previous[get_global_id(0)] + next[get_global_id(0)];
	
	
}


