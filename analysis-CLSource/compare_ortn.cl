
//! This is a silly hack to make Eclipse's syntax highlighting work with OpenCL kernels

#ifndef __OPENCL_VERSION__

#define __kernel
#define __global
#define __local
#define get_local_id
#define get_local_id
#define get_global_id
#define get_local_size
#define barrier(CLK_LOCAL_MEM_FENCE)
#define get_group_id
#endif


#pragma OPENCL EXTENSION cl_amd_printf : enable


//		<-----xid-->
//    -----------------
//    |				|    |
//    |				|  yid
//    |				|    |
//    -----------------
//    <  --- W  -- >
//



float read_data(	__global float * img, int W, int H,
				int xid, int yid)
{
	return img[yid * W + xid];
}

__kernel
void compare_ortn_adk(
			__global float * previous, __global float * next,
			__global float * result,
			__local float * buff,
			int numIpts)
{
	//printf("begin compare orntn\n");
	int x;

	int thx = get_local_id(0);
    int tha = get_local_id(1) * get_local_size(0) + thx;
    if(get_global_id(0) >= (numIpts/2) )
    	return;

	float prev_img = 0;
	float next_img = 0;
	prev_img = previous[get_global_id(0)];

	//! This is a hack done for testing
	next = previous + numIpts/2;
	next_img = next[get_global_id(0)];
	
	buff[tha] = prev_img - next_img;
	//if(buff[tha] > 0.00000f)
	//{
	//}
	//printf("%d \t Prev img %f\t Next Img %f \n",get_global_id(0),prev_img,next_img);
	// Local memory based reduction to calculate average of all the differences

 	barrier(CLK_LOCAL_MEM_FENCE);
    int stride ; // 64/2
    for (stride = 16; stride>0; stride>>=1)
    {
        if (tha < stride)	
        {
            buff[tha] += buff[tha + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0)  == 0)
    {
    	result[get_group_id(0)] = buff[0];
    	//printf("Result is %f\n",buff[0]);
    }
 
    barrier(CLK_LOCAL_MEM_FENCE);
}
