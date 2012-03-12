
//! This is a silly hack to make Eclipse's syntax highlighting work with OpenCL kernels

#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
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
void compare(
			__global float * previous, __global float * next,
			__global float * result,
			__local float * buff,
			int W, int H)
{
	int x;
    // get the x & y indexes and absolute


	int thx = get_local_id(0);
    int tha = get_local_id(1) * get_local_size(0) + thx;
    //printf("Tha %d W %d H %d \n",tha,W,H);

    if(get_global_id(0) >= W )
    	return;
    if(get_global_id(1) >= H )
		return;

	float prev_img = //0;
			read_data(previous, W,H, get_global_id(0), get_global_id(1));
	float next_img = //0;
			read_data(next, W,H, get_global_id(0), get_global_id(1));
	

	buff[tha] = prev_img - next_img;
	//printf("Prev img %f\t Next Img %f \n",prev_img,next_img);

	if(buff[tha] > 0.00000f)
	{
		//printf("Prev img %f\t Next Img %f \n",prev_img,next_img);
	}
	// Local memory based reduction to calculate average of all the differences

 	barrier(CLK_LOCAL_MEM_FENCE);
    int stride ; // 64/2
    for (stride = 32; stride>0; stride>>=1)
    {
        if (tha < stride)	
        {
            buff[tha] += buff[tha + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0) == 0)
    {
    	//	printf("Locn is %d \n",get_group_id(0));
    	//printf("Result is %f\n",buff[0]);
    	result[get_group_id(0)] = buff[0];

    }
    barrier(CLK_LOCAL_MEM_FENCE);

	
}


