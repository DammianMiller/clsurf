#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __local
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

typedef struct{
    float x;
    float y;
    float scale;
    float orientation;
    int laplacian;
    int clusterIndex;
    float descriptor[64];
} Ipoint;


int get_bucket(float scale)
{
	int t;
	t= (int)ceil(scale);
	return t;
}

__kernel void
bucketize_features(	__global Ipoint * d_ip,
					__global Ipoint * d_op,
					__global uint  * bucket_size_list,
					const unsigned int n_features,
					const unsigned int n_buckets,
					const unsigned int n_elements_per_bucket)
{
	int global_id = get_global_id(0);
	int bucket_no = get_bucket(d_ip[global_id].scale);
	int new_position = atom_add(&bucket_size_list[bucket_no],1);
	int op_posn = new_position + bucket_no*n_elements_per_bucket;
	d_op[new_position] =  d_ip[global_id];
}
