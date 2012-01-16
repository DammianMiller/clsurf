#include "device-bucketize.h"
#include "opencl_utils.h"

bucketize_features::bucketize_features()
{
	no_of_features = 0;
	no_of_buckets = 0;
}

void bucketize_features::assign_buffers(cl_mem input_feature_buffer)
{
	ip_feature_list = input_feature_buffer;
}

void bucketize_features::init_buffers(int ip_max_feature_count)
{
	printf("init bucket buffers\n");
	max_feature_count = ip_max_feature_count;
	max_feature_per_bucket = 1024;
	no_of_buckets = 5;

	cl_int status = CL_SUCCESS;
	ip_feature_list = clCreateBuffer(getContext(),CL_MEM_READ_WRITE,
			max_feature_count*sizeof(Ipoint),NULL,&status);
	ad_errChk(status,"error init feature buffer",true);

	op_feature_list = clCreateBuffer(getContext(),CL_MEM_READ_WRITE,
			no_of_buckets*max_feature_per_bucket*sizeof(Ipoint),NULL,&status);
	ad_errChk(status,"error init feature buffer",true);

	bucket_size_list = clCreateBuffer(getContext(),CL_MEM_READ_WRITE,
			no_of_buckets*sizeof(float),NULL,&status);
	ad_errChk(status,"error init bucket size list",true);

}

void bucketize_features::configure_analysis_kernel( int  n_features)
{

	ad_setKernelArg(getKernel(0),0,sizeof(cl_mem),(void *)&ip_feature_list);
	ad_setKernelArg(getKernel(0),1,sizeof(cl_mem),(void *)&op_feature_list);
	ad_setKernelArg(getKernel(0),2,sizeof(cl_mem),(void *)&bucket_size_list);
	ad_setKernelArg(getKernel(0),3,sizeof(cl_uint),&n_features);
	ad_setKernelArg(getKernel(0),4,sizeof(cl_uint),&no_of_buckets);
	ad_setKernelArg(getKernel(0),5,sizeof(cl_uint),&max_feature_per_bucket);

	kernel_vec.at(0)->localws[0]= 1;
	kernel_vec.at(0)->localws[1]= 1;
	kernel_vec.at(0)->globalws[0]= n_features;
	kernel_vec.at(0)->globalws[1]= 1;


}

//! Initialize the optimizations
void bucketize_features::make_buckets()
{
	inject_analysis(0);
}
