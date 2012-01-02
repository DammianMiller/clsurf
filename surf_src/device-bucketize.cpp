#include "device-bucketize.h"
#include "opencl_utils.h"

bucketize_features::bucketize_features()
{
	no_of_features = 0;
}

void bucketize_features::init_buffers(int ip_feature_count)
{
	no_of_features = ip_feature_count;
	cl_int status;
	ip_feature_list = clCreateBuffer(getContext(),NULL,
			no_of_features*sizeof(Ipoint),NULL,&status);
	ad_errChk(status,"error init feature buffer");

}

void bucketize_features::configure_analysis_kernel( int  )
{
	ad_setKernelArg(getKernel(0),0,sizeof(cl_mem),ip_feature_list);
	ad_setKernelArg(getKernel(0),1,sizeof(cl_mem),op_feature_list);
}

//! Initialize the optimizations
void bucketize_features::make_buckets()
{
	inject_analysis(0);
}
