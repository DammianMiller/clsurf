
#include <CL/cl.h>

#include "analysis-devices.h"
#include "opencl_utils.h"
#include "surf.h"


#ifndef BUCKETIZE_DEVICE_H
#define BUCKETIZE_DEVICE_H

class bucketize_features : public analysis_device
{
	cl_mem ip_feature_list;
	cl_mem op_feature_list;
	int no_of_features;

	int no_of_buckets;

public:
	bucketize_features();
	void make_buckets();
	void init_buffers(int);
	void configure_analysis_kernel( int  );
};

#endif //__BUCKETIZE_DEVICE
