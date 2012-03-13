
#include <CL/cl.h>
#include "stdio.h"
#include "stdlib.h"

#include "analysis-devices-utils.h"
#include "surf.h"

#include "opencl_utils.h"

#include "analysis-devices.h"


#ifndef COMPARE_ORTN_DEVICE_H
#define COMPARE_ORTN_DEVICE_H


class compare_ortn : public analysis_device
{

private:

	float THRESHOLD;
	result_buffer opbuff;
	cl_mem p_features;
	cl_mem n_features;

	cl_mem feature_count;
public:
	compare_ortn();

	bool  get_analysis_result(bool);

	//! This function handles initial set up only
	//! void initialize_analysis_kernels();
	void init_buffers(size_t mem_size);
	void assign_buffers_copy(float * prev, float * next,size_t mem_size);
	void assign_buffers_mapping(cl_mem prev, cl_mem next, size_t mem_size);
	//! Kernel Configuration function.
	//! Should be called before the analysis_device::inject_analysis() function
	void configure_analysis_kernel( int  );


	//! Not used.
	//! Has been disabled for now since the user cannot control when the
	//! analysis is launched exactly
	void launch_compare(
			cl_mem present_features, cl_mem next_features,
			int W, int H );

};

#endif // __COMPARE_DEVICE_
