#include <CL/cl.h>
#include <math.h>
#include <eventlist.h>
#include "ad_rule_vec.h"

#include "device-compare-ortn.h"

#include "fissionutils.h"

//enum image_similarity;

/**
 * Call base constructor too.
 */
compare_ortn::compare_ortn():analysis_device()
{
	printf("Derived Class - Compare Orientation Device\n");
	THRESHOLD = 2.0f;

}

void compare_ortn::init_buffers(size_t mem_size)
{

	printf("Allocating %d bytes for orntn",mem_size);
    cl_int status;
    p_features = clCreateBuffer(getContext(),
						CL_MEM_READ_WRITE ,
						mem_size, NULL, &status);
    ad_errChk(status, "Error allocating pinned memory", true);
    n_features = clCreateBuffer(getContext(),
						CL_MEM_READ_WRITE ,
						mem_size, NULL, &status);
	ad_errChk(status, "Error allocating  memory", true);

	opbuff.allocate_buffer(mem_size,getContext());

	//analysis_rules.add_rule(i);

}

/**
 * Assign data to the analysis device's buffers
 * @param prev Previous image
 * @param next Next image
 * @param mem_size Data size
 */
void compare_ortn::assign_buffers_copy(float * prev, float * next, size_t mem_size)
{
	//! Uses the cl_map calls to map the pointers passed to the
	//! buffer objects
	printf("Copy %d bytes for orntn",mem_size);

 	copyHostToAd(p_features,prev,mem_size);
//	copyHostToAd(n_features,next,mem_size);

	//p_img = NULL;
	//n_img = NULL;
}



bool compare_ortn::get_analysis_result()
{
	bool return_state;
	//! Read results from processing
	//! Assume that the kernel injection is finished now
	sync();
 	float * data = (float *)mapBuffer(opbuff.buffer, opbuff.mem_size,CL_MAP_READ);
	float diff_value = 0.0f;
	//for(int i=0;i < (kernel_vec.at(0)->globalws[0]); i++)
	for(int i=0;i < 100; i++)
	{
 		diff_value = diff_value + data[i];
	}

	if(abs(diff_value) > THRESHOLD)
		return_state = ENABLED;
	else
		return_state = DISABLED;
	//printf("Diff is %f \n",diff_value);
	return return_state;
}

//! Configure the analysis kernel.
//! At this stage the kernel should be allocated and compiled
//! \param p_img Present Image
//! \param p_img Next Image
void compare_ortn::configure_analysis_kernel( int numIpts)
{
	//	printf("Setting Arguments and Config Analysis Kernel\n");

	//! If present_image and next_image
	kernel_vec.at(0)->dim_globalws = 1;
	kernel_vec.at(0)->dim_localws = 1;
	kernel_vec.at(0)->localws[0] = 16;
	printf("No of ipts %d as seen from AD\n",numIpts);
	kernel_vec.at(0)->globalws[0] = idivup(numIpts,kernel_vec.at(0)->localws[0]);

	kernel_vec.at(0)->localmemsize = (sizeof(float)*(kernel_vec.at(0)->localws[0]));

 	ad_setKernelArg(getKernel(0), 0,sizeof(cl_mem),(void *)&p_features);
	ad_setKernelArg(getKernel(0), 1,sizeof(cl_mem),(void *)&n_features);
	ad_setKernelArg(getKernel(0), 2,sizeof(cl_mem),(void *)&(opbuff.buffer));
	ad_setKernelArg(getKernel(0), 3,kernel_vec.at(0)->localmemsize, NULL);
	ad_setKernelArg(getKernel(0), 4,sizeof(cl_int), (void *)&numIpts);
	ad_setKernelArg(getKernel(0), 5,sizeof(cl_int), (void *)&numIpts);


}



//! Input- Feature set
//! Output- Sorted set (sorted as per all Xcord followed by all Ycoord)
/*
int sort_features()
{


}

int orntn_check()
{
	cl_mem buff1;
	cl_mem buff2;
	cl_kernel ortn_compare;
	//! Get Feature Sets
	cl_setKernelArg()

	//! Get Feature Set2


}
*/

