 /****************************************************************************\ 
 * Copyright (c) 2011, Advanced Micro Devices, Inc.                           *
 * All rights reserved.                                                       *
 *                                                                            *
 * Redistribution and use in source and binary forms, with or without         *
 * modification, are permitted provided that the following conditions         *
 * are met:                                                                   *
 *                                                                            *
 * Redistributions of source code must retain the above copyright notice,     *
 * this list of conditions and the following disclaimer.                      *
 *                                                                            *
 * Redistributions in binary form must reproduce the above copyright notice,  *
 * this list of conditions and the following disclaimer in the documentation  *
 * and/or other materials provided with the distribution.                     *
 *                                                                            *
 * Neither the name of the copyright holder nor the names of its contributors *
 * may be used to endorse or promote products derived from this software      *
 * without specific prior written permission.                                 *
 *                                                                            *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        *
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  *
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR *
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          *
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        *
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         *
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       *
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         *
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               *
 *                                                                            *
 * If you use the software (in whole or in part), you shall adhere to all     *
 * applicable U.S., European, and other export laws, including but not        *
 * limited to the U.S. Export Administration Regulations (�EAR�), (15 C.F.R.  *
 * Sections 730 through 774), and E.U. Council Regulation (EC) No 1334/2000   *
 * of 22 June 2000.  Further, pursuant to Section 740.6 of the EAR, you       *
 * hereby certify that, except pursuant to a license granted by the United    *
 * States Department of Commerce Bureau of Industry and Security or as        *
 * otherwise permitted pursuant to a License Exception under the U.S. Export  *
 * Administration Regulations ("EAR"), you will not (1) export, re-export or  *
 * release to a national of a country in Country Groups D:1, E:1 or E:2 any   *
 * restricted technology, software, or source code you receive hereunder,     *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such *
 * technology or software, if such foreign produced direct product is subject *
 * to national security controls as identified on the Commerce Control List   *
 *(currently found in Supplement 1 to Part 774 of EAR).  For the most current *
 * Country Group listings, or for additional information about the EAR or     *
 * your obligations under those regulations, please refer to the U.S. Bureau  *
 * of Industry and Security�s website at http://www.bis.doc.gov/.             *
 \****************************************************************************/

#ifndef SURF_H
#define SURF_H

#include <cv.h>


#include <CL/cl.h>
#include <ctime>
#include <vector>

#include "fasthessian.h"
#include "eventlist.h"
#include "device-bucketize.h"
#include "device-compare-images.h"
#include "device-compare-ortn.h"

// Uncomment the following define to use optimized data transfers
// when possible.  Note that AMD's use of memory mapping is 
// different than NVIDIA, so it will crash on NVIDIA's devices
#define OPTIMIZED_TRANSFERS

#define DESC_SIZE 64

#define _IMAGE_COMPARE

//#define _ORTN_CHECK

//#define _BUCKETIZE

#define _USE_ANALYSIS_DEVICES

//! Ipoint structure holds a interest point descriptor
typedef struct{
        float x;
        float y;
        float scale;
        float orientation;
        int laplacian;
        int clusterIndex;
        float descriptor[64];
} Ipoint;

typedef std::vector<Ipoint> IpVec;

class Surf {

  public:

	bool adevice_state;
	IplImage *img_gray ;
 	IplImage *prev_img_gray ;

	int runcount ;
	int skipcount;

#ifdef _IMAGE_COMPARE
	//! Analysis devices interface
	compare_images * adevice;

#endif

#ifdef _ORTN_CHECK

	compare_ortn * odevice;

#endif

#ifdef _BUCKETIZE

	bucketize_features * bdevice;

#endif

	Surf(int initialPoints, int i_height, int i_width,  int octaves,
           int intervals, int sample_step, float threshold, 
           cl_kernel* kernel_list);

    ~Surf();
    
    void reset_phase();
    //! Compute the integral image
    void computeIntegralImage(IplImage* source);
    
    //! Create the SURF descriptors
    void createDescriptors(int i_width, int i_height);

    //! Calculate Orientation for each Ipoint
    void  getOrientations(int i_width, int i_height);

    //! Rellocate OpenCL buffers if the number of ipoints is too high
    void reallocateIptBuffers();

    //! Resets the object state so that SURF can be run on a new frame
    void reset();

    //! Copy the descriptors from the GPU to the host
    IpVec* retrieveDescriptors();

    //! Run the main SURF loop
    void run(IplImage* img, bool upright);

    void set_pipeline_state(bool new_pipeline_state);

  private:

    bool pipeline_state;
    bool run_orientation_stage;
    // The actual number of ipoints for this image
    int numIpts; 

    //! The amount of ipoints we have allocated space for
    int maxIpts;

    //! A fast hessian object that will be used for detecting ipoints
    FastHessian* fh;

    //! The integral image
    cl_mem d_intImage;
    cl_mem d_tmpIntImage;   // orig orientation
    cl_mem d_tmpIntImageT1; // transposed
    cl_mem d_tmpIntImageT2; // transposed

    //! Number of surf descriptors
    cl_mem d_length;

    //! List of precompiled kernels
    cl_kernel* kernel_list;

    //! Array of Descriptors for each Ipoint
    cl_mem d_desc;

    //! Orientation of each Ipoint an array of float
    cl_mem d_orientation;
    
    cl_mem d_gauss25;
    
    cl_mem d_id;	

    cl_mem d_i;

    cl_mem d_j;

    //! Position data on the host
    float2* pixPos;

    //! Scale data on the host
    float* scale;

    //! Laplacian data on the host
    int* laplacian;

    //! Descriptor data on the host
    float* desc;

    //! Orientation data on the host
    float* orientation;

    //! Position buffer on the device
    cl_mem d_pixPos;

    //! Scale buffer on the device
    cl_mem d_scale;

    //! Laplacian buffer on the device
    cl_mem d_laplacian;

    //! Res buffer on the device
    cl_mem d_res;

#ifdef OPTIMIZED_TRANSFERS
    // If we are using pinned memory, we need additional
    // buffers on the host

    //! Position buffer on the host
    cl_mem h_pixPos;

    //! Scale buffer on the host
    cl_mem h_scale;

    //! Laplacian buffer on the host
    cl_mem h_laplacian;

    //! Descriptor buffer on the host
    cl_mem h_desc;

    //! Orientation buffer on the host
    cl_mem h_orientation;
#endif

    const static int j[16];
    
    const static int i[16];

    const static unsigned int id[13];

    const static float gauss25[49];
};
#endif
