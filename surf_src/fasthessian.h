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

#ifndef FASTHESSIAN_H
#define FASTHESSIAN_H

#include <cv.h>
#include "surf.h"

#include <CL/cl.h>

#include <vector>
#include <time.h>
#include <ctime>

#include "eventlist.h"
#include "responselayer.h"

static const int OCTAVES = 5;
static const int INTERVALS = 4;
static const float THRES = 0.0001f;
static const int SAMPLE_STEP = 2;

//! FastHessian Calculates array of hessian and co-ordinates of ipoints 
/*!
    FastHessian declaration\n
    Calculates array of hessian and co-ordinates of ipoints 
*/
class FastHessian {
  
  public:
    
    //! Destructor
    ~FastHessian();

    //! Constructor without image
    FastHessian(int i_height, 
                int i_width,
                const int octaves = OCTAVES, 
                const int intervals = INTERVALS, 
                const int sample_step = SAMPLE_STEP, 
                const float thres = THRES,
                cl_kernel* kernel_list = NULL);

    // TODO Fix this name
    void selectIpoints(cl_mem d_laplacian, cl_mem d_pixPos, cl_mem d_scale,
                       cl_kernel* kernel_list, int maxPoints);
    
    // TODO Fix this name
    void computeHessianDet(cl_mem d_intImage, int i_width, int i_height, 
                           cl_kernel* kernel_list);

    //! Find the image features and write into vector of features
    int getIpoints(IplImage *img, cl_mem d_intImage, cl_mem d_laplacian,
                            cl_mem d_pixPos, cl_mem d_scale, int maxIpts);

    //! Resets the information required for the next frame to compute
    void reset();

  private:

    void createResponseMap(int octaves, int imgWidth, int 
        imgHeight, int sample_step);

    //! Number of Ipoints
    int num_ipts;

    //! Number of Octaves
    int octaves;

    //! Number of Intervals per octave
    int intervals;

    //! Initial sampling step for Ipoint detection
    int sample_step;

    //! Threshold value for blob resonses
    float thres;

    cl_kernel* kernel_list;

    std::vector<ResponseLayer*> responseMap;

    //! Number of Ipoints on GPU 
    cl_mem d_ipt_count;
};

#endif
