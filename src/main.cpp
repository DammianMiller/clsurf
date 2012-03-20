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
 * limited to the U.S. Export Administration Regulations (“EAR”), (15 C.F.R.  *
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
 * of Industry and Security’s website at http://www.bis.doc.gov/.             *
 \****************************************************************************/

#ifdef _WIN32
#pragma warning(disable:4127) // while loop has 'true' as parameter
#endif

#include <stdio.h>
#include <CL/cl.h>

#include "clutils.h"
#include "cv.h"
#include "eventlist.h"
#include "highgui.h"
#include "nearestNeighbor.h"
#include "cvutils.h"
#include "utils.h"
#include "fasthessian.h"
#include "surf.h"


// Signatures for main SURF functions
int mainImage(cl_kernel* kernel_list,char* inputImage, char* eventsPath,
              char* iptsPath, bool verifyResults);
int mainVideo(cl_kernel* kernel_list,char* inputImage, char* eventsPath,
              char* iptsPath);
int mainStabilization(cl_kernel* kernel_list,char* inputImage,
              char* eventsPath, char* iptsPath);
int mainBenchmark(cl_kernel* kernel_list,char* inputImage, char* eventsPath,
              char* iptsPath, bool verifyResults);

// Signature for reference implementation of SURF
int surfRef(char* imagePath, int octaves, int intervals, int step,
              float threshold, void** iptsPtr);

/**-------------------------------------------------------
 // Run the executable with no command line arguments to
 // see a complete list of options
 //
 // NOTE this program will crash with very small images
-------------------------------------------------------
*/
int main(int argc, char **argv)
{
    // At minimum, 2 arguments are required
    if(argc < 2)
    {
        printUsage();
        exit(-1);
    }

    // The procedure number determines what the remaining inputs should be
    int procedure = atoi(argv[1]);

    // Optional arguments
    char* inputPath = NULL;      // Path to input file
    char* iptsLogPath = NULL;    // Directory to store Ipts data
    char* eventsLogPath = NULL;  // Directory to store events
    char  devicePref = '\0';     // Device preference
    bool  verifyResults = false;

    // Check remaining arguments here
    parseArguments(argc, argv, &inputPath, &eventsLogPath, &iptsLogPath,
        &devicePref, &verifyResults);

    // If parameters were supplied, sanity check them
    if(inputPath != NULL) {
        checkFile(inputPath);
    }
    if(iptsLogPath != NULL) {
        checkDir(iptsLogPath);
    }
    if(eventsLogPath != NULL) {
        checkDir(eventsLogPath);
        cl_enableEvents();
    }

    // Check for required inputs based on procedure
    switch(procedure) {
    case 1:
        if(inputPath == NULL) {
            printf("Usage: Procedure 1 requires an input image\n");
            printUsage();
            exit(-1);
        }
        break;
    case 2:
        if(inputPath == NULL) {
            printf("No input video provided... webcam will be used\n");
        }
        break;
    case 3:
        if(inputPath == NULL) {
            printf("No input video provided... webcam will be used\n");
        }
        break;
    case 4:
        printf("Procedure 4 currently disabled\n");
        exit(-1);
    case 5:
        printf("Procedure 5 currently disabled\n");
        exit(-1);
    case 6:
        if(inputPath == NULL) {
            printf("Usage: Procedure 6 requires an input image\n");
            printUsage();
            exit(-1);
        }
        break;
    default:
        printf("Usage: Invalid procedure number was entered.  Must be 1-6.\n");
        printUsage();
        exit(-1);
    }

    // Arguments are correct, so print banner, initialize OpenCL,
    // and call selected procedure

    printf("|--------------------------------------------------|\n");
    printf("|--                                              --|\n");
    printf("|--        Welcome to SURF Imaging at AMD        --|\n");
    printf("|--                                              --|\n");
    printf("|--------------------------------------------------|\n\n");

    // Initialize OpenCL
    cl_init(devicePref);

	// NVIDIA's OpenCL cuurently doesn't support single-channel images
	if(cl_deviceIsNVIDIA())
	{

 		setUsingImages(false);
	}

    // Print a message saying whether or not images are being used
    if(isUsingImages())
    {
        printf("Using OpenCL images\n\n");
    }
    else
    {
        printf("Not using OpenCL images\n\n");
    }

    // Compile kernels off the critical path
    cl_kernel* kernel_list;
	if(isUsingImages())
	{
		kernel_list = cl_precompileKernels("-DIMAGES_SUPPORTED");
	}
	else {
		kernel_list = cl_precompileKernels(NULL);
	}

    // Call the selected procedure
    int retval = 0;
    switch(procedure) {
    case 1:
        retval = mainImage(kernel_list, inputPath, eventsLogPath, iptsLogPath,
                     verifyResults);
        break;
    case 2:
        retval = mainVideo(kernel_list, inputPath, eventsLogPath, iptsLogPath);
        break;
    case 3:
        retval = mainStabilization(kernel_list, inputPath, eventsLogPath,
                     iptsLogPath);
        break;
    case 6:
        retval = mainBenchmark(kernel_list, inputPath, eventsLogPath,
                     iptsLogPath, verifyResults);
        break;
    default:
        printf("Error: don't know how we got here\n");
        exit(-1);
    }

    return retval;
}


//--------------------------------------------------------
//  Procedure == 1: Image File
//--------------------------------------------------------
int mainImage(cl_kernel* kernel_list, char* inputImage, char* eventsPath,
              char* iptsPath, bool verifyResults)
{

    printf("Running an Image: %s\n", inputImage);

    // Set retval to negative if any errors occur
    int retval = 0;

    // Used to time execution
    cl_time surfStart, surfEnd;
    cl_time totalStart, totalEnd;

    // Start timing (OpenCL + OpenCV + host)
    cl_getTime(&totalStart);

    // Load the image using OpenCV
    IplImage *img=cvLoadImage(inputImage);

    // Initialize some SURF parameters
    int octaves = 5;
    int intervals = 4;
    int sample_step = 2;
    float threshold = 0.00005f;
    unsigned int initialIpts = 1000;

    // Create Surf Object
    Surf* surf = new Surf(initialIpts, img->height, img->width, octaves,
        intervals, sample_step, threshold, kernel_list);

    // Start timing (OpenCL only)
    cl_getTime(&surfStart);

    // This is the main SURF algorithm.  It detects and describes
    // interesting points in the image.   When the function completes
    // the descriptors are still on the device.
    surf->run(img, false);

    // Done timing SURF (OpenCL only)
    cl_getTime(&surfEnd);

    // Copy the surf descriptors back to the host for rendering
    IpVec* ipts;
    ipts = surf->retrieveDescriptors();

    // Done timing (OpenCV + OpenCL + host)
    cl_getTime(&totalEnd);

    // Create events based on the timer values
    cl_createUserEvent(surfStart, surfEnd, "OpenCL only");
    cl_createUserEvent(totalStart, totalEnd, "OpenCL+OpenCV+host");

    // Write interest points to file if path was supplied
    if(iptsPath != NULL) {
        writeIptsToFile(iptsPath, *ipts);
    }

    // Write events to file if path was supplied
    if(eventsPath != NULL) {
       cl_writeEventsToFile(eventsPath);
    }

    // If requested, compare the ipoints to the reference SURF implementation
    if(verifyResults) {
#ifdef _WIN32
        // Get Ipoints from the reference algorithm
        Ipoint* refIptsPtr;
        int numRefIpts = surfRef(inputImage, octaves, intervals,
                                 sample_step, threshold, (void**)&refIptsPtr);

        IpVec* refIpts = new IpVec(refIptsPtr, refIptsPtr+numRefIpts);

        IplImage *refImg = cvCloneImage(img);
        drawIpoints(refImg, *refIpts);

        // Compare the reference results to the OpenCL results
        if(!compareIpts(refIpts, ipts)) {
            retval = -1;
        }

        // Display the reference image
        showImage("reference", refImg, false);

        // Clean up the reference data
        cvReleaseImage(&refImg);
        delete refIpts;
        free(refIptsPtr);
#else
        printf("Verification only supported on Windows\n");
#endif
    }

    // Draw the descriptors on the image
    drawIpoints(img, *ipts);

    // Save the output image to file
    saveImage(img, inputImage);

    // Display the output image
    showImage(img);

    printf("Done with SURF Imaging\n");

    // Clean up
    delete surf;
    delete ipts;
    cvReleaseImage(&img);
    cvDestroyAllWindows();
    cl_cleanup();

    return retval;
}


//--------------------------------------------------------
//  Procedure == 2: Video file
//--------------------------------------------------------
int mainVideo(cl_kernel* kernel_list, char* inputImage, char* eventsPath, char* iptsPath)
{

    printf("Running a Video\n");

    // Open the video or webcam
    CvCapture* capture;
    if(inputImage == NULL)
    {
        // Using the webcam
        capture = cvCaptureFromCAM(CV_CAP_ANY);
        if(!capture) {
            printf("No Webcam Found\n");
            exit(-1);
        }
    }
    else
    {
        // Open the AVI video file
        capture = cvCaptureFromAVI(inputImage);
        if(!capture) {
            printf("Video could not be opened\n");
            exit(-1);
        }
    }

    // Create a window
    cvNamedWindow("OpenSURF", CV_WINDOW_AUTOSIZE);

    // Frame is the pointer to the current frame
    IplImage* frame = NULL;
    // These are the dimensions of the first frame (used for sanity checking)
    int firstWidth, firstHeight;

    // Initialize some SURF parameters
    int octaves = 3;
    int intervals = 4;
    int sample_step = 2;
    float threshold = THRES;
    unsigned int initialIpts = 1000;

    // Grab frame from the capture source
    frame = cvQueryFrame(capture);
    if(frame == NULL) {
        printf("No Frames Available\n");
        cvReleaseCapture(&capture);
        cvDestroyWindow("OpenSURF");
        return 0;
    }

    // Remember the height and width of the first frame to sanity
    // check future frames
    firstHeight = frame->height;
    firstWidth = frame->width;

    // Create Surf Descriptor Object
    Surf* surf = new Surf(initialIpts, frame->height, frame->width, octaves,
        intervals, sample_step, threshold, kernel_list);

    // ---------- Main capture loop -----------

    // Limit the loop to 1000 iterations
    int limit = 1000;
    int frame_id = 0;

    while(limit--)
    {

#ifdef PER_FRAME_TIMING

        cl_time t_start;
        cl_time t_end;
        cl_getTime(&t_start);

#endif //PER_FRAME_TIMING

        
        // Sanity check frame sizes
        if(frame->width != firstWidth || frame->height != firstHeight) {
            printf("Frames are inconsistent sizes, exiting loop\n");
            break;
        }

        // This is the main SURF algorithm.  It detects and describes
        // interesting points in the image.   When the function completes
        // the descriptors are still on the device.
        surf->run(frame, false);

        // The ipts hold the descriptors that describe the interesting
        // points in the image
        IpVec* ipts;

        // Copy the surf descriptors back to the host for rendering
        ipts = surf->retrieveDescriptors();

        // Draw the detected points
        drawIpoints(frame, *ipts);

        // Draw the FPS figure
        drawFPS(frame);

		// Write interest points to file if path was supplied
        if(iptsPath != NULL) {
            writeIptsToFile(iptsPath, *ipts);
        }

        // Display the result
        cvShowImage("OpenSURF", frame);

        // Clean up for the iteration
        delete ipts;
        surf->reset();

        // If ESC key pressed exit loop
        if( (cvWaitKey(2) & 255) == 27 ) break;

        // Grab frame from the capture source
        frame = cvQueryFrame(capture);
        if(frame == NULL) {
            printf("No Frames Available\n");
            break;
        }
        if(frame_id % 4 == 0)
        markphase(frame_id);

        recordphase(frame_id);

        frame_id=frame_id+1;

#ifdef PER_FRAME_TIMING

        cl_getTime(&t_end);
        printf("Time per Frame %f \n",cl_computeTime(t_start,t_end));
#endif  //PER_FRAME_TIMING

    }

    // Write events to file if path was supplied
    if(eventsPath != NULL) {
        cl_writeEventsToFile(eventsPath);
    }
    // Clean up
    delete surf;
    cvReleaseCapture(&capture);
    cvDestroyAllWindows();
    cl_cleanup();

    return 0;
}


//--------------------------------------------------------
//  Procedure == 3: Video Stabilization
//--------------------------------------------------------
int mainStabilization(cl_kernel* kernel_list, char* inputImage, char* eventsPath, char* iptsPath)
{
    printf("Running Video Stabilization\n");

    // Open the video or webcam
    CvCapture* capture;
    if(inputImage == NULL)
    {
        // Using the webcam
        capture = cvCaptureFromCAM(CV_CAP_ANY);
        if(!capture) {
            printf("No Webcam Found\n");
            exit(-1);
        }
    }
    else
    {
        // Open the AVI video file
        capture = cvCaptureFromAVI(inputImage);
        if(!capture) {
            printf("Video could not be opened\n");
            exit(-1);
        }
    }

    float scale = 0.5f;

    IplImage* frame = NULL;
    IplImage* firstFrame = NULL;
    IplImage* origFrame = NULL;

    // When we query the frame, the same buffer gets reused
    // (cannot free the frame, cannot use pointer assignment to
    // store it)
    origFrame = cvQueryFrame(capture);
    frame = cvCreateImage(cvSize((int)(origFrame->width*scale),
        (int)(origFrame->height*scale)), origFrame->depth, origFrame->nChannels);
    cvResize(origFrame, frame);

    std::vector<distPoint>* distancePoints;

    float shakeThreshold=10;
    printf("Shake threshold: %f\n", shakeThreshold);

    // Initialize some SURF parameters
    int octaves = 3;
    int intervals = 4;
    int sample_step = 2;
    float threshold = 0.001f; //THRES;
    unsigned int initialIpts = 1000;

    // Create Surf Descriptor Object
    Surf* surf = new Surf(initialIpts, frame->height, frame->width, octaves,
        intervals, sample_step, threshold, kernel_list);

    IpVec* firstIpts;
    IpVec* prevIpts = new IpVec;
    IpVec* nextIpts = NULL;

    surf->run(frame, false);

    // Set the previous frame to the first frame for the first
    // iteration of the loop
    firstIpts = surf->retrieveDescriptors();
    *prevIpts = *firstIpts;
    float** distTable = computeDistanceTable(firstIpts);

    // Store this frame to display
    firstFrame = cvCloneImage(frame);
    drawIpoints(firstFrame, *firstIpts);
    surf->reset();

    int frameCount = 1;

    createStabilizationWindows();

    while(true)
    {
        printf("Frame Count:%d\n",frameCount);

        // Grab the next frame from the capture source
        origFrame = cvQueryFrame(capture);
        if(origFrame == NULL) {
            printf("Reached Last Frame\n");
            break;
        }
        cvResize(origFrame, frame);

        // Run SURF on the next frame
        surf->run(frame, false);

        // Get the ipoints
        nextIpts = surf->retrieveDescriptors();

        // Find nearest neighbors
        distancePoints = findNearestNeighbors(*nextIpts, *firstIpts,
            kernel_list);

        // Draw the images on the screen
        drawIpoints(frame, *nextIpts);
        drawImages(firstFrame, frame, distancePoints, shakeThreshold, distTable);

        surf->reset();

        delete prevIpts;
        prevIpts = nextIpts;

        delete distancePoints;

        int waitKey = cvWaitKey(2) & 255;
        if(waitKey == 27) {

            // If ESC key pressed exit loop
            break;
        }
        else if(waitKey == 32) {
            // If SPACEBAR is pressed, set new reference image

            // Cleanup from the old reference image
            freeDistanceTable(distTable, firstIpts->size());
            delete firstIpts;
            cvReleaseImage(&firstFrame);

            // Grab the latest Ipts and set them as reference
            firstIpts = new IpVec;
            *firstIpts = *prevIpts;

            // Compute a new distance table
            distTable = computeDistanceTable(firstIpts);

            // Copy the latest image and draw the Ipoints on it
            firstFrame = cvCloneImage(frame);
            drawIpoints(firstFrame, *firstIpts);
        }


        frameCount++;
    }

    // Write reference interest points to file if path was supplied
    if(iptsPath != NULL) {
        writeIptsToFile(iptsPath, *firstIpts);
    }

    // Write events to file if path was supplied
    if(eventsPath != NULL) {
        cl_writeEventsToFile(eventsPath);
    }
    // Clean up
    if(nextIpts != NULL) {
        delete nextIpts;
    }

    freeDistanceTable(distTable, firstIpts->size());

    delete surf;
    delete firstIpts;
    cvReleaseImage(&firstFrame);
    cvReleaseCapture(&capture);
    cvDestroyAllWindows();
    cl_cleanup();

    return 0;
}


//--------------------------------------------------------
//  Procedure == 6: Benchmark
//--------------------------------------------------------
int mainBenchmark(cl_kernel* kernel_list, char* inputImage, char* eventsPath,
                  char* iptsPath, bool verifyResults)
{
    printf("Running benchmark on %s\n", inputImage);

    int retval = 0;

    // Load the input image using OpenCV
    IplImage *img=cvLoadImage(inputImage);
    if(img == NULL)
    {
        printf("Error loading image - Benchmark Mode \n");
        printf("You may have loaded a video or the file is wrong\n");
        exit(-1);
    }


    // Initialize some SURF parameters
    int octaves = 5;
    int intervals = 4;
    int sample_step = 2;
    float threshold = 0.00005f;
    unsigned int initialIpts = 1000;

    //----------------------------------------
    //             WARM UP RUN
    //----------------------------------------

    // Disable event logging while performing the warmup
    cl_disableEvents();

    // Create Surf Descriptor Object
    Surf* surf = new Surf(initialIpts, img->height, img->width, octaves,
        intervals, sample_step, threshold, kernel_list);

    // Since we're benchmarking, perform a warm-up run
    for(int i = 0; i < 5; i++) {
    surf->run(img, false);

    surf->reset();
    }

    // Re-enable event logging
    cl_enableEvents();

    //----------------------------------------
    //             TIMING RUN
    //----------------------------------------

    // Used to time execution
    cl_time totalStart, totalEnd;
    cl_time surfStart, surfEnd;
    cl_time copyStart, copyEnd;

    // Start overall timing
    cl_getTime(&totalStart);
    // Start timing SURF
    surfStart = totalStart;

    // This is the main SURF algorithm.  It detects and describes
    // interesting points in the image.  When the function completes
    // the descriptors are still on the device.
    surf->run(img, false);

    // Algorithm is complete
    cl_getTime(&surfEnd);
    // Start copying data back
    copyStart = surfEnd;

    // Copy the SURF descriptors to the host
    IpVec* ipts;
    ipts = surf->retrieveDescriptors();

    // This time includes the transfers back to the host
    cl_getTime(&copyEnd);
    totalEnd = copyEnd;

    // Create events based on the timer values
    cl_createUserEvent(surfStart, surfEnd, "RunningSurf");
    cl_createUserEvent(copyStart, copyEnd, "TransferBack");
    cl_createUserEvent(totalStart, totalEnd, "Total");

    // Write interest points to file if path was supplied
    if(iptsPath != NULL) {
        writeIptsToFile(iptsPath, *ipts);
    }

    // Write events to file if path was supplied
    if(eventsPath != NULL) {
       cl_writeEventsToFile(eventsPath);
    }

    // If requested, compare the ipoints to the reference SURF implementation
    if(verifyResults) {
#ifdef _WIN32
        // Get Ipoints from the reference algorithm
        Ipoint* refIptsPtr;
        int numRefIpts = surfRef(inputImage, octaves, intervals,
                                 sample_step, threshold, (void**)&refIptsPtr);

        IpVec* refIpts = new IpVec(refIptsPtr, refIptsPtr+numRefIpts);

        IplImage *refImg = cvCloneImage(img);
        drawIpoints(refImg, *refIpts);

        // Compare the reference results to the OpenCL results
        if(!compareIpts(refIpts, ipts)) {
            retval = -1;
        }

        // Display the reference image
        showImage("reference", refImg, false);

        // Clean up the reference data
        cvReleaseImage(&refImg);
        delete refIpts;
        free(refIptsPtr);
#else
        printf("Verification only supported on Windows\n");
#endif
    }

    printf("\nDone with SURF Imaging\n");

    // Clean up
    delete surf;
    delete ipts;
    cvDestroyAllWindows();
    cvReleaseImage(&img);
    cl_cleanup();

    return retval;
}
