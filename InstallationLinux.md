# Dependencies #
  * OpenCV: This is a well known computer vision library. It can be downloaded from http://opencv.willowgarage.com.
    * Alternatively Ubuntu users can install OpenCV using the commands below

```
  sudo apt-get install build-essential 
  sudo apt-get install libavformat-dev
  sudo apt-get install ffmpeg
  sudo apt-get install libcv2.1 libcvaux2.1 libhighgui2.1 python-opencv 
  sudo apt-get install opencv-doc libcv-dev libcvaux-dev libhighgui-dev
```

  * OpenCL Runtime and any compliant device:
    * SURF has been tested with AMD's OpenCL implementation which can be downloaded from http://developer.amd.com/gpu/AMDAPPSDK/
    * It also works with Nvidia's OpenCL implementation which can be downloaded from http://developer.nvidia.com/cuda-downloads

# Installation steps #
  * Check out read-only copy of the source
  * Modify Makefile to reflect where you have installed OpenCV and the Nvidia OpenCL implementation (An Autools based build process will be added soon)
  * Run make
  * The executable should be run from the source directory as
```
  ./bin/amd/OpenSurf 2 -i  video.avi
```