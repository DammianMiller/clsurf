EXECUTABLE    := OpenSURF

CCFILES      := clutils.cpp cvutils.cpp eventlist.cpp fasthessian.cpp \
                main.cpp nearestNeighbor.cpp responselayer.cpp surf.cpp \
                utils.cpp

C_DEPS       := clutils.h cvutils.h eventlist.h fasthessian.h \
                kmeans.h nearestNeighbor.h prf_util.h responselayer.h \
                surf.h utils.h

# Comment the following to disable building 
BUILD_AMD    = 1
#BUILD_NVIDIA = 1

# TODO: Replace with AMDAPPSDKROOT
AMD_OPENCL_INSTALL_PATH := $(ATISTREAMSDKROOT)
NVIDIA_OPENCL_INSTALL_PATH = /usr/local/cuda

OPENCV_INC := ../OpenCV2.2/include/opencv
OPENCV_LIB := ../OpenCV2.2/lib

# Basic directory setup 
SRCDIR         = src
AMD_BINDIR     = bin/amd
NVIDIA_BINDIR  = bin/nvidia
AMD_OBJDIR     = obj/amd
NVIDIA_OBJDIR  = obj/nvidia

# Compilers
CXX  = g++ -O3
CC   = gcc -O3
LINK = g++ -O3 

# Includes
COMMON_INCLUDES += -I$(OPENCV_INC)  
AMD_INCLUDES    += -I$(AMD_OPENCL_INSTALL_PATH)/include $(COMMON_INCLUDES)
NVIDIA_INCLUDES += -I$(NVIDIA_OPENCL_INSTALL_PATH)/include $(COMMON_INCLUDES)

# Libs
# NVIDIA installs their OpenCL library in /usr/lib64
COMMON_LIBS := -L$(OPENCV_LIB) -lopencv_imgproc -lopencv_core \
               -lopencv_highgui -lopencv_video 
AMD_LIB     := -L$(AMD_OPENCL_INSTALL_PATH)/lib/x86_64 -lOpenCL 
NVIDIA_LIB  := -L/usr/lib64 -lOpenCL 

# Warning flags
CXXWARN_FLAGS := -W -Wall -Wno-write-strings
CWARN_FLAGS := $(CXXWARN_FLAGS) -W -Wall -Wno-write-strings

# Compiler-specific flags
CXXFLAGS  :=  $(CXXWARN_FLAGS)
CFLAGS    :=  $(CWARN_FLAGS)

# Common flags
COMMONFLAGS = -DUNIX -O3

# Build executable commands
ifeq ($(BUILD_AMD),1)
AMD_TARGET     := $(AMD_BINDIR)/$(EXECUTABLE)
endif
ifeq ($(BUILD_NVIDIA),1)
NVIDIA_TARGET  := $(NVIDIA_BINDIR)/$(EXECUTABLE)
endif
AMD_LINKLINE    = $(LINK) -o $(AMD_TARGET) $(AMD_OBJS) $(AMD_LIB) \
                  $(COMMON_LIBS)
NVIDIA_LINKLINE = $(LINK) -o $(NVIDIA_TARGET) $(NVIDIA_OBJS) $(NVIDIA_LIB) \
                  $(COMMON_LIBS)


################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

################################################################################
# Set up object files
################################################################################
AMD_OBJS :=  $(patsubst %.cpp,$(AMD_OBJDIR)/%.cpp_o,$(notdir $(CCFILES)))
NVIDIA_OBJS :=  $(patsubst %.cpp,$(NVIDIA_OBJDIR)/%.cpp_o,$(notdir $(CCFILES)))
AMD_OBJS +=  $(patsubst %.c,$(AMD_OBJDIR)/%.c_o,$(notdir $(CFILES)))
NVIDIA_OBJS +=  $(patsubst %.c,$(NVIDIA_OBJDIR)/%.c_o,$(notdir $(CFILES)))


################################################################################
# Rules
################################################################################
$(AMD_OBJDIR)/%.c_o : $(SRCDIR)%.c
	$(VERBOSE)$(CC) $(AMD_INCLUDES) $(CFLAGS) -o $@ -c $<
$(NVIDIA_OBJDIR)/%.c_o : $(SRCDIR)%.c
	$(VERBOSE)$(CC) $(NVIDIA_INCLUDES) $(CFLAGS) -o $@ -c $<

$(AMD_OBJDIR)/%.cpp_o : $(SRCDIR)/%.cpp
	$(VERBOSE)$(CXX) $(AMD_INCLUDES) $(CXXFLAGS) -o $@ -c $<
$(NVIDIA_OBJDIR)/%.cpp_o : $(SRCDIR)/%.cpp
	$(VERBOSE)$(CXX) $(NVIDIA_INCLUDES) $(CXXFLAGS) -o $@ -c $<

all: $(AMD_TARGET) $(NVIDIA_TARGET)

$(AMD_TARGET): makedirectories $(AMD_OBJS)
	$(VERBOSE)$(AMD_LINKLINE)
$(NVIDIA_TARGET): makedirectories $(NVIDIA_OBJS)
	$(VERBOSE)$(NVIDIA_LINKLINE)

makedirectories:
	@mkdir -p $(AMD_OBJDIR)
	@mkdir -p $(NVIDIA_OBJDIR)
	@mkdir -p $(AMD_BINDIR)
	@mkdir -p $(NVIDIA_BINDIR)

clean:
	$(VERBOSE)rm -rf obj
