COMMON_REPO := ../../..

#Common Includes
include $(COMMON_REPO)/utility/boards.mk
include $(COMMON_REPO)/libs/xcl2/xcl2.mk
include $(COMMON_REPO)/libs/opencl/opencl.mk

# Host Application
hellohls4ml_SRCS=./src/host.cpp $(oclHelper_SRCS) $(xcl2_SRCS)
hellohls4ml_HDRS=$(xcl2_HDRS)
hellohls4ml_CXXFLAGS=-I./src/ $(opencl_CXXFLAGS) $(xcl2_CXXFLAGS) $(oclHelper_CXXFLAGS) -std=c++0x
hellohls4ml_LDFLAGS=$(opencl_LDFLAGS)

EXES=hellohls4ml

# Kernel
hls4ml_1layer_SRCS=./src/hls4ml_1layer.cpp
hls4ml_1layer_CLFLAGS=-k hls4ml_1layer -I./src

XOS=hls4ml_1layer
hls4ml_1layer_XOS=hls4ml_1layer
XCLBINS=hls4ml_1layer

# check
check_EXE=hellohls4ml
check_XCLBINS=hls4ml_1layer

CHECKS=check

include $(COMMON_REPO)/utility/rules.mk
