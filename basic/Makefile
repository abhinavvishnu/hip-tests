HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES=square.cpp

all: square.out vectoradd.out hipmemcpyasync.out

# Step
square.out: square.cpp
	$(HIPCC) $(CXXFLAGS) square.cpp -o $@

vectoradd.out: vectoradd.cpp
	$(HIPCC) $(CXXFLAGS) vectoradd.cpp -o $@

hipmemcpyasync.out: hipmemcpyasync.cpp
	$(HIPCC) $(CXXFLAGS) hipmemcpyasync.cpp -o $@
clean:
	rm -f *.o *.out
