HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIP_PLATFORM=$(shell $(HIP_PATH)/bin/hipconfig --platform)
HIPCC=$(HIP_PATH)/bin/hipcc

SOURCES=square.cpp

all: square.out

# Step
square.out: $(SOURCES)
	$(HIPCC) $(CXXFLAGS) $(SOURCES) -o $@

clean:
	rm -f *.o *.out
