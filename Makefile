HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif

HIPCC=$(HIP_PATH)/bin/hipcc

TARGET=hcc

SOURCES = mfma_inline_asm.cpp
OBJECTS = $(SOURCES:.cpp=.o)

EXECUTABLE=./mfma_inline_asm



all: $(EXECUTABLE) test

CXXFLAGS =-g
CXX=$(HIPCC)

.PHONY: test

$(EXECUTABLE): $(OBJECTS)
	$(HIPCC) $(OBJECTS) -o $@


test: $(EXECUTABLE)
	$(EXECUTABLE)



clean:
	rm -f $(EXECUTABLE)
	rm -f $(OBJECTS)

