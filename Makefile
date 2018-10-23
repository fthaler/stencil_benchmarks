-include Makefile.user

FLOAT_TYPE?=double

CXXFLAGS=-std=c++11 -O3 -MMD -MP -Wall -fopenmp -DNDEBUG -Isrc $(USERFLAGS) -DFLOAT_TYPE=$(FLOAT_TYPE)
NVCCFLAGS=-std=c++11 -arch=sm_60 -O3 -g -Xcompiler -fopenmp -DNDEBUG -Isrc $(USERFLAGS_CUDA) -DFLOAT_TYPE=$(FLOAT_TYPE)
LIBS=$(USERLIBS)


CXXVERSION=$(shell $(CXX) --version)
ifneq (,$(findstring icpc,$(CXXVERSION)))
	CXXFLAGS+=-ffreestanding
	CXXFLAGS_KNL=-xmic-avx512
else ifneq (,$(findstring g++,$(CXXVERSION)))
	LIBS+=-lgomp
	CXXFLAGS+=-fvect-cost-model=unlimited -Wno-unknown-pragmas
	CXXFLAGS_KNL+=-march=knl -mtune=knl -fvect-cost-model=unlimited
else ifneq (,$(findstring clang++,$(CXXVERSION)))
	CXXFLAGS_KNL+=-march=knl -mtune=knl
endif

SRCS=$(shell find . -name '*.cpp')
OBJS=$(SRCS:%.cpp=%.o)
DEPS=$(SRCS:%.cpp=%.d)

NVCCAVAIL:=$(shell command nvcc --version 2> /dev/null)

sbench: $(OBJS)
ifdef NVCCAVAIL
	nvcc $(NVCCFLAGS) $+ $(LIBS) -o $@
else
	$(CXX) $(CXXFLAGS) $+ $(LIBS) -o $@
endif

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/platform/knl.o: src/platform/knl.cpp
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_KNL) -c $< -o $@

src/platform/nvidia.o: src/platform/nvidia.cpp
ifdef NVCCAVAIL
	nvcc $(NVCCFLAGS) $(CXXFLAGS_CUDA) -x cu -c $< -o $@ 
else
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_CUDA) -c $< -o $@
endif

-include $(DEPS)

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS) sbench

.PHONY: format
format:
	clang-format -i `find . -regex '.*\.\(cpp\|h\|cu\)'`
