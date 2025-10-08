	CC:=mpicc
	CXX:=mpicxx
#	OMPI_MPICC=nextcc
#	OMPI_MPICXX=nextcxx
	TFLAGS:= -march=sapphirerapids
	CFLAGS:=-O3 -fopenmp $(TFLAGS) $(DEFINES) -DUSE_NS_API -DTELEM_REGION
	LDFLAGS:=-lnsapi
	INCLUDES:=config.h iterators.h stencil_2.h stencil_7.h stencil_13.h stencil_25.h stencils.h aux.h grid.h
	SOURCES:=main.cpp iterators.cpp gilbert2d.cpp aux.cpp grid.cpp
	OBJECTS:=$(SOURCES:%.cpp=%.o)

all: stencil3d

$(OBJECTS): $(INCLUDES)

$(OBJECTS):%.o:%.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

stencil3d: $(OBJECTS)
	$(CXX) $(CFLAGS) -o $@ $(OBJECTS) $(LDFLAGS)

clean:
	rm stencil3d $(OBJECTS)
