.PHONY: all clean

all: driver_api_demo

driver_api_demo: square_kernel.ptx driver_api_demo.o
	g++ driver_api_demo.o -o driver_api_demo -lcuda

driver_api_demo.o: driver_api_demo.cpp
	nvcc -c driver_api_demo.cpp -o driver_api_demo.o -lcuda

square_kernel.ptx: square_kernel.cu
	nvcc -ptx square_kernel.cu -o square_kernel.ptx

clean:
	rm -f square_kernel.ptx driver_api_demo.o driver_api_demo
