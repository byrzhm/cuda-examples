.PHONY: all clean

all: 
	nvcc -o demo_default demo.cu
	nvcc -arch=sm_89 -o demo_89 demo.cu

clean:
	rm -f demo_default demo_89


