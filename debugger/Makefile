.PHONY: all clean cuda-gdb

all: buggy

buggy: buggy.cu
	nvcc -g -G -o buggy buggy.cu

cuda-gdb: buggy
	cuda-gdb ./buggy

clean:
	rm -f buggy
