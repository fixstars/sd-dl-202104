all: matmul

matmul: main.cu
	nvcc -O2 -o matmul -lcublas main.cu

clean:
	rm -f matmul
