all: matmul

matmul: main.cu
	nvcc -O2 -o matmul main.cu

clean:
	rm -f matmul
