CUB=cub-1.8.0

all: test-cub test-batch

test-cub: cub_baseline.cu helper.cu.h
	nvcc -I$(CUB) -o test-cub cub_baseline.cu

test-batch: cub_batch.cu helper.cu.h
	nvcc -std=c++11 -I$(CUB) -o test-batch cub_batch.cu

clean:
	rm -f test-cub test-batch