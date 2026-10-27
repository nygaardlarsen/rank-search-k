//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub/cub.cuh"
#include "helper.cu.h"

template<class Z>
bool validateZ(Z* A, uint32_t sizeAB) {
    for(uint32_t i = 1; i < sizeAB; i++)
      if (A[i-1] > A[i]){
        printf("INVALID RESULT for i:%d, (A[i-1]=%d > A[i]=%d)\n", i, A[i-1], A[i]);
        return false;
      }
    return true;
}

bool validateK(const uint32_t* original, uint64_t N, uint64_t k, uint32_t kth) {
    uint64_t less = 0;
    uint64_t equal = 0;

    for (uint64_t i = 0; i < N; i++) {
        less += original[i] < kth;
        equal += original[i] == kth;
    }

    return less < k && k <= less + equal;
}

void randomInitNat(uint32_t* data, const uint32_t size, const uint32_t H) {
    for (int i = 0; i < size; ++i) {
        unsigned long int r = rand();
        data[i] = r % H;
    }
}

__global__ void selectK(const uint32_t* sorted, uint32_t* result, uint64_t k) {
    result[0] = sorted[k - 1];
}

double rankSearchCUB(uint32_t* data_keys_in, uint32_t* data_keys_out, uint32_t* result, uint64_t N, uint64_t k) {
    int beg_bit = 0;
    int end_bit = 32;

    void * tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    { // sort prelude
        cub::DeviceRadixSort::SortKeys( tmp_sort_mem, tmp_sort_len
                                      , data_keys_in, data_keys_out
                                      , N,   beg_bit,  end_bit
                                      );
        cudaMalloc(&tmp_sort_mem, tmp_sort_len);
    }
    cudaCheckError();

    { // one dry run
        cub::DeviceRadixSort::SortKeys(tmp_sort_mem, tmp_sort_len, data_keys_in, data_keys_out, N, beg_bit, end_bit);
        selectK<<<1, 1>>>(data_keys_out, result, k);
        cudaDeviceSynchronize();
    }
    cudaCheckError();

    // timing
    cudaEvent_t start, stop;
    cudaSucceeded(cudaEventCreate(&start));
    cudaSucceeded(cudaEventCreate(&stop));

    cudaSucceeded(cudaEventRecord(start));

    for (int run = 0; run < GPU_RUNS; run++) {
        cudaSucceeded(cub::DeviceRadixSort::SortKeys(tmp_sort_mem, tmp_sort_len, data_keys_in, data_keys_out, N, beg_bit, end_bit));
        selectK<<<1, 1>>>(data_keys_out, result, k);
    }

    cudaSucceeded(cudaEventRecord(stop));
    cudaSucceeded(cudaEventSynchronize(stop));
    cudaCheckError();

    float milliseconds;
    cudaSucceeded(cudaEventElapsedTime(&milliseconds, start, stop));
    double elapsed = milliseconds * 1000.0 / GPU_RUNS;

    cudaSucceeded(cudaEventDestroy(start));
    cudaSucceeded(cudaEventDestroy(stop));

    cudaFree(tmp_sort_mem);

    return elapsed;
}


int main (int argc, char * argv[]) {
    if (argc != 3) {
        printf("Usage: %s <size-of-array> <k>\n", argv[0]);
        exit(1);
    }

    const uint64_t N = strtoull(argv[1], NULL, 10);
    const uint64_t k = strtoull(argv[2], NULL, 10);

    if (N == 0 || N > UINT32_MAX || k < 1 || k > N || N < 10) {
        printf("Invalid N or k\n");
        exit(1);
    }

    //Allocate and Initialize Host data with random values
    uint32_t* h_keys  = (uint32_t*) malloc(N*sizeof(uint32_t));
    randomInitNat(h_keys, N, N/10);

    //Allocate and Initialize Device data
    uint32_t* d_keys_in;
    uint32_t* d_keys_out;
    uint32_t* d_result;
    cudaSucceeded(cudaMalloc((void**) &d_keys_in,  N * sizeof(uint32_t)));
    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMalloc((void**) &d_keys_out, N * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**) &d_result, sizeof(uint32_t)));

    double elapsed = rankSearchCUB(d_keys_in, d_keys_out, d_result, N, k);

    uint32_t kth;
    cudaSucceeded(cudaMemcpy(&kth, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaCheckError();

   bool success = validateK(h_keys, N, k, kth);

    printf("CUB Rank-k: N=%lu, k=%lu, result=%u\n", (unsigned long)N, (unsigned long)k, kth);
    printf("Sorting + selection time: %.2f us, VALID: %d\n", elapsed, success);

    // Cleanup and closing
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_result);
    free(h_keys);

    return success ? 0 : 1;
}
