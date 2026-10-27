//#include "../../cub-1.8.0/cub/cub.cuh"   // or equivalently <cub/device/device_histogram.cuh>
#include "cub/cub.cuh"
#include "helper.cu.h"

#include <vector>
#include <string>
#include <fstream>
#include <climits>
#include <cstdint>

bool validateK(const uint32_t* original, uint64_t N, uint64_t k, uint32_t kth) {
    uint64_t less = 0;
    uint64_t equal = 0;

    for (uint64_t i = 0; i < N; i++) {
        less += original[i] < kth;
        equal += original[i] == kth;
    }

    return less < k && k <= less + equal;
}

__global__ void selectKBatch(const uint32_t* sorted, const int* offsets, const int* ks, uint32_t* result, int num_segments) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_segments) {
        result[i] = sorted[offsets[i] + ks[i] - 1];
    }
}


double rankSearchCUBBatch(uint32_t* data_keys_in, uint32_t* data_keys_out, uint32_t* result, int N, int num_segments, int* offsets, int* ks) {
    void* tmp_sort_mem = NULL;
    size_t tmp_sort_len = 0;

    cudaSucceeded(cub::DeviceSegmentedRadixSort::SortKeys(tmp_sort_mem, tmp_sort_len, data_keys_in, data_keys_out, N, num_segments, offsets, offsets + 1));
    cudaSucceeded(cudaMalloc(&tmp_sort_mem, tmp_sort_len));

    int threads = 256;
    int blocks = (num_segments + threads - 1) / threads;

    cudaSucceeded(cub::DeviceSegmentedRadixSort::SortKeys(tmp_sort_mem, tmp_sort_len, data_keys_in, data_keys_out, N, num_segments, offsets, offsets + 1));
    selectKBatch<<<blocks, threads>>>(data_keys_out, offsets, ks, result, num_segments);
    cudaSucceeded(cudaDeviceSynchronize());
    cudaCheckError();

    cudaEvent_t start, stop;
    cudaSucceeded(cudaEventCreate(&start));
    cudaSucceeded(cudaEventCreate(&stop));

    cudaSucceeded(cudaEventRecord(start));

    for (int run = 0; run < GPU_RUNS; run++) {
        cudaSucceeded(cub::DeviceSegmentedRadixSort::SortKeys(tmp_sort_mem, tmp_sort_len, data_keys_in, data_keys_out, N, num_segments, offsets, offsets + 1));
        selectKBatch<<<blocks, threads>>>(data_keys_out, offsets, ks, result, num_segments);
    }

    cudaSucceeded(cudaEventRecord(stop));
    cudaSucceeded(cudaEventSynchronize(stop));
    cudaCheckError();

    float milliseconds;
    cudaSucceeded(cudaEventElapsedTime(&milliseconds, start, stop));
    double elapsed = milliseconds * 1000.0 / GPU_RUNS;

    cudaSucceeded(cudaEventDestroy(start));
    cudaSucceeded(cudaEventDestroy(stop));
    cudaSucceeded(cudaFree(tmp_sort_mem));

    return elapsed;
}

template<typename T>
bool readBinary(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file) {
        printf("Cannot open %s\n", filename.c_str());
        return false;
    }

    std::streamoff bytes = file.tellg();

    if (bytes < 0 || bytes % sizeof(T) != 0) {
        printf("Invalid file size: %s\n", filename.c_str());
        return false;
    }

    data.resize(static_cast<size_t>(bytes) / sizeof(T));
    file.seekg(0);

    if (bytes > 0 && !file.read(reinterpret_cast<char*>(data.data()), bytes)) {
        printf("Cannot read %s\n", filename.c_str());
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <dataset-directory>\n", argv[0]);
        return 1;
    }

    std::string path = argv[1];

    std::vector<uint32_t> h_keys;
    std::vector<int> h_offsets;
    std::vector<int> h_ks;

    if (!readBinary(path + "/values.bin", h_keys) ||
        !readBinary(path + "/offsets.bin", h_offsets) ||
        !readBinary(path + "/ks.bin", h_ks)) {
        return 1;
    }

    if (h_keys.empty() || h_ks.empty() || h_offsets.size() != h_ks.size() + 1 ||
        h_keys.size() > INT_MAX || h_ks.size() > INT_MAX) {
        printf("Invalid dataset dimensions\n");
        return 1;
    }

    int N = static_cast<int>(h_keys.size());
    int num_segments = static_cast<int>(h_ks.size());

    if (h_offsets[0] != 0 || h_offsets[num_segments] != N) {
        printf("Invalid offsets\n");
        return 1;
    }

    for (int i = 0; i < num_segments; i++) {
        int length = h_offsets[i + 1] - h_offsets[i];

        if (length <= 0 || h_ks[i] < 1 || h_ks[i] > length) {
            printf("Invalid segment or k at index %d\n", i);
            return 1;
        }
    }

    std::vector<uint32_t> h_result(num_segments);

    uint32_t *d_keys_in, *d_keys_out, *d_result;
    int *d_offsets, *d_ks;

    cudaSucceeded(cudaMalloc((void**)&d_keys_in, (size_t)N * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**)&d_keys_out, (size_t)N * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**)&d_result, (size_t)num_segments * sizeof(uint32_t)));
    cudaSucceeded(cudaMalloc((void**)&d_offsets, (size_t)(num_segments + 1) * sizeof(int)));
    cudaSucceeded(cudaMalloc((void**)&d_ks, (size_t)num_segments * sizeof(int)));

    cudaSucceeded(cudaMemcpy(d_keys_in, h_keys.data(), (size_t)N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMemcpy(d_offsets, h_offsets.data(), (size_t)(num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice));
    cudaSucceeded(cudaMemcpy(d_ks, h_ks.data(), (size_t)num_segments * sizeof(int), cudaMemcpyHostToDevice));

    double elapsed = rankSearchCUBBatch(d_keys_in, d_keys_out, d_result, N, num_segments, d_offsets, d_ks);

    cudaSucceeded(cudaMemcpy(h_result.data(), d_result, (size_t)num_segments * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    bool success = true;

    for (int i = 0; i < num_segments; i++) {
        int length = h_offsets[i + 1] - h_offsets[i];

        if (!validateK(h_keys.data() + h_offsets[i], length, h_ks[i], h_result[i])) {
            printf("Invalid result for segment %d\n", i);
            success = false;
        }
    }

    printf("CUB Batch Rank-k: segments=%d, total=%d\n", num_segments, N);
    printf("Sorting + selection time: %.2f us, VALID: %d\n", elapsed, success);

    cudaSucceeded(cudaFree(d_keys_in));
    cudaSucceeded(cudaFree(d_keys_out));
    cudaSucceeded(cudaFree(d_result));
    cudaSucceeded(cudaFree(d_offsets));
    cudaSucceeded(cudaFree(d_ks));

    return success ? 0 : 1;
}