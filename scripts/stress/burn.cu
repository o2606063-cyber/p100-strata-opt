// Full-GPU burn: one thread per device, FMA storm for N seconds.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

__global__ void burn(float* buf, int n, long long iters) {
    float acc = buf[threadIdx.x & 255];
    const int stride = blockDim.x * gridDim.x;
    for (long long it = 0; it < iters; ++it) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            acc = fmaf(acc, 1.0000001f, buf[i]);
            buf[i] = acc;
        }
    }
    buf[threadIdx.x & 255] = acc;
}

static void run(int dev, int n, int secs) {
    cudaSetDevice(dev);
    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, dev);
    float* d = nullptr;
    if (cudaMalloc(&d, (size_t)n * sizeof(float)) != cudaSuccess) {
        printf("dev %d: malloc failed\n", dev);
        return;
    }
    cudaMemset(d, 1, (size_t)n * sizeof(float));
    dim3 grid(p.multiProcessorCount * 8), block(256);
    auto end = std::chrono::steady_clock::now() + std::chrono::seconds(secs);
    long long rounds = 0;
    while (std::chrono::steady_clock::now() < end) {
        burn<<<grid, block>>>(d, n, 500);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            printf("dev %d: kernel error %s\n", dev, cudaGetErrorString(cudaGetLastError()));
            break;
        }
        ++rounds;
    }
    cudaFree(d);
    printf("dev %d (%s): %lld rounds\n", dev, p.name, rounds);
}

int main(int argc, char** argv) {
    int secs = argc > 1 ? atoi(argv[1]) : 600;
    int mb = argc > 2 ? atoi(argv[2]) : 1024;
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("burning %d GPUs for %d s with %d MB each\n", count, secs, mb);
    fflush(stdout);
    std::vector<std::thread> ts;
    for (int i = 0; i < count; ++i) ts.emplace_back(run, i, mb * 1024 * 1024 / 4, secs);
    for (auto& t : ts) t.join();
    printf("done\n");
    return 0;
}
