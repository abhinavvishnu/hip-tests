#include <stdio.h>
#include "hip/hip_runtime.h"
#include <chrono>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(hipLaunchParm lp, T* C_d, const T* A_d, const T* B_d, size_t N) {
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = A_d[i] + B_d[i];
    }
}


int main(int argc, char* argv[]) {
    int ndev = 0;
    //! Find number of devices system contains
    hipGetDeviceCount(&ndev);

    //! Force to use no more than 4 gpus
    ndev = std::min(ndev, 4);
    //! Store device indices
    std::vector<int> device_list(ndev);
    for (int i = 0; i < ndev; i++) {
        device_list[i] = i;
    }

    std::vector<int*> source_host_buffers(ndev), dest_host_buffers(ndev),
        source_device_buffers(ndev), dest_device_buffers(ndev);
    //! multi-gpu streams
    std::vector<hipStream_t> streams(ndev);

    //! prepare data
    for (int i = 0; i < ndev; i++) {
        //! switch to device i
        hipSetDevice(i);

        //! create stream on device i
        hipStreamCreate(&streams[i]);

        //! allocate source host memory corresponding to current gpu
        source_host_buffers[i] = new int[len];

        //! fill source host buffer with integer value 1
        std::fill(source_host_buffers[i], source_host_buffers[i] + len, 1);

        //! allocate destination host memory corresponding to current gpu
        dest_host_buffers[i] = new int[len];

        //! fill destination host buffer with integer value 0
        std::fill(dest_host_buffers[i], dest_host_buffers[i] + len, 0);

        //! allocate source and destination buffers on current gpu
        hipMalloc(&source_device_buffers[i], size);
        hipMalloc(&dest_device_buffers[i], size);

        //! copy host data for source and destination to device buffers
        hipMemcpy(source_device_buffers[i], source_host_buffers[i], size,
                  hipMemcpyHostToDevice);
        hipMemcpy(dest_device_buffers[i], dest_host_buffers[i], size,
                  hipMemcpyHostToDevice);
    }

}
