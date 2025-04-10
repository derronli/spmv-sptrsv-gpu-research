#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Define warp_reduce function (assuming it's implemented elsewhere)
template <typename data_type>
__device__ data_type warp_reduce(data_type val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <typename data_type>
__global__ void csr_spmv_vector_kernel (
    unsigned int n_rows,
    const unsigned int *col_ids,
    const unsigned int *row_ptr,
    const data_type*data,
    const data_type*x,
    data_type*y)
{
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int lane = thread_id % 32;

  const unsigned int row = warp_id; ///< One warp per row

  data_type dot = 0;
  if (row < n_rows)
  {
    const unsigned int row_start = row_ptr[row];
    const unsigned int row_end = row_ptr[row + 1];

    for (unsigned int element = row_start + lane; element < row_end; element += 32)
      dot += data[element] * x[col_ids[element]];
  }

  dot = warp_reduce (dot);

  if (lane == 0 && row < n_rows)
  {
    y[row] = dot;
  }
}

// Host code
template <typename data_type>
void csr_spmv_vector_host(
    unsigned int n_rows,
    const std::vector<unsigned int>& h_col_ids,
    const std::vector<unsigned int>& h_row_ptr,
    const std::vector<data_type>& h_data,
    const std::vector<data_type>& h_x,
    std::vector<data_type>& h_y)
{
    // Allocate device memory
    unsigned int* d_col_ids;
    unsigned int* d_row_ptr;
    data_type* d_data;
    data_type* d_x;
    data_type* d_y;

    cudaMalloc(&d_col_ids, h_col_ids.size() * sizeof(unsigned int));
    cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(unsigned int));
    cudaMalloc(&d_data, h_data.size() * sizeof(data_type));
    cudaMalloc(&d_x, h_x.size() * sizeof(data_type));
    cudaMalloc(&d_y, h_y.size() * sizeof(data_type));

    // Copy data from host to device
    cudaMemcpy(d_col_ids, h_col_ids.data(), h_col_ids.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(data_type), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(data_type), cudaMemcpyHostToDevice);

    // Define kernel launch parameters
    const unsigned int threads_per_block = 256; // Number of threads per block
    const unsigned int warps_per_block = threads_per_block / 32; // Number of warps per block
    const unsigned int num_blocks = (n_rows + warps_per_block - 1) / warps_per_block;

    // Launch the kernel
    csr_spmv_vector_kernel<data_type><<<num_blocks, threads_per_block>>>(
        n_rows, d_col_ids, d_row_ptr, d_data, d_x, d_y);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copy results back to host
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(data_type), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_col_ids);
    cudaFree(d_row_ptr);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);
}

int main() {
    // Example input data
    unsigned int n_rows = 4;
    std::vector<unsigned int> h_col_ids = {0, 1, 2, 0, 2, 1, 3};
    std::vector<unsigned int> h_row_ptr = {0, 3, 5, 6, 7};
    std::vector<float> h_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    std::vector<float> h_x = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_y(n_rows, 0.0f);

    // Call the host function
    csr_spmv_vector_host<float>(n_rows, h_col_ids, h_row_ptr, h_data, h_x, h_y);

    // Print the result
    std::cout << "Result vector y:" << std::endl;
    for (const auto& val : h_y) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
