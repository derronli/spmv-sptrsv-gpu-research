#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

// Define warp_reduce function
template <typename data_type>
__device__ data_type warp_reduce(data_type val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <typename data_type>
__global__ void csr_spmv_vector_kernel(
    unsigned int n_rows,
    const unsigned int* col_ids,
    const unsigned int* row_ptr,
    const data_type* data,
    const data_type* x,
    data_type* y)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int warp_id = thread_id / 32;
    const unsigned int lane = thread_id % 32;

    const unsigned int row = warp_id; ///< One warp per row

    data_type dot = 0;
    if (row < n_rows) {
        const unsigned int row_start = row_ptr[row];
        const unsigned int row_end = row_ptr[row + 1];

        for (unsigned int element = row_start + lane; element < row_end; element += 32)
            dot += data[element] * x[col_ids[element]];
    }

    dot = warp_reduce(dot);

    if (lane == 0 && row < n_rows) {
        y[row] = dot;
    }
}

// Host function to measure performance
template <typename data_type>
void csr_spmv_vector_host(
    unsigned int n_rows,
    const std::vector<unsigned int>& h_col_ids,
    const std::vector<unsigned int>& h_row_ptr,
    const std::vector<data_type>& h_data,
    const std::vector<data_type>& h_x,
    std::vector<data_type>& h_y,
    float& execution_time, float& throughput, float& bandwidth)
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

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel multiple times for averaging
    const int num_experiments = 5;
    cudaEventRecord(start);
    for (int i = 0; i < num_experiments; ++i) {
        csr_spmv_vector_kernel<data_type> << <num_blocks, threads_per_block >> > (
            n_rows, d_col_ids, d_row_ptr, d_data, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= num_experiments; // Average time per kernel launch
    execution_time = milliseconds;

    // Copy results back to host
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(data_type), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_col_ids);
    cudaFree(d_row_ptr);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);

    // Calculate performance metrics
    unsigned int nnz = h_data.size(); // Number of non-zero elements
    throughput = 2.0f * nnz / (milliseconds / 1000.0f); // FLOPS (2 operations per non-zero element)
    float memory_bytes = (h_col_ids.size() + h_row_ptr.size() + h_data.size() + h_x.size() + h_y.size()) * sizeof(data_type);
    bandwidth = memory_bytes / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s
}

// cuSPARSE implementation
void cusparse_spmv(
    unsigned int n_rows,
    unsigned int n_cols,
    unsigned int nnz,
    const std::vector<unsigned int>& h_row_ptr,
    const std::vector<unsigned int>& h_col_ids,
    const std::vector<float>& h_data,
    const std::vector<float>& h_x,
    std::vector<float>& h_y,
    float& execution_time, float& throughput, float& bandwidth)
{
    // Allocate device memory
    unsigned int* d_row_ptr;
    unsigned int* d_col_ids;
    float* d_data;
    float* d_x;
    float* d_y;

    cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(unsigned int));
    cudaMalloc(&d_col_ids, h_col_ids.size() * sizeof(unsigned int));
    cudaMalloc(&d_data, h_data.size() * sizeof(float));
    cudaMalloc(&d_x, h_x.size() * sizeof(float));
    cudaMalloc(&d_y, h_y.size() * sizeof(float));

    cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ids, h_col_ids.data(), h_col_ids.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);

    // cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&matA, n_rows, n_cols, nnz, d_row_ptr, d_col_ids, d_data,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, n_cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n_rows, d_y, CUDA_R_32F);

    // Buffer size and allocation
    size_t bufferSize;
    void* dBuffer;
    float alpha = 1.0;
    float beta = 0.0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_experiments = 5;
    cudaEventRecord(start);
    for (int i = 0; i < num_experiments; ++i) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= num_experiments; // Average time per kernel launch
    execution_time = milliseconds;

    // Copy result back to host
    cudaMemcpy(h_y.data(), d_y, h_y.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate performance metrics
    throughput = 2.0f * nnz / (milliseconds / 1000.0f); // FLOPS
    float memory_bytes = (h_row_ptr.size() + h_col_ids.size() + h_data.size() + h_x.size() + h_y.size()) * sizeof(float);
    bandwidth = memory_bytes / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ids);
    cudaFree(d_data);
    cudaFree(d_x);
    cudaFree(d_y);
}

void read_matrix_market(const std::string& filename,
    std::vector<unsigned int>& row_ptr,
    std::vector<unsigned int>& col_ids,
    std::vector<float>& data,
    unsigned int& n_rows,
    unsigned int& n_cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::string line;
    // Skip comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix dimensions
    std::istringstream iss(line);
    unsigned int nnz;
    iss >> n_rows >> n_cols >> nnz;

    std::vector<unsigned int> row_indices(nnz);
    std::vector<unsigned int> col_indices(nnz);
    std::vector<float> values(nnz);

    // Read COO data
    for (unsigned int i = 0; i < nnz; ++i) {
        file >> row_indices[i] >> col_indices[i] >> values[i];
        row_indices[i]--; // Convert to 0-based indexing
        col_indices[i]--; // Convert to 0-based indexing
    }

    // Convert COO to CSR
    row_ptr.resize(n_rows + 1, 0);
    col_ids.resize(nnz);
    data.resize(nnz);

    for (unsigned int i = 0; i < nnz; ++i) {
        row_ptr[row_indices[i] + 1]++;
    }

    for (unsigned int i = 1; i <= n_rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }

    std::vector<unsigned int> temp_row_ptr = row_ptr;
    for (unsigned int i = 0; i < nnz; ++i) {
        unsigned int row = row_indices[i];
        unsigned int dest = temp_row_ptr[row]++;
        col_ids[dest] = col_indices[i];
        data[dest] = values[i];
    }
}

template <typename data_type>
bool compare_results(unsigned int y_size, const data_type* a, const data_type* b) {
    data_type numerator = 0.0;
    data_type denominator = 0.0;

    for (unsigned int i = 0; i < y_size; i++) {
        numerator += (a[i] - b[i]) * (a[i] - b[i]);
        denominator += b[i] * b[i];
    }

    const data_type error = numerator / denominator;

    if (error > 1e-9) {
        std::cerr << "ERROR: Mean Squared Error = " << error << std::endl;

        for (unsigned int i = 0; i < y_size; i++) {
            if (std::abs(a[i] - b[i]) > 1e-8) {
                std::cerr << "Mismatch at index " << i << ": a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << std::endl;
                break; // Print the first mismatch and exit
            }
        }

        std::cerr.flush();
        return false; // Results do not match
    }

    std::cout << "Results match! Mean Squared Error = " << error << std::endl;
    return true; // Results match
}


int main() {
    std::string filename = "matrices/ML_Laplace.mtx"; // NEED TO MANUALLY CHANGE PATH TO RUN THE CODE
    std::vector<unsigned int> row_ptr, col_ids;
    std::vector<float> data;
    unsigned int n_rows, n_cols;

    // Read the matrix from the Matrix Market file and convert it to CSR format
    read_matrix_market(filename, row_ptr, col_ids, data, n_rows, n_cols);

    // Generate a randomized input vector h_x
    std::vector<float> h_x(n_cols);
    std::vector<float> h_y_manual(n_rows, 0.0f); // Result from manual implementation
    std::vector<float> h_y_cusparse(n_rows, 0.0f); // Result from cuSPARSE implementation
    srand(static_cast<unsigned int>(time(NULL))); // Seed the random number generator
    for (unsigned int i = 0; i < n_cols; ++i) {
        h_x[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1
    }

    // Performance metrics
    float manual_time, manual_throughput, manual_bandwidth;
    float cusparse_time, cusparse_throughput, cusparse_bandwidth;

    // Call the manual CSR SpMV host function
    csr_spmv_vector_host<float>(n_rows, col_ids, row_ptr, data, h_x, h_y_manual, manual_time, manual_throughput, manual_bandwidth);

    // Call the cuSPARSE SpMV function
    cusparse_spmv(n_rows, n_cols, data.size(), row_ptr, col_ids, data, h_x, h_y_cusparse, cusparse_time, cusparse_throughput, cusparse_bandwidth);

    bool results_match = compare_results(n_rows, h_y_manual.data(), h_y_cusparse.data());


    if (results_match) {
        std::cout << "Results match!" << std::endl;
        std::cout << "Manual Implementation Performance:" << std::endl;
        std::cout << "Execution time (ms): " << manual_time << std::endl;
        std::cout << "Throughput (GFLOPS): " << manual_throughput / 1e9 << std::endl;
        std::cout << "Memory bandwidth (GB/s): " << manual_bandwidth << std::endl;

        std::cout << "cuSPARSE Implementation Performance:" << std::endl;
        std::cout << "Execution time (ms): " << cusparse_time << std::endl;
        std::cout << "Throughput (GFLOPS): " << cusparse_throughput / 1e9 << std::endl;
        std::cout << "Memory bandwidth (GB/s): " << cusparse_bandwidth << std::endl;
    }

    return 0;
}
