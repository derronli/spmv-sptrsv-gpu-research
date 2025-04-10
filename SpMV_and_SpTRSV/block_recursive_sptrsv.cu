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
#include <tuple>
#include <algorithm>
#include <cuda.h>

// Define a sparse matrix in CSR format
struct CSRMatrix {
    unsigned int rows;                // Number of rows
    unsigned int cols;                // Number of columns
    std::vector<unsigned int> rowPtr; // Row pointers
    std::vector<unsigned int> colIdx; // Column indices
    std::vector<float> values;        // Non-zero values
};

// Function prototypes
void spTRSV(const CSRMatrix& tri, const std::vector<float>& b, std::vector<float>& x);
void spMV(const CSRMatrix& mat, const std::vector<float>& x, std::vector<float>& b);
CSRMatrix extractSubMatrix(const CSRMatrix& mat, unsigned int startRow, unsigned int endRow, unsigned int startCol, unsigned int endCol);

// cuSPARSE implementation for SpTRSV
void cusparse_sptrsv(
    unsigned int n_rows,
    unsigned int nnz,
    const std::vector<unsigned int>& h_row_ptr,
    const std::vector<unsigned int>& h_col_ids,
    const std::vector<float>& h_data,
    const std::vector<float>& h_b,
    std::vector<float>& h_x,
    float& execution_time, float& throughput, float& bandwidth)
{
    // Allocate device memory
    unsigned int* d_row_ptr = nullptr;
    unsigned int* d_col_ids = nullptr;
    float* d_data = nullptr;
    float* d_b = nullptr;
    float* d_x = nullptr;

    cudaMalloc(&d_row_ptr, h_row_ptr.size() * sizeof(unsigned int));
    cudaMalloc(&d_col_ids, h_col_ids.size() * sizeof(unsigned int));
    cudaMalloc(&d_data, h_data.size() * sizeof(float));
    cudaMalloc(&d_b, h_b.size() * sizeof(float));
    cudaMalloc(&d_x, n_rows * sizeof(float)); // Ensure size matches n_rows

    // Copy data from host to device
    cudaMemcpy(d_row_ptr, h_row_ptr.data(), h_row_ptr.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ids, h_col_ids.data(), h_col_ids.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, n_rows * sizeof(float)); // Initialize solution vector to zero

    // cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecB, vecX;
    cusparseCreateCsr(&matA, n_rows, n_rows, nnz, d_row_ptr, d_col_ids, d_data,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecB, n_rows, d_b, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, n_rows, d_x, CUDA_R_32F);

    // SpSV descriptor
    cusparseSpSVDescr_t spsvDescr;
    cusparseSpSV_createDescr(&spsvDescr);

    // Buffer size and allocation
    size_t bufferSize;
    void* dBuffer = nullptr;
    float alpha = 1.0;
    cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB, vecX,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Analysis step
    cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB, vecX,
        CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr, dBuffer);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int num_experiments = 5;
    cudaEventRecord(start);
    for (int i = 0; i < num_experiments; ++i) {
        cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecB, vecX,
            CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, spsvDescr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= num_experiments; // Average time per kernel launch
    execution_time = milliseconds;

    // Copy result back to host
    cudaMemcpy(h_x.data(), d_x, n_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate performance metrics
    throughput = 2.0f * nnz / (milliseconds / 1000.0f); // FLOPS
    float memory_bytes = (h_row_ptr.size() + h_col_ids.size() + h_data.size() + h_b.size() + h_x.size()) * sizeof(float);
    bandwidth = memory_bytes / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s

    // Cleanup
    cusparseSpSV_destroyDescr(spsvDescr);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecB);
    cusparseDestroyDnVec(vecX);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ids);
    cudaFree(d_data);
    cudaFree(d_b);
    cudaFree(d_x);
}

// ========================================================== SpMV Implementation 
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
    const unsigned int threads_per_block = 256;
    const unsigned int warps_per_block = threads_per_block / 32; 
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

// ===================================================== SpMV Implementation  END

// Recursive block algorithm for SpTRSV
void spTRSVRecursiveBlock(const CSRMatrix& tri, std::vector<float>& b, std::vector<float>& x, int depth) {
    if (depth == 0) {
        // Base case: Perform standard sparse triangular solve
        spTRSV(tri, b, x);
        //cusparse_sptrsv(n_rows, data.size(), row_ptr, col_ids, data, h_b, h_x_cusparse,
        //    cusparse_time, cusparse_throughput, cusparse_bandwidth);

    }
    else {
        // Divide the matrix into top, bottom, and square blocks
        unsigned int mid = tri.rows / 2; // Divide the matrix at the midpoint

        // Top triangular block
        CSRMatrix triTop = extractSubMatrix(tri, 0, mid, 0, mid);

        // Square block
        CSRMatrix rec = extractSubMatrix(tri, mid, tri.rows, 0, mid);

        // Bottom triangular block
        CSRMatrix triBottom = extractSubMatrix(tri, mid, tri.rows, mid, tri.cols);

        // Solution vector for the top triangular block
        std::vector<float> xTop(mid, 0.0f);
        spTRSVRecursiveBlock(triTop, b, xTop, depth - 1);

        // Update the right-hand side vector using the square block
        std::vector<float> bBottom(b.begin() + mid, b.end());
        std::vector<float> temp(mid, 0.0f); // Temporary vector for SpMV result

        // Perform sparse matrix-vector multiplication using the custom SpMV implementation
        float spmv_execution_time, spmv_throughput, spmv_bandwidth;
        csr_spmv_vector_host<float>(
            rec.rows, rec.colIdx, rec.rowPtr, rec.values, xTop, temp,
            spmv_execution_time, spmv_throughput, spmv_bandwidth
        );

        // Update bBottom with the result of SpMV
        for (unsigned int i = 0; i < bBottom.size(); ++i) {
            bBottom[i] -= temp[i];
        }

        // Solution vector for the bottom triangular block
        std::vector<float> xBottom(tri.rows - mid, 0.0f);
        spTRSVRecursiveBlock(triBottom, bBottom, xBottom, depth - 1);

        // Combine the results
        std::copy(xTop.begin(), xTop.end(), x.begin());
        std::copy(xBottom.begin(), xBottom.end(), x.begin() + mid);
    }
}


// Fallback SpTRSV
void spTRSV(const CSRMatrix& tri, const std::vector<float>& b, std::vector<float>& x) {
    for (unsigned int i = 0; i < tri.rows; ++i) {
        float sum = b[i];
        for (unsigned int j = tri.rowPtr[i]; j < tri.rowPtr[i + 1]; ++j) {
            unsigned int col = tri.colIdx[j];
            sum -= tri.values[j] * x[col];
        }
        x[i] = sum / tri.values[tri.rowPtr[i + 1] - 1]; // Diagonal element
    }
}

// Extract a submatrix from a CSR matrix
CSRMatrix extractSubMatrix(const CSRMatrix& mat, unsigned int startRow, unsigned int endRow, unsigned int startCol, unsigned int endCol) {
    CSRMatrix subMat;
    subMat.rows = endRow - startRow;
    subMat.cols = endCol - startCol;

    subMat.rowPtr.resize(subMat.rows + 1, 0);
    std::vector<unsigned int> newColIdx;
    std::vector<float> newValues;

    for (unsigned int i = startRow; i < endRow; ++i) {
        unsigned int newRow = i - startRow;
        subMat.rowPtr[newRow] = newValues.size();
        for (unsigned int j = mat.rowPtr[i]; j < mat.rowPtr[i + 1]; ++j) {
            unsigned int col = mat.colIdx[j];
            if (col >= startCol && col < endCol) {
                newColIdx.push_back(col - startCol);
                newValues.push_back(mat.values[j]);
            }
        }
    }
    subMat.rowPtr[subMat.rows] = newValues.size();
    subMat.colIdx = newColIdx;
    subMat.values = newValues;

    return subMat;
}

// Host function to measure performance for recursive SpTRSV
void spTRSVRecursiveBlockHost(
    unsigned int n_rows,
    const std::vector<unsigned int>& h_row_ptr,
    const std::vector<unsigned int>& h_col_ids,
    const std::vector<float>& h_data,
    std::vector<float>& h_b,
    std::vector<float>& h_x,
    float& execution_time, float& throughput, float& bandwidth)
{
    // Convert input data to CSRMatrix format
    CSRMatrix tri = { n_rows, n_rows, h_row_ptr, h_col_ids, h_data };

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the recursive block algorithm
    const int depth = 2; // Hardcoded recursion depth

    // Determine recursion depth dynamically
    //const int min_rows = 128; // Minimum rows for base case
    //int depth = static_cast<int>(std::log2(n_rows / min_rows));
    //depth = std::max(0, depth); // Ensure depth is non-negative

    //std::cout << "Dynamic recursion depth: " << depth << std::endl;

    cudaEventRecord(start);
    spTRSVRecursiveBlock(tri, h_b, h_x, depth);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    execution_time = milliseconds;

    // Calculate performance metrics
    unsigned int nnz = h_data.size(); // Number of non-zero elements
    throughput = 2.0f * nnz / (milliseconds / 1000.0f); // FLOPS (2 operations per non-zero element)
    float memory_bytes = (h_row_ptr.size() + h_col_ids.size() + h_data.size() + h_b.size() + h_x.size()) * sizeof(float);
    bandwidth = memory_bytes / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s
}

template <typename data_type>
bool compare_results(unsigned int y_size, const data_type* a, const data_type* b) {
    data_type numerator = 0.0;
    data_type denominator = 0.0;

    // Print all elements of a
    std::cerr << "Contents of array a:" << std::endl;
    for (unsigned int j = 0; j < 10; j++) {
        std::cerr << "a[" << j << "] = " << a[j] << std::endl;
    }

    // Print all elements of b
    std::cerr << "Contents of array b:" << std::endl;
    for (unsigned int j = 0; j < 10; j++) {
        std::cerr << "b[" << j << "] = " << b[j] << std::endl;
    }

    for (unsigned int i = 0; i < y_size; i++) {
        numerator += (a[i] - b[i]) * (a[i] - b[i]);
        denominator += b[i] * b[i];
    }

    const data_type error = numerator / denominator;

    if (error > 1e-3) {
        std::cerr << "ERROR: Mean Squared Error = " << error << std::endl;

        for (unsigned int i = 0; i < y_size; i++) {
            if (std::abs(a[i] - b[i]) > 1e-8) {
                std::cerr << "Mismatch at index " << i << ": a[" << i << "] = " << a[i] << ", b[" << i << "] = " << b[i] << std::endl;

                break;
            }
        }

        std::cerr.flush();
        return false;
    }

    std::cout << "Results match! Mean Squared Error = " << error << std::endl;
    return true;
}

// Function to read a matrix in Matrix Market format and convert it to CSR format
void read_matrix_market(const std::string& filename,
    std::vector<unsigned int>& row_ptr,
    std::vector<unsigned int>& col_ids,
    std::vector<float>& data,
    unsigned int& n_rows,
    unsigned int& n_cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    bool is_symmetric = false;

    // Skip comments and check for symmetry flag
    while (std::getline(file, line)) {
        if (line[0] == '%') {
            // Check if the matrix is symmetric
            if (line.find("symmetric") != std::string::npos) {
                is_symmetric = true;
            }
            continue;
        }
        break;
    }

    // Read matrix dimensions and number of non-zero entries
    int rows, cols, nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    n_rows = rows;
    n_cols = cols;

    // Temporary storage for COO format
    std::vector<std::tuple<int, int, float>> coo_entries;
    coo_entries.reserve(nnz);

    int row, col;
    float value;
    while (file >> row >> col >> value) {
        // Convert 1-based indexing to 0-based indexing
        row -= 1;
        col -= 1;

        // For symmetric matrices, we need to consider both entries
        if (is_symmetric) {
            // If it's in the lower triangular part, add it directly
            if (row >= col) {
                coo_entries.push_back(std::make_tuple(row, col, value));
            }
            // If it's in the upper triangular part, add its symmetric counterpart
            else {
                coo_entries.push_back(std::make_tuple(col, row, value));
            }
        }
        // For non-symmetric matrices, only keep lower triangular entries
        else if (row >= col) {
            coo_entries.push_back(std::make_tuple(row, col, value));
        }
    }

    // Sort entries by row, then by column for CSR format
    std::sort(coo_entries.begin(), coo_entries.end());

    // Convert sorted COO to CSR format
    row_ptr.resize(rows + 1, 0);
    col_ids.clear();
    data.clear();

    col_ids.reserve(coo_entries.size());
    data.reserve(coo_entries.size());

    int current_row = -1;
    for (const auto& entry : coo_entries) {
        int r = std::get<0>(entry);
        int c = std::get<1>(entry);
        float v = std::get<2>(entry);

        // Fill in empty rows
        while (current_row < r) {
            current_row++;
            row_ptr[current_row] = col_ids.size();
        }

        col_ids.push_back(c);
        data.push_back(v);
    }

    // Fill in remaining row pointers
    while (current_row < rows) {
        current_row++;
        row_ptr[current_row] = col_ids.size();
    }
}

int main() {
    std::string filename = "matrices/tmt_sym.mtx"; // CHANGE THIS DEPENDING ON WHAT YOU WANT YOUR INPUT TO BE

    // Vectors to store CSR format data
    std::vector<unsigned int> row_ptr, col_ids;
    std::vector<float> data;
    unsigned int n_rows, n_cols;

    // Read the matrix from the file and convert it to CSR format
    try {
        read_matrix_market(filename, row_ptr, col_ids, data, n_rows, n_cols);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Right-hand side vector
    std::vector<float> h_b(n_cols);
    std::vector<float> h_x_recursive(n_rows, 0.0f); // Result from recursive block implementation
    std::vector<float> h_x_cusparse(n_rows, 0.0f); // Result from cuSPARSE implementation
    srand(static_cast<unsigned int>(time(NULL)));
    for (unsigned int i = 0; i < n_cols; ++i) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1
    }


    // Performance metrics
    float recursive_time, recursive_throughput, recursive_bandwidth;
    float cusparse_time, cusparse_throughput, cusparse_bandwidth;

    // Perform sparse triangular solve using recursive block implementation
    spTRSVRecursiveBlockHost(n_rows, row_ptr, col_ids, data, h_b, h_x_recursive,
        recursive_time, recursive_throughput, recursive_bandwidth);

    // Perform sparse triangular solve using cuSPARSE
    cusparse_sptrsv(n_rows, data.size(), row_ptr, col_ids, data, h_b, h_x_cusparse,
        cusparse_time, cusparse_throughput, cusparse_bandwidth);

    // Compare results
    bool results_match = compare_results(n_rows, h_x_recursive.data(), h_x_cusparse.data());

    if (results_match) {
        std::cout << "Results match!" << std::endl;
        std::cout << "Recursive Block Implementation Performance:" << std::endl;
        std::cout << "Execution time (ms): " << recursive_time << std::endl;
        std::cout << "Throughput (GFLOPS): " << recursive_throughput / 1e9 << std::endl;
        std::cout << "Memory bandwidth (GB/s): " << recursive_bandwidth << std::endl;

        std::cout << "cuSPARSE Implementation Performance:" << std::endl;
        std::cout << "Execution time (ms): " << cusparse_time << std::endl;
        std::cout << "Throughput (GFLOPS): " << cusparse_throughput / 1e9 << std::endl;
        std::cout << "Memory bandwidth (GB/s): " << cusparse_bandwidth << std::endl;
    }
    else {
        std::cout << "Results do not match!" << std::endl;
    }

    return 0;
}
