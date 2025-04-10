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
#include <device_launch_parameters.h>


// Define a sparse matrix in CSR format
struct CSRMatrix {
    int rows;                // Number of rows
    int cols;                // Number of columns
    int nnz;                 // Number of non-zero elements
    int* d_rowPtr;           // Row pointers (device memory)
    int* d_colIdx;           // Column indices (device memory)
    float* d_values;         // Non-zero values (device memory)
};

// Function prototypes
void spTRSVRecursiveBlock(const CSRMatrix& tri, float* d_b, float* d_x, int depth);
CSRMatrix extractSubMatrix(const CSRMatrix& mat, int startRow, int endRow, int startCol, int endCol);
void allocateCSRMatrix(CSRMatrix& mat, int rows, int cols, int nnz);
void freeCSRMatrix(CSRMatrix& mat);
__global__ void spTRSVKernel(const int* rowPtr, const int* colIdx, const float* values, const float* b, float* x, int rows);
__global__ void spMVKernel(const int* rowPtr, const int* colIdx, const float* values, const float* x, float* b, int rows);

// Host function to measure performance for recursive block SpTRSV
void sptrsv_recursive_block_host(
    unsigned int n_rows,
    const std::vector<unsigned int>& h_row_ptr,
    const std::vector<unsigned int>& h_col_ids,
    const std::vector<float>& h_data,
    const std::vector<float>& h_b,
    std::vector<float>& h_x,
    float& execution_time, float& throughput, float& bandwidth)
{
    // Allocate device memory for the matrix
    CSRMatrix tri;
    allocateCSRMatrix(tri, n_rows, n_rows, h_data.size());

    // Copy data to device
    cudaMemcpy(tri.d_rowPtr, h_row_ptr.data(), (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tri.d_colIdx, h_col_ids.data(), h_col_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tri.d_values, h_data.data(), h_data.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate device memory for vectors
    float* d_b, * d_x;
    cudaMalloc((void**)&d_b, n_rows * sizeof(float));
    cudaMalloc((void**)&d_x, n_rows * sizeof(float));
    cudaMemcpy(d_b, h_b.data(), n_rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_x, 0, n_rows * sizeof(float));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Recursion depth (hardcoded to 2)
    int depth = 2;

    // Launch the recursive block SpTRSV
    const int num_experiments = 5;
    cudaEventRecord(start);
    for (int i = 0; i < num_experiments; ++i) {
        spTRSVRecursiveBlock(tri, d_b, d_x, depth);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Measure elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    milliseconds /= num_experiments; // Average time per kernel launch
    execution_time = milliseconds;

    // Copy result back to host
    cudaMemcpy(h_x.data(), d_x, n_rows * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    freeCSRMatrix(tri);
    cudaFree(d_b);
    cudaFree(d_x);

    // Calculate performance metrics
    unsigned int nnz = h_data.size(); // Number of non-zero elements
    throughput = 2.0f * nnz / (milliseconds / 1000.0f); // FLOPS (2 operations per non-zero element)
    float memory_bytes = (h_row_ptr.size() + h_col_ids.size() + h_data.size() + h_b.size() + h_x.size()) * sizeof(float);
    bandwidth = memory_bytes / (milliseconds / 1000.0f) / (1024.0f * 1024.0f * 1024.0f); // GB/s
}

// CUDA kernel for sparse triangular solve (SpTRSV)
__global__ void spTRSVKernel(const int* rowPtr, const int* colIdx, const float* values, const float* b, float* x, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = b[row];
        for (int j = rowPtr[row]; j < rowPtr[row + 1] - 1; ++j) {
            sum -= values[j] * x[colIdx[j]];
        }
        x[row] = sum / values[rowPtr[row + 1] - 1]; // Diagonal element
    }
}

// CUDA kernel for sparse matrix-vector multiplication (SpMV)
__global__ void spMVKernel(const int* rowPtr, const int* colIdx, const float* values, const float* x, float* b, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        float sum = 0.0f;
        for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
            sum += values[j] * x[colIdx[j]];
        }
        b[row] -= sum;
    }
}

// Recursive block algorithm for SpTRSV
void spTRSVRecursiveBlock(const CSRMatrix& tri, float* d_b, float* d_x, int depth) {
    if (depth == 0) {
        // Base case: Perform standard sparse triangular solve
        int numThreads = 256;
        int numBlocks = (tri.rows + numThreads - 1) / numThreads;
        spTRSVKernel<<<numBlocks, numThreads>>> (tri.d_rowPtr, tri.d_colIdx, tri.d_values, d_b, d_x, tri.rows);
        cudaDeviceSynchronize();
    }
    else {
        // Divide the matrix into top, bottom, and square blocks
        int mid = tri.rows / 2;

        // Extract submatrices
        CSRMatrix triTop = extractSubMatrix(tri, 0, mid, 0, mid);
        CSRMatrix rec = extractSubMatrix(tri, mid, tri.rows, 0, mid);
        CSRMatrix triBottom = extractSubMatrix(tri, mid, tri.rows, mid, tri.cols);

        // Allocate temporary vectors
        float* d_xTop, * d_bBottom;
        cudaMalloc((void**)&d_xTop, mid * sizeof(float));
        cudaMalloc((void**)&d_bBottom, (tri.rows - mid) * sizeof(float));
        cudaMemset(d_xTop, 0, mid * sizeof(float));
        cudaMemcpy(d_bBottom, d_b + mid, (tri.rows - mid) * sizeof(float), cudaMemcpyDeviceToDevice);

        // Solve top triangular block
        spTRSVRecursiveBlock(triTop, d_b, d_xTop, depth - 1);

        // Update the right-hand side vector using the square block
        int numThreads = 256;
        int numBlocks = (rec.rows + numThreads - 1) / numThreads;
        spMVKernel << <numBlocks, numThreads >> > (rec.d_rowPtr, rec.d_colIdx, rec.d_values, d_xTop, d_bBottom, rec.rows);
        cudaDeviceSynchronize();

        // Solve bottom triangular block
        float* d_xBottom;
        cudaMalloc((void**)&d_xBottom, (tri.rows - mid) * sizeof(float));
        cudaMemset(d_xBottom, 0, (tri.rows - mid) * sizeof(float));
        spTRSVRecursiveBlock(triBottom, d_bBottom, d_xBottom, depth - 1);

        // Combine the results
        cudaMemcpy(d_x, d_xTop, mid * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_x + mid, d_xBottom, (tri.rows - mid) * sizeof(float), cudaMemcpyDeviceToDevice);

        // Free temporary vectors and submatrices
        cudaFree(d_xTop);
        cudaFree(d_bBottom);
        cudaFree(d_xBottom);
        freeCSRMatrix(triTop);
        freeCSRMatrix(rec);
        freeCSRMatrix(triBottom);
    }
}

// Extract a submatrix from a CSR matrix
CSRMatrix extractSubMatrix(const CSRMatrix& mat, int startRow, int endRow, int startCol, int endCol) {
    CSRMatrix subMat;
    subMat.rows = endRow - startRow;
    subMat.cols = endCol - startCol;

    // Count non-zero elements in the submatrix
    int nnz = 0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = mat.d_rowPtr[i]; j < mat.d_rowPtr[i + 1]; ++j) {
            int col = mat.d_colIdx[j];
            if (col >= startCol && col < endCol) {
                nnz++;
            }
        }
    }
    subMat.nnz = nnz;

    // Allocate device memory for the submatrix
    allocateCSRMatrix(subMat, subMat.rows, subMat.cols, subMat.nnz);

    // Extract submatrix data
    std::vector<int> h_rowPtr(subMat.rows + 1, 0);
    std::vector<int> h_colIdx;
    std::vector<float> h_values;

    for (int i = startRow; i < endRow; ++i) {
        int newRow = i - startRow;
        h_rowPtr[newRow] = h_values.size();
        for (int j = mat.d_rowPtr[i]; j < mat.d_rowPtr[i + 1]; ++j) {
            int col = mat.d_colIdx[j];
            if (col >= startCol && col < endCol) {
                h_colIdx.push_back(col - startCol);
                h_values.push_back(mat.d_values[j]);
            }
        }
    }
    h_rowPtr[subMat.rows] = h_values.size();

    // Copy data to device
    cudaMemcpy(subMat.d_rowPtr, h_rowPtr.data(), (subMat.rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subMat.d_colIdx, h_colIdx.data(), h_colIdx.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(subMat.d_values, h_values.data(), h_values.size() * sizeof(float), cudaMemcpyHostToDevice);

    return subMat;
}

// Allocate device memory for a CSR matrix
void allocateCSRMatrix(CSRMatrix& mat, int rows, int cols, int nnz) {
    mat.rows = rows;
    mat.cols = cols;
    mat.nnz = nnz;
    cudaMalloc((void**)&mat.d_rowPtr, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&mat.d_colIdx, nnz * sizeof(int));
    cudaMalloc((void**)&mat.d_values, nnz * sizeof(float));
}

// Free device memory for a CSR matrix
void freeCSRMatrix(CSRMatrix& mat) {
    cudaFree(mat.d_rowPtr);
    cudaFree(mat.d_colIdx);
    cudaFree(mat.d_values);
}

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


//
//int main() {
//    // File path to the MTX file
//    std::string filename = "matrices/chipcool0.mtx";
//
//    // Vectors to store CSR format data
//    std::vector<unsigned int> row_ptr, col_ids;
//    std::vector<float> data;
//    unsigned int n_rows, n_cols;
//
//    // Read the matrix from the file and convert it to CSR format
//    try {
//        read_matrix_market(filename, row_ptr, col_ids, data, n_rows, n_cols);
//    }
//    catch (const std::exception& e) {
//        std::cerr << e.what() << std::endl;
//        return EXIT_FAILURE;
//    }
//
//    // Right-hand side vector
//    std::vector<float> h_b(n_cols, 1.0f);
//    std::vector<float> h_x_manual(n_rows, 0.0f); // Result from manual implementation
//    std::vector<float> h_x_cusparse(n_rows, 0.0f); // Result from cuSPARSE implementation
//
//    //srand(static_cast<unsigned int>(time(NULL))); // Seed the random number generator
//    //for (unsigned int i = 0; i < n_cols; ++i) {
//    //    h_b[i] = static_cast<float>(rand()) / RAND_MAX; // Random value between 0 and 1
//    //}
//
//    // Performance metrics
//    float manual_time, manual_throughput, manual_bandwidth;
//    float cusparse_time, cusparse_throughput, cusparse_bandwidth;
//
//    //// Perform sparse triangular solve using manual implementation
//    //csr_sptrsv_host<float>(n_rows, col_ids, row_ptr, data, h_b, h_x_manual,
//    //    manual_time, manual_throughput, manual_bandwidth);
//
//    sptrsv_recursive_block_host(
//        n_rows, row_ptr, col_ids, data, h_b, h_x_manual,
//        manual_time, manual_throughput, manual_bandwidth
//    );
//
//    // Perform sparse triangular solve using cuSPARSE
//    cusparse_sptrsv(n_rows, data.size(), row_ptr, col_ids, data, h_b, h_x_cusparse,
//        cusparse_time, cusparse_throughput, cusparse_bandwidth);
//
//    // Compare results
//    bool results_match = compare_results(n_rows, h_x_manual.data(), h_x_cusparse.data());
//
//
//
//    if (results_match) {
//        std::cout << "Results match!" << std::endl;
//        std::cout << "Manual Implementation Performance:" << std::endl;
//        std::cout << "Execution time (ms): " << manual_time << std::endl;
//        std::cout << "Throughput (GFLOPS): " << manual_throughput / 1e9 << std::endl;
//        std::cout << "Memory bandwidth (GB/s): " << manual_bandwidth << std::endl;
//
//        std::cout << "cuSPARSE Implementation Performance:" << std::endl;
//        std::cout << "Execution time (ms): " << cusparse_time << std::endl;
//        std::cout << "Throughput (GFLOPS): " << cusparse_throughput / 1e9 << std::endl;
//        std::cout << "Memory bandwidth (GB/s): " << cusparse_bandwidth << std::endl;
//    }
//    else {
//        std::cout << "Results do not match!" << std::endl;
//    }
//
//    return 0;
//}

int main() {
    // Hardcoded 5x5 lower triangular matrix in CSR format
    unsigned int n_rows = 8;
    unsigned int n_cols = 8;

    std::vector<unsigned int> row_ptr = {
        0, 1, 3, 4, 6, 8, 11, 13, 15
    };

    std::vector<unsigned int> col_ids = {
        0,       // row 0
        0, 1,    // row 1
        2,       // row 2
        0, 3,    // row 3
        1, 4,  // row 4
        0, 3, 5, // row 5
        2, 6,    // row 6
        1, 7     // row 7
    };

    std::vector<float> data = {
        2.0,      // row 0
        -1.5, 3.0, // row 1
        4.0,      // row 2
        2.2, 1.0, // row 3
        5.5, 6.0, // row 4
        1.1, 3.3, 7.0, // row 5
        2.2, 8.0, // row 6
        4.4, 9.0  // row 7
    };

    // Right-hand side vector
    std::vector<float> h_b(n_cols, 1.0f);
    std::vector<float> h_x_manual(n_rows, 0.0f); // Result from manual implementation
    std::vector<float> h_x_cusparse(n_rows, 0.0f); // Result from cuSPARSE implementation

    // Performance metrics
    float manual_time, manual_throughput, manual_bandwidth;
    float cusparse_time, cusparse_throughput, cusparse_bandwidth;

    sptrsv_recursive_block_host(
        n_rows, row_ptr, col_ids, data, h_b, h_x_manual,
        manual_time, manual_throughput, manual_bandwidth
    );

    // Perform sparse triangular solve using cuSPARSE
    cusparse_sptrsv(n_rows, data.size(), row_ptr, col_ids, data, h_b, h_x_cusparse,
        cusparse_time, cusparse_throughput, cusparse_bandwidth);

    // Compare results
    bool results_match = compare_results(n_rows, h_x_manual.data(), h_x_cusparse.data());

    // Print the matrix in CSR format
    std::cout << "Matrix in CSR format:" << std::endl;
    for (unsigned int i = 0; i < n_rows; ++i) {
        for (unsigned int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            std::cout << "Row " << i << ", Col " << col_ids[j]
                << ", Value " << data[j] << std::endl;
        }
    }

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
    else {
        std::cout << "Results do not match!" << std::endl;
    }

    return 0;
}
