﻿# spmv-sptrsv-gpu-research

### Files
There are 3 main files. Each file corresponds to specific sections in the paper. 

`spmv.cu` --> Section 2

`sync_free_sptrsv.cu` --> Section 3.1

`block_recursive_sptrsv.cu` --> Section 3.4


### Running the code
Each file is **its own standalone unit of code and can be compiled and run on its own** (I didn't use any custom header files or anything, mainly because it kept bothering me having to transfer it back and forth between the server)

Input matrices need to be downloaded from SuiteSparse Matrix Collection

The name and path of the matrix must be specified in the `main()` function of each file prior to compiling and running
