# Parallel-Distributed-Computer-Systems-2023

Here is my implementation of the CSR matrix multiplication using the PThreads, OpenMP and OpenCilk parallel libraries.


To see the detailed report of a parallel code implemention check [this](./Report.pdf).


# Compilation Instructions

If you have configured you '/opt/opencilk/bin/clang' to handle Pthreads and OpenMP flags you can run 'csr.c' and choose the parallelism aproach.
If your confifured clang doesn't support OpenMP or PThreads flas, you may compile the following csr executables indepedently, as follows;

To compile csr_pthreads write: 'gcc -pthread csr_pthreads.c -o csr_pthreads mmio.c'

To compile csr_openmp write: 'gcc -openmp csr_openmp.c -o csr_openmp mmio.c' (adjust for your compiler accordingly) 

To compile csr_opencilk write: '/opt/opencilk/bin/clang -fopencilk -O3 csr_opencilk.c -o csr_opencilk mmio.c' (adjust for your compiler accordingly) 

If you still meet terminal compilation or runtime erros, please contact with me.
