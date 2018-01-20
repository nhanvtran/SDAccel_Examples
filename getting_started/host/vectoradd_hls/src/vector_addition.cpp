/**********
Copyright (c) 2017, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

// This function represents an OpenCL kernel. The kernel will be call from
// host application using the xcl_run_kernels call. The pointers in kernel
// parameters with the global keyword represents cl_mem objects on the FPGA
// DDR memory.
//
#define BUFFER_SIZE 256
#define DATA_SIZE 16
extern "C" void vector_add(int* c, int* a, int* b)
//                int n_elements_ptr)
{
 
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem
  //#pragma HLS INTERFACE m_axi port=n_elements_ptr offset=slave bundle=gmem

    #pragma HLS INTERFACE s_axilite port=c bundle=control
    #pragma HLS INTERFACE s_axilite port=a bundle=control
    #pragma HLS INTERFACE s_axilite port=b bundle=control
  //#pragma HLS INTERFACE s_axilite port=n_elements_ptr bundle=control

    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int arrayA[BUFFER_SIZE];
    int arrayB[BUFFER_SIZE];

    #pragma HLS array_partition variable=arrayA complete
    #pragma HLS array_partition variable=arrayB complete

    int n_elements = DATA_SIZE;

    for (int i = 0 ; i < n_elements ; i += BUFFER_SIZE)
    {
        int size = BUFFER_SIZE;
        if (i + size > n_elements) size = n_elements - i;
        readA: for (int j = 0 ; j < size ; j++) arrayA[j] = a[i+j];
        readB: for (int j = 0 ; j < size ; j++) arrayB[j] = b[i+j];
        vadd_writeC: for (int j = 0 ; j < size ; j++) c[i+j] = arrayA[j] + arrayB[j];
    }

    return;
}
