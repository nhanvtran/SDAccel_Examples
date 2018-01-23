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

// #include "utils.h"

#include "parameters.h"
// #include "myproject.h"
#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_layer.h"
#include "nnet_utils/nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w2.h"
#include "weights/b2.h"

#define BUFFER_SIZE 256
#define DATA_SIZE 16

void addVectors(int* a, int* c){
 for(int i = 0; i < N_OUTPUTS; i++) c[i] = a[i] + 1;
}

// need to do conversion of int to right precision
void myproject_hw(input_t* data, result_t* res)
{
    //hls-fpga-machine-learning insert IO
    // #pragma HLS STREAM variable=data dim=1
    // #pragma HLS STREAM variable=res dim=1

    // #pragma HLS PIPELINE
    // #pragma DATAFLOW
    
    // const_size_in   = N_INPUTS;
    // const_size_out  = N_OUTPUTS;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer1_t layer1_out[N_LAYER_1];
    // #pragma HLS STREAM variable=layer1_out dim=1
    layer1_t logits1[N_LAYER_1];
    // #pragma HLS STREAM variable=logits1 dim=1
    nnet::compute_layer<input_t, layer1_t, config1>(data, logits1, w1, b1);
    nnet::relu<layer1_t, layer1_t, relu_config1>(logits1, layer1_out);

    result_t logits2[N_OUTPUTS];
    // #pragma HLS STREAM variable=logits2 dim=1
    nnet::compute_layer<layer1_t, result_t, config2>(layer1_out, logits2, w2, b2);
    nnet::sigmoid<result_t, result_t, sigmoid_config2>(logits2, res);
}

extern "C" void hls4ml_1layer(int* a, int* c)
//                int n_elements_ptr)
{
 
    #pragma HLS INTERFACE m_axi port=c offset=slave bundle=gmem
    #pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
    #pragma HLS INTERFACE s_axilite port=c bundle=control
    #pragma HLS INTERFACE s_axilite port=a bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    input_t arrayA[N_INPUTS];
    result_t arrayC[N_OUTPUTS];

    #pragma HLS array_partition variable=arrayA complete
    #pragma HLS array_partition variable=arrayC complete

    for (int i = 0 ; i < N_INPUTS ; i += BUFFER_SIZE)
    {
        int size = BUFFER_SIZE;
        if (i + size > N_INPUTS) size = N_INPUTS - i;
        readA: for (int j = 0 ; j < size ; j++) arrayA[j] = (input_t) a[i+j];
    }

    // addVectors(arrayA,arrayC);
    myproject_hw(arrayA,arrayC);

    for (int i = 0 ; i < N_OUTPUTS ; i += BUFFER_SIZE){
        int size = BUFFER_SIZE;
        for (int j = 0 ; j < size ; j++) c[i+j] = (int) arrayC[j];
    }  

    return;
}
