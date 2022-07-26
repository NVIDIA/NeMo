/* coding=utf-8
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <cuda_fp16.h>
#include <c10/macros/Macros.h>

namespace {

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN_NATIVE(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
#if CUDA_VERSION >= 9000
    return __shfl_down_sync(mask, value, laneMask, width);
#else
    return __shfl_down(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ acc_t warp_reduce_new(acc_t val) {
  ReduceOp<acc_t> r;
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
  {
      val = r(val, WARP_SHFL_DOWN_NATIVE(val, offset, WARP_SIZE));
  }
  return val;
}


template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void scaled_masked_softmax_warp_backward_new(
    output_t *gradInput, //[batches, attn_heads, q_len, k_len]
    input_t *grad, 
    const input_t *output, //[batches, attn_heads, q_len, k_len]
    acc_t scale, 
    int element_count)
{
    int threads_per_block = blockDim.x; 
    //the first element_count*2 elements are used for cache, the last 128 is used for reduction
    extern __shared__ acc_t shared_data[];
    input_t *local_data = (input_t *)shared_data;
    input_t *output_data = &local_data[element_count];
    // maximum shared cached 128, enough for 4096 elements reduction into 4096/32= 128 elements
    acc_t *shared = (acc_t *)(&(local_data[element_count*2]));

    int num_reductions =  (element_count - 1) / threads_per_block + 1;

    int offset = blockIdx.x * element_count;

    int local_idx = threadIdx.x;
    int lane = threadIdx.x % C10_WARP_SIZE;
    int wid = threadIdx.x / C10_WARP_SIZE;
    int warps_per_thread_block = threads_per_block / C10_WARP_SIZE; 

    // load the data to local data
    acc_t val = 0.0;
    for (int i = 0; i < num_reductions; i++){
        if (i*threads_per_block + local_idx < element_count){
            val = output[offset + i*threads_per_block + local_idx];
            output_data[i*threads_per_block + local_idx] = val;
            local_data[i*threads_per_block + local_idx] = val * grad[offset + i*threads_per_block + local_idx];
        }
        __syncthreads();
    }

    // find the sum 
    for (int i = local_idx; i < (element_count - 1) / C10_WARP_SIZE + 1; i += threads_per_block){
        shared[i] = 0.0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < num_reductions; i++){
        if (i*threads_per_block + local_idx < element_count){
            val = local_data[i*threads_per_block + local_idx];
        }
        else{
            val = 0.0;
        }
        __syncthreads();
        val = warp_reduce_new<acc_t, C10_WARP_SIZE, Add>(val);
        if (lane==0 && wid + warps_per_thread_block * i < (element_count - 1) / C10_WARP_SIZE + 1) {
            shared[wid + warps_per_thread_block*i] = val;
        }
        __syncthreads();
    }

    // final shared reduction

    int shared_mem_len = (element_count - 1) / C10_WARP_SIZE + 1;
    int num_warps = (shared_mem_len - 1) / C10_WARP_SIZE + 1;
    while ( shared_mem_len > 1 ){
        #pragma unroll
        for (int i = 0; i < num_reductions; i++){
            if (i*threads_per_block + local_idx < shared_mem_len){
                val = shared[i*threads_per_block + local_idx];
            }
            else{
                val = 0.0;
            }
            __syncthreads();
            val = warp_reduce_new<acc_t, C10_WARP_SIZE, Add>(val);
            if (lane==0) {
                shared[wid + warps_per_thread_block * i] = val;
            }
            __syncthreads();
        }
        shared_mem_len = num_warps;
        num_warps = (shared_mem_len - 1) / C10_WARP_SIZE + 1;
    }
    val = shared[0];
    #pragma unroll
    for (int i = local_idx; i < element_count; i += threads_per_block){
        gradInput[offset + i] = (output_t)(scale*(local_data[i] - output_data[i]*val)); 
    }
}

} // end of anonymous namespace

template<typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_masked_softmax_backward_new(
    output_t *grad_input, 
    input_t *grad, 
    const input_t *output, 
    const acc_t scale, 
    int query_seq_len, 
    int key_seq_len, 
    int batches,
    int attn_heads)
{
    if (key_seq_len == 0)
    {
        return;
    }
    else
    {
        int batch_count = batches * attn_heads * query_seq_len;
        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;
        int num_warps = (key_seq_len - 1) / C10_WARP_SIZE + 1;
        dim3 blocks(batch_count, 1, 1);
        dim3 threads(threads_per_block, 1, 1);

        scaled_masked_softmax_warp_backward_new<input_t, output_t, acc_t, 12>
            <<<blocks, threads, sizeof(input_t)*key_seq_len*2 + sizeof(acc_t)*num_warps, at::cuda::getCurrentCUDAStream()>>>(grad_input, grad, output, scale, key_seq_len);
    }
}

/*
 * Extended softmax (from native aten pytorch) with following additional features
 * 1) input scaling
 * 2) Explicit masking
 */	
template <typename input_t, typename output_t, typename acc_t>
__global__ void scaled_masked_softmax_warp_forward_new(
    output_t *dst, 
    const input_t *src,
    const uint8_t *mask, 
    const acc_t scale, 
    int query_len,          // query_len
    int attn_heads,
    int element_count,      // key_len
    int pad_batches)        // mask batch size 
{
    // min threawds_per_block has to be bigger than 128
    int threads_per_block = blockDim.x; 
    //  the first element_count is used for cache, the last 128 is used for reduction
    extern __shared__ acc_t local_data[];
    // maximum shared cached 128, enough for 4096 elements reduction into 4096/32= 128 elements
    acc_t *shared = &(local_data[element_count]);
    // number of 1024 threads reductions 
    int num_reductions =  (element_count - 1) / threads_per_block + 1;

    int offset = blockIdx.x * element_count;
    int mask_offset;
    int query_id = blockIdx.x % query_len; 
    if (pad_batches == 1){
        // broadcaste the mask tensor 
        mask_offset = query_id * element_count; 
    }
    else{
        int mask_batch_id = blockIdx.x / attn_heads / query_len;
        mask_offset = (mask_batch_id * query_len + query_id) * element_count;
    }

    int local_idx = threadIdx.x;
    int lane = threadIdx.x % C10_WARP_SIZE;
    int wid = threadIdx.x / C10_WARP_SIZE;
    int warps_per_thread_block = threads_per_block / C10_WARP_SIZE; 

    // load the data to local data
    for (int i = local_idx; i < element_count; i += threads_per_block)
    {
        // TODO, use the copy vector method
        if (mask[mask_offset + i] == 1)
        {
            local_data[i] = -10000.0;
        }
        else
        {
            local_data[i] = src[offset + i] * scale;
        }
    }

    // first find the max value
    for (int i = local_idx; i < (element_count - 1) / C10_WARP_SIZE + 1; i += threads_per_block){
        shared[i] = -10000.0;
    }
    __syncthreads();
    acc_t val = -10000.0;
    #pragma unroll
    for (int i = 0; i < num_reductions; i++){
        if (i*threads_per_block + local_idx < element_count){
            val = local_data[i*threads_per_block + local_idx];
        }
        else{
            val = -10000.0;
        }
        __syncthreads();
        val = warp_reduce_new<acc_t, C10_WARP_SIZE, Max>(val);

        if (lane==0 && wid + warps_per_thread_block * i < (element_count - 1) / C10_WARP_SIZE + 1) {
            shared[wid + warps_per_thread_block*i] = val;
        }
        __syncthreads();
    }

    // final shared reduction
    int shared_mem_len = (element_count - 1) / C10_WARP_SIZE + 1;
    int num_warps = (shared_mem_len - 1) / C10_WARP_SIZE + 1;
    while ( shared_mem_len > 1 ){
        #pragma unroll
        for (int i = 0; i < num_reductions; i++){
            if (i*threads_per_block + local_idx < shared_mem_len){
                val = shared[i*threads_per_block + local_idx];
            }
            else{
                val = -10000.0;
            }
            __syncthreads();
            val = warp_reduce_new<acc_t, C10_WARP_SIZE, Max>(val);
            if (lane==0) {
                shared[wid + warps_per_thread_block * i] = val;
            }
            __syncthreads();
        }
        shared_mem_len = num_warps;
        num_warps = (shared_mem_len - 1) / C10_WARP_SIZE + 1;
    }

    acc_t reduced_val = shared[0];
    if (reduced_val < -10000.0 + 0.1){
        // if everything is masked, pay attention to nothing
        #pragma unroll
        for (int i = local_idx; i < element_count; i += threads_per_block){
             dst[offset + i] = 0.0;
        }
        return;
    }

    // update the values
    #pragma unroll
    for (int i = local_idx; i < element_count; i += threads_per_block){
        local_data[i] = std::exp(local_data[i] - reduced_val);
    }

    // find the sum 
    for (int i = local_idx; i < (element_count - 1) / C10_WARP_SIZE + 1; i += threads_per_block){
        shared[i] = 0.0;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < num_reductions; i++){
        if (i*threads_per_block + local_idx < element_count){
            val = local_data[i*threads_per_block + local_idx];
        }
        else{
            val = 0.0;
        }
        __syncthreads();

        val = warp_reduce_new<acc_t, C10_WARP_SIZE, Add>(val);
        if (lane==0 && wid + warps_per_thread_block * i < (element_count - 1) / C10_WARP_SIZE + 1) {
            shared[wid + warps_per_thread_block*i] = val;
        }
        __syncthreads();
    }

    shared_mem_len = (element_count - 1) / C10_WARP_SIZE + 1;
    num_warps = (shared_mem_len - 1) / C10_WARP_SIZE + 1;
    while ( shared_mem_len > 1 ){
        #pragma unroll
        for (int i = 0; i < num_reductions; i++){
            if (i*threads_per_block + local_idx < shared_mem_len){
                val = shared[i*threads_per_block + local_idx];
            }
            else{
                val = 0.0;
            }
            __syncthreads();
            val = warp_reduce_new<acc_t, C10_WARP_SIZE, Add>(val);
            if (lane==0) {
                shared[wid + warps_per_thread_block * i] = val;
            }
            __syncthreads();
        }
        shared_mem_len = num_warps;
        num_warps = (shared_mem_len - 1) / C10_WARP_SIZE + 1;
    }

    reduced_val = shared[0];

    #pragma unroll
    for (int i = local_idx; i < element_count; i += threads_per_block){
         dst[offset + i] = local_data[i] / reduced_val;
    }
}


template<typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_masked_softmax_forward_new(
    output_t *dst, 
    const input_t *src, 
    const uint8_t *mask,
    const input_t scale, 
    int query_seq_len, 
    int key_seq_len, 
    int batches,
    int attn_heads,
    int pad_batches)
{
    if (key_seq_len == 0) {
        return;
    } else {
        int batch_count = batches * attn_heads * query_seq_len;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        // calculate the needed shared memory
        int num_warps = (key_seq_len - 1) / C10_WARP_SIZE + 1;

        dim3 blocks(batch_count, 1, 1);
        dim3 threads(threads_per_block, 1, 1);
        scaled_masked_softmax_warp_forward_new<input_t, output_t, acc_t>
            <<<blocks, threads, sizeof(acc_t) * (key_seq_len + num_warps), at::cuda::getCurrentCUDAStream()>>>(dst, src, mask, scale, query_seq_len, attn_heads, key_seq_len, pad_batches);
    }
}
