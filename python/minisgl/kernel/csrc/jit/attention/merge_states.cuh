// Device code from kernels/aot/csrc/attention/merge_attn_states.cu.
// The host entry point uses the same 128-thread launch through TVM-FFI.
#pragma once
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>

namespace sglang::merge_states {
// Helper functions to convert between different data types
// (float, half, bfloat16) for the merge attention states kernel.
inline __device__ float to_float(float u) {
  return u;
}
inline __device__ float to_float(half u) {
  return __half2float(u);
}
inline __device__ float to_float(__nv_bfloat16 u) {
  return __bfloat162float(u);
}
inline __device__ void from_float(float& d, float s) {
  d = s;
}
inline __device__ void from_float(half& d, float s) {
  d = __float2half(s);
}
inline __device__ void from_float(__nv_bfloat16& d, float s) {
  d = __float2bfloat16(s);
}

// Implements section 2.2 of https://www.arxiv.org/pdf/2501.01005
template <typename scalar_t, const uint NUM_THREADS>
__global__ void merge_attn_states_kernel(
    scalar_t* output,
    float* output_lse,
    const scalar_t* prefix_output,
    const float* prefix_lse,
    const scalar_t* suffix_output,
    const float* suffix_lse,
    const uint num_tokens,
    const uint num_heads,
    const uint head_size) {
  using pack_128b_t = uint4;
  const uint pack_size = 16 / sizeof(scalar_t);
  const uint threads_per_head = head_size / pack_size;

  const uint global_idx = blockIdx.x * NUM_THREADS + threadIdx.x;
  const uint token_head_threads = num_tokens * num_heads * threads_per_head;

  if (global_idx >= token_head_threads) return;

  // global_idx -> token_idx + head_idx + pack_idx
  const uint token_head_idx = global_idx / threads_per_head;
  const uint pack_idx = global_idx % threads_per_head;

  const uint token_idx = token_head_idx / num_heads;
  const uint head_idx = token_head_idx % num_heads;

  const uint pack_offset = pack_idx * pack_size;  // (0~15)*8, etc.
  const uint head_offset = token_idx * num_heads * head_size + head_idx * head_size;
  const scalar_t* prefix_head_ptr = prefix_output + head_offset;
  const scalar_t* suffix_head_ptr = suffix_output + head_offset;
  scalar_t* output_head_ptr = output + head_offset;

  // float p_lse = prefix_lse[head_idx * num_tokens + token_idx];
  // float s_lse = suffix_lse[head_idx * num_tokens + token_idx];
  float p_lse = prefix_lse[token_idx * num_heads + head_idx];
  float s_lse = suffix_lse[token_idx * num_heads + head_idx];
  p_lse = std::isinf(p_lse) ? -std::numeric_limits<float>::infinity() : p_lse;
  s_lse = std::isinf(s_lse) ? -std::numeric_limits<float>::infinity() : s_lse;

  const float max_lse = fmaxf(p_lse, s_lse);
  p_lse = p_lse - max_lse;
  s_lse = s_lse - max_lse;
  const float p_se = expf(p_lse);
  const float s_se = expf(s_lse);
  const float out_se = p_se + s_se;
  const float p_scale = p_se / out_se;
  const float s_scale = s_se / out_se;

  if (pack_offset < head_size) {
    // Pack 128b load
    pack_128b_t p_out_pack = reinterpret_cast<const pack_128b_t*>(prefix_head_ptr)[pack_offset / pack_size];
    pack_128b_t s_out_pack = reinterpret_cast<const pack_128b_t*>(suffix_head_ptr)[pack_offset / pack_size];
    pack_128b_t o_out_pack;

#pragma unroll
    for (uint i = 0; i < pack_size; ++i) {
      // Always use float for FMA to keep high precision.
      // half(uint16_t), bfloat16, float -> float.
      const float p_out_f = to_float(reinterpret_cast<const scalar_t*>(&p_out_pack)[i]);
      const float s_out_f = to_float(reinterpret_cast<const scalar_t*>(&s_out_pack)[i]);
      // fma: a * b + c = p_out_f * p_scale + (s_out_f * s_scale)
      const float o_out_f = p_out_f * p_scale + (s_out_f * s_scale);
      // float -> half(uint16_t), bfloat16, float.
      from_float(reinterpret_cast<scalar_t*>(&o_out_pack)[i], o_out_f);
    }

    // Pack 128b storage
    reinterpret_cast<pack_128b_t*>(output_head_ptr)[pack_offset / pack_size] = o_out_pack;
  }
  // We only need to write to output_lse once per head.
  if (output_lse != nullptr && pack_idx == 0) {
    float out_lse = logf(out_se) + max_lse;
    output_lse[token_idx * num_heads + head_idx] = out_lse;
  }
}

template <typename T>
void run(tvm::ffi::TensorView prefix, tvm::ffi::TensorView prefix_lse,
         tvm::ffi::TensorView suffix, tvm::ffi::TensorView suffix_lse,
         tvm::ffi::TensorView output, tvm::ffi::TensorView output_lse) {
  using namespace host;
  auto tokens = SymbolicSize{"tokens"};
  auto heads = SymbolicSize{"heads"};
  auto width = SymbolicSize{"width"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  for (auto value : {prefix, suffix, output}) {
    TensorMatcher({tokens, heads, width}).with_dtype<T>().with_device(device).verify(value);
  }
  for (auto value : {prefix_lse, suffix_lse, output_lse}) {
    TensorMatcher({tokens, heads}).with_dtype<float>().with_device(device).verify(value);
  }
  constexpr uint threads = 128;
  constexpr uint pack = 16 / sizeof(T);
  const uint n = tokens.unwrap(), h = heads.unwrap(), d = width.unwrap();
  RuntimeCheck(d % pack == 0, "Head width must be a multiple of the vector size");
  const uint work = n * h * (d / pack);
  if (work == 0) return;
  LaunchKernel((work + threads - 1) / threads, threads, device.unwrap())(
      merge_attn_states_kernel<T, threads>, static_cast<T*>(output.data_ptr()),
      static_cast<float*>(output_lse.data_ptr()), static_cast<const T*>(prefix.data_ptr()),
      static_cast<const float*>(prefix_lse.data_ptr()), static_cast<const T*>(suffix.data_ptr()),
      static_cast<const float*>(suffix_lse.data_ptr()), n, h, d);
}
}  // namespace sglang::merge_states
