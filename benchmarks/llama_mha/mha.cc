//===- mha.cc ---------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "lut_based_ops.h"
#include "zero.cc"

template <typename T_in, typename T_out, unsigned k, unsigned n, unsigned s,
          unsigned t>
void vecmat_vectorized(int offset, T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(n % t == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 because that is the number of
                         // column vectors (a_vec_0_0..a_vec_3_1) we create
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value ||
                std::is_same<T_in, int16_t>::value);

  event0();
  T_in *__restrict a_ptr = a;
  T_in *__restrict b_ptr = b;

  T_out *__restrict c_ptr = c + offset; // reset to the first row of C output on
  // each outer loop tieration
  for (int col = 0; col < n; col += t) {

    const T_in *__restrict a_ptr1 = a_ptr;
    const T_in *__restrict b_ptr1 = b_ptr;
    for (int row = 0; row < k; row += 8)
      chess_loop_range(k / 8, ) {
        aie::vector<T_in, 8> a_vec = aie::load_v<8>(a_ptr1);
        a_ptr1 += 8;
        aie::accum<accfloat, t> c_acc_in;
        c_acc_in.from_vector(aie::load_v<t>(c_ptr));

        for (int i = 0; i < 8; i++) {
          const aie::vector<T_in, t> b_vec = aie::load_v<t>(b_ptr1);
          b_ptr1 += n;
          const aie::vector<T_in, t> s0 = aie::broadcast<T_in, t>(a_vec[i]);
          c_acc_in = mac(c_acc_in, s0, b_vec);
        }
        aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
      }
    b_ptr += t;
    c_ptr += t;
  }
  event1();
}

void cosf_poly_bf16_scalarized(const bfloat16* __restrict inputs,
                               bfloat16* __restrict outputs, unsigned N) {
  constexpr float sin_poly_factors[4] = {
      -0x1.5555555555555p-3f, 0x1.1111111110bb3p-7f,
      -0x1.a01a019e83e5cp-13f, 0x1.71de3796cde01p-19f};
  constexpr float cos_poly_factors[4] = {
      0x1.5555555555555p-5f, -0x1.6c16c16c16967p-10f,
      0x1.A01A019F4EC91p-16f, -0x1.27E4FA17F667Bp-22f};

  const float twobypi = 0x1.45f306dc9c883p-1f;
  const float piby2 = 0x1.921fb54400000p0f;

  for (unsigned i = 0; i < N; ++i) {
    float ux = static_cast<float>(inputs[i]);
    int z = static_cast<int>(ux * twobypi + 0.5f);
    float x = ux - static_cast<float>(z) * piby2;
    float r = x * x;
    float r2 = r * r;

    float sin_poly = ((sin_poly_factors[3] * r + sin_poly_factors[2]) * r +
                      sin_poly_factors[1]) *
                         r +
                     sin_poly_factors[0];

    float cos_poly = ((cos_poly_factors[3] * r + cos_poly_factors[2]) * r +
                      cos_poly_factors[1]) *
                         r +
                     cos_poly_factors[0];

    float val;
    if ((z & 1) == 0) {
      // even: use cosine approximation
      val = 1.0f - 0.5f * r + r2 * cos_poly;
    } else {
      // odd: use sine approximation
      val = x + x * r * sin_poly;
    }

    // Determine final sign
    if (((z >> 1) ^ (z & 1)) & 1) val = -val;
    outputs[i] = static_cast<bfloat16>(val);
  }
}


void sinf_poly_bf16_scalarized(const bfloat16* __restrict inputs,
                               bfloat16* __restrict outputs, unsigned N) {
  constexpr float sin_poly_factors[4] = {
      -0x1.5555555555555p-3f, 0x1.1111111110bb3p-7f,
      -0x1.a01a019e83e5cp-13f, 0x1.71de3796cde01p-19f};
  constexpr float cos_poly_factors[4] = {
      0x1.5555555555555p-5f, -0x1.6c16c16c16967p-10f,
      0x1.A01A019F4EC91p-16f, -0x1.27E4FA17F667Bp-22f};

  const float twobypi = 0x1.45f306dc9c883p-1f;
  const float piby2 = 0x1.921fb54400000p0f;

  for (unsigned i = 0; i < N; ++i) {
    float ux = static_cast<float>(inputs[i]);
    int z = static_cast<int>(ux * twobypi + 0.5f);
    float x = ux - static_cast<float>(z) * piby2;
    float r = x * x;
    float r2 = r * r;

    float sin_poly = ((sin_poly_factors[3] * r + sin_poly_factors[2]) * r +
                      sin_poly_factors[1]) *
                         r +
                     sin_poly_factors[0];

    float cos_poly = ((cos_poly_factors[3] * r + cos_poly_factors[2]) * r +
                      cos_poly_factors[1]) *
                         r +
                     cos_poly_factors[0];

    float val;
    if (z & 1) {
      // odd: use cosine approximation
      val = 1.0f - 0.5f * r + r2 * cos_poly;
    } else {
      // even: use sine approximation
      val = x + x * r * sin_poly;
    }

    if (z & 2) val = -val;
    outputs[i] = static_cast<bfloat16>(val);
  }
}

template <unsigned n>
void freq_pos_bf16_24(const int pos, bfloat16 *__restrict outputs) {

  alignas(aie::vector_decl_align) const bfloat16 freq[] = {
      0x1p0,          0x1.5cd25p-1,   0x1.db4c78p-2,  0x1.43d136p-2,
      0x1.b93a6cp-3,  0x1.2c9af4p-3,  0x1.99999ap-4,  0x1.170ea6p-4,
      0x1.7c3d2cp-5,  0x1.030dc6p-5,  0x1.60fb8ap-6,  0x1.e0f7eep-7,
      0x1.47ae14p-7,  0x1.be7dd8p-8,  0x1.3030fp-8,   0x1.9e7c72p-9,
      0x1.1a62d8p-9,  0x1.80c652p-10, 0x1.0624dep-10, 0x1.653176p-11,
      0x1.e6b4bcp-12, 0x1.4b96cep-12, 0x1.c3d114p-13, 0x1.33d1eap-13,
  };

  const bfloat16 *freq_ptr = freq;
  aie::vector<bfloat16, n> vec0 = aie::load_v<n>(freq_ptr);
  freq_ptr += n;
  aie::vector<bfloat16, n> vec1 = aie::load_v<n>(freq_ptr);
  freq_ptr += n;
  aie::vector<bfloat16, n> vec2 = aie::load_v<n>(freq_ptr);
  freq_ptr += n;
  aie::vector<bfloat16, n> vecPos =
      aie::broadcast<bfloat16, n>(aie::to_float<bfloat16>(pos));

  bfloat16 *__restrict pOut = outputs;
  aie::store_v(pOut, aie::mul(vecPos, vec0).template to_vector<bfloat16>());
  pOut += n;
  aie::store_v(pOut, aie::mul(vecPos, vec1).template to_vector<bfloat16>());
  pOut += n;
  aie::store_v(pOut, aie::mul(vecPos, vec2).template to_vector<bfloat16>());
  pOut += n;
}

template <unsigned N>
void shuffle_apply_rope_bf16_8(int offset, const bfloat16 *__restrict fcr,
                               const bfloat16 *__restrict fci,
                               bfloat16 *__restrict outputs) {

  constexpr unsigned n = 8;
  constexpr unsigned two_n = n * 2;

  const bfloat16 *__restrict pFcr = fcr;
  const bfloat16 *__restrict pFci = fci;
  for (unsigned i = 0; i < N; i += two_n) {
    const bfloat16 *__restrict pA1 = outputs + i + offset;
    aie::vector<bfloat16, n> v0Lo = aie::load_v<n>(pA1);
    pA1 += n;
    aie::vector<bfloat16, n> v0Hi = aie::load_v<n>(pA1);
    pA1 += n;
    aie::vector<bfloat16, two_n> v0 = aie::concat(v0Lo, v0Hi);

    aie::vector<bfloat16, n> zerosV8 = aie::zeros<bfloat16, n>();
    aie::vector<bfloat16, two_n> zerosV16 = aie::zeros<bfloat16, two_n>();
    aie::vector<bfloat16, two_n> v0Shuffled = extract_v16bfloat16(
        ::shuffle(aie::concat(zerosV16, v0), T16_8x2), 1);
    aie::vector<bfloat16, n> v0ShuffledEven = extract_v8bfloat16(v0Shuffled, 0);
    aie::vector<bfloat16, n> v0ShuffledOdd = extract_v8bfloat16(v0Shuffled, 1);

    aie::vector<bfloat16, n> vFcr = aie::load_v<n>(pFcr);
    pFcr += n;
    aie::vector<bfloat16, n> vFci = aie::load_v<n>(pFci);
    pFci += n;
    aie::vector<bfloat16, n> vOutEven =
        aie::sub(aie::mul(v0ShuffledEven, vFcr), aie::mul(v0ShuffledOdd, vFci));
    aie::vector<bfloat16, n> vOutOdd =
        aie::add(aie::mul(v0ShuffledEven, vFci), aie::mul(v0ShuffledOdd, vFcr));
    aie::vector<bfloat16, two_n> vOutUnshuffled = extract_v16bfloat16(
        ::shuffle(aie::concat(zerosV8, zerosV8, vOutEven, vOutOdd),
                  T16_2x8),
        1);
    aie::vector<bfloat16, n> vOutLo = extract_v8bfloat16(vOutUnshuffled, 0);
    aie::vector<bfloat16, n> vOutHi = extract_v8bfloat16(vOutUnshuffled, 1);

    bfloat16 *__restrict pOut = outputs + i + offset;
    aie::store_v(pOut, vOutLo);
    pOut += n;
    aie::store_v(pOut, vOutHi);
    pOut += n;
  }
}

extern "C" {

void sinf_bf16_24_8(const bfloat16 *__restrict inputs,
                    bfloat16 *__restrict outputs) {
  sinf_poly_bf16_scalarized(inputs, outputs, 24);
}
void cosf_bf16_24_8(const bfloat16 *__restrict inputs,
                    bfloat16 *__restrict outputs) {
  cosf_poly_bf16_scalarized(inputs, outputs, 24);
}

void freq_pos_bf16_24_8(const int pos, bfloat16 *__restrict outputs) {
  // Head size 48 (24 even + 24 odd indexed elements), vector size 8.
  freq_pos_bf16_24<8>(pos, outputs);
}

void shuffle_apply_rope_bf16_48(int offset, const bfloat16 *__restrict fcr,
                                const bfloat16 *__restrict fci,
                                bfloat16 *__restrict outputs) {
  shuffle_apply_rope_bf16_8<48>(offset, fcr, fci, outputs);
}

void roundI32ToBf16(const int *__restrict inputs, int offset,
                    bfloat16 *__restrict outputs) {
  const int *__restrict pIn = inputs;
  bfloat16 *__restrict pOut = outputs;
  pOut += offset;
  for (unsigned i = 0; i < 3; i++) {
    aie::store_v(pOut, to_float<bfloat16>(aie::load_v<16>(pIn)));
    pIn += 16;
    pOut += 16;
  }
}

// TODO: figure out better way to do this
void split_qkv(const bfloat16 *__restrict qkv,
                       bfloat16 *__restrict q,
                       bfloat16 *__restrict k,
                       bfloat16 *__restrict v) {
  for (int i = 0; i < 48; i++) {
    q[i] = qkv[i];         // Q from [0, 48)
    k[i] = qkv[i + 48];    // K from [48, 96)
    v[i] = qkv[i + 96];    // V from [96, 144)
  }
}

#ifndef DIM_N
#define DIM_N 48
#endif

#ifndef DIM_K
#define DIM_K 96
#endif

#define combos(X) X(bfloat16, bf16, bfloat16, bf16)

#define vecmat_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out)                                \
  void vecmat_##mlir_type_in##_##mlir_type_out(                                \
      int offset, ctype_in *a_in, ctype_in *b_in, ctype_out *c_out) {          \
    vecmat_vectorized<ctype_in, ctype_out, DIM_K, DIM_N, 8, 16>(offset, a_in,  \
                                                                b_in, c_out);  \
  }
  
#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out)                                  \
  void linalg_fill_##mlir_type_out(int offset, ctype_out *c_out) {             \
    zero_vectorized<ctype_out, 3, DIM_N, 32>(offset, c_out);                   \
  }

combos(zero_vectorized_c_func) combos(vecmat_vectorized_c_func) 

} // extern "C"