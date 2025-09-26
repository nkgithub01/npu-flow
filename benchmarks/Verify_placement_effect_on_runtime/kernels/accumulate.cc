#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

template <typename T, const int N>
void zero(T *a) {
  for (int i = 0; i < N; i++) {
    a[i] = 0;
  }
}

template <typename T_in, typename T_out, const int N>
void accumulate(T_in *a, T_in b, T_out *c) {
  for (int i = 0; i < N; i++) {
    c[i] += a[i] + b;
  }
}

extern "C" {

#define ACCUMULATE_FUNCTION(T_in, T_out, N) \
	void accumulate_##T_in##_##T_out##_##N(T_in *a, T_in b, T_out *c) { \
		accumulate<T_in, T_out, N>(a, b, c); \
	}

#define ZERO_FUNCTION(T, N) \
	void zero_##T##_##N(T *a) { \
		zero<T, N>(a); \
	}

ACCUMULATE_FUNCTION(int8_t, int8_t, 1)
ACCUMULATE_FUNCTION(int16_t, int16_t, 1)
ACCUMULATE_FUNCTION(int32_t, int32_t, 1)

ZERO_FUNCTION(int8_t, 1)
ZERO_FUNCTION(int16_t, 1)
ZERO_FUNCTION(int32_t, 1)

}