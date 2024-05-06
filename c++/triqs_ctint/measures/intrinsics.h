#pragma once
#ifdef  USE_INTRINSICS
#include <immintrin.h> // for AVX2 intrinsics
#include <utility>
#include <complex>

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> separate_real_imaginary(const __m512d v1, const __m512d v2) noexcept {
  // Create index vectors for separating real and imaginary parts
  static const __m512i idx_real = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  static const __m512i idx_imag = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  // Separate real and imaginary parts
  __m512d real_v = _mm512_permutex2var_pd(v1, idx_real, v2);
  __m512d imag_v = _mm512_permutex2var_pd(v1, idx_imag, v2);
  return {real_v, imag_v};
}

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> interleave_vectors(__m512d v1, __m512d v2) noexcept {
  // Create index vectors for interleaving
  static const __m512i idx_real = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
  static const __m512i idx_imag = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
  // Separate real and imaginary parts
  __m512d real_v = _mm512_permutex2var_pd(v1, idx_real, v2);
  __m512d imag_v = _mm512_permutex2var_pd(v1, idx_imag, v2);
  return {real_v, imag_v};
}

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> load(std::complex<double> *a) {
  // Cast pointers to double*
  auto *a0_ptr = reinterpret_cast<double *>(a);
  auto *a1_ptr = reinterpret_cast<double *>(a + 4);

  // Load arrays into AVX-512 registers
  __m512d va0 = _mm512_loadu_pd(a0_ptr);
  __m512d va1 = _mm512_loadu_pd(a1_ptr);

  return {va0, va1};
}

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> load_and_separate(std::complex<double> *a) {
  auto [va0, va1] = load(a);
  // Separate real and imaginary parts
  auto [real, imag] = separate_real_imaginary(va0, va1);
  return {real, imag};
}

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> set_vector_to_complex(const std::complex<double> &a) {
  return {_mm512_set1_pd(a.real()), _mm512_set1_pd(a.imag())};
}

__attribute__((always_inline)) inline static std::pair<__m512d, __m512d> complex_mul_avx512(__m512d real0, __m512d imag0, __m512d real1,
                                                                                            __m512d imag1) {
  // (a+ib) * (c+id) = (ac - bd) + i(ad + bc)
  // (real0 + i*imag0) * (real1 + i*imag1) = (real0*real1 - imag0*imag1) + i*(real0*imag1 + imag0*real1)
  // Multiply complex numbers
  __m512d real = (real0 * real1 - imag0 * imag1);
  __m512d imag = (real0 * imag1 + imag0 * real1);
  // Interleave real and imaginary parts
  return {real, imag};
}

#endif