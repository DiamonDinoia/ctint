#pragma once
#ifdef  USE_INTRINSICS
#include <immintrin.h> // for AVX2 intrinsics
#include <utility>
#include <complex>

__attribute__((always_inline, const)) inline constexpr static auto separate_real_imaginary(const __m512d v1, const __m512d v2) noexcept {
  // Create index vectors for separating real and imaginary parts
  static constexpr const __m512i idx_real{0, 2, 4, 6, 8, 10, 12, 14};
  static constexpr const __m512i idx_imag{1, 3, 5, 7, 9, 11, 13, 15};
  // reverse order of idx_real
  // Separate real and imaginary parts
  const auto&& real_v = _mm512_permutex2var_pd(v1, idx_real, v2);
  const auto&& imag_v = _mm512_permutex2var_pd(v1, idx_imag, v2);
  return std::make_pair(real_v, imag_v);
}

__attribute__((always_inline, const)) inline constexpr static auto interleave_vectors(const __m512d v1, const __m512d v2) noexcept {
  // Create index vectors for interleaving
  static constexpr const __m512i idx_real{0, 8, 1, 9, 2, 10, 3, 11};
  static constexpr const __m512i idx_imag{4, 12, 5, 13, 6, 14, 7, 15};
  // Separate real and imaginary parts
  const auto&& real_v = _mm512_permutex2var_pd(v1, idx_real, v2);
  const auto&& imag_v = _mm512_permutex2var_pd(v1, idx_imag, v2);
  return std::make_pair(real_v, imag_v);
}

__attribute__((always_inline)) inline static auto load(std::complex<double> *const __restrict__ a) {
  // Cast pointers to double*
  const auto *const a0_ptr = reinterpret_cast<const double *>(a);
  const auto *const a1_ptr = reinterpret_cast<const double *>(a + 4);

  // Load arrays into AVX-512 registers
  const auto&& va0 = _mm512_loadu_pd(a0_ptr);
  const auto&& va1 = _mm512_loadu_pd(a1_ptr);

  return std::make_pair(va0, va1);
}

__attribute__((always_inline)) inline constexpr static auto load_and_separate(std::complex<double> *__restrict__ a) {
  const auto [va0, va1] = load(a);
  // Separate real and imaginary parts
  const auto&& [real, imag] = separate_real_imaginary(va0, va1);
  return std::make_pair(real, imag);
}

__attribute__((always_inline, const)) inline constexpr static auto set_vector_to_complex(const std::complex<double> a) {
  return std::make_pair(_mm512_set1_pd(a.real()), _mm512_set1_pd(a.imag()));
}

__attribute__((always_inline, const)) inline constexpr static auto complex_mul_avx512(const __m512d real0, const __m512d imag0, const __m512d real1,
                                                                     const __m512d imag1) {
  // (a+ib) * (c+id) = (ac - bd) + i(ad + bc)
  // (real0 + i*imag0) * (real1 + i*imag1) = (real0*real1 - imag0*imag1) + i*(real0*imag1 + imag0*real1)
  // Multiply complex numbers
  const auto&& real = _mm512_fmsub_pd(real0, real1, _mm512_mul_pd(imag0, imag1));
  const auto&& imag = _mm512_fmadd_pd(real0, imag1, _mm512_mul_pd(imag0, real1));
  // Interleave real and imaginary parts
  return std::make_pair(real, imag);
}

#endif