#pragma once

#include <iostream>

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_2__)
#include <immintrin.h>
#include <utility>
#include <complex>

template <class T, std::uint8_t elems> using Vec = T __attribute__((vector_size(elems * sizeof(T))));

/**
 * @brief Get the widest simd type for the current architecture
 * Currently only supports AVX512, AVX2 and SSE4.2 architectures
 */
template<class T>
static constexpr uint widest_simd() noexcept {
  if constexpr (__AVX512F__) {
    return 512U/sizeof(T);
  }
  if constexpr (__AVX2__) {
    return 256U/sizeof(T);
  }
  if constexpr (__SSE4_2__) {
    return 128U/sizeof(T);
  }
  return 1U;
}

/**
 *  This selects the indices I ... from v1 and v2
 *  The indices are from 0 to 2N where N is the number of elements in the vector
 *  if i < N then return v1[i] else return v2[i - N]
 */
template <std::size_t... I>
__attribute__((always_inline)) inline static constexpr auto permute_vectors(std::index_sequence<I...>, const auto v1, const auto v2) noexcept {
  // I  -> [0, 2N] where N is the number of elements in the vector
  // if i < N then return v1[i] else return v2[i - N]
  return __builtin_shufflevector(v1, v2, I...);

}

/**
 * @brief Separate real and imaginary parts of a complex vector
 * @param v1 vector containing real parts
 * @param v2 vector containing imaginary parts
 * @return std::pair<Vec<T, N>, Vec<T, N>> real and imaginary parts
 */
__attribute__((always_inline)) inline static constexpr auto separate_real_imaginary(const auto &v1, const auto &v2) noexcept {
  // Create index vectors for separating real and imaginary parts using fold expressions
  // it returns 0, 2, 4, 6, 8, 10, 12, 14 or 1, 3, 5, 7, 9, 11, 13, 15
  static constexpr auto generate_indices = []<std::size_t N, bool odd>() constexpr {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return std::index_sequence<(2 * I + static_cast<int>(odd))...>{};
    }(std::make_index_sequence<N>{});
  };
  static constexpr auto N            = sizeof(decltype(v1)) / sizeof(decltype(v1[0]));
  static constexpr auto even_indices = generate_indices.template operator()<N, false>();
  static constexpr auto odd_indices  = generate_indices.template operator()<N, true>();
  // Separate real and imaginary parts using the index vectors created above
  const auto &&real_v                = permute_vectors(even_indices, v1, v2);
  const auto &&imag_v                = permute_vectors(odd_indices, v1, v2);
  return std::make_pair(real_v, imag_v);
}

__attribute__((always_inline)) inline constexpr static auto interleave_vectors(const auto real, const auto imag) noexcept {
  // Create index vectors for interleaving real and imaginary parts using fold expressions
  // it returns 0, N, 1, N+1, 2, N+2, 3, N+3 and so on
  // S is the starting index
  // N is the number of elements in the vector
  // add_even is the offset for even indices
  // add_odd is the offset for odd indices
  // this is more general than what needed here as the offsets are always 0 and N respectively
  static constexpr auto generate_indices = []<std::size_t S, std::size_t N, std::size_t add_even, std::size_t add_odd>() constexpr {
    return []<std::size_t... I>(std::index_sequence<I...>) {
      return std::index_sequence<(I % 2 == 0 ? S + I / 2 + add_even : S + I / 2 + add_odd)...>{};
    }(std::make_index_sequence<N>{});
  };
  static constexpr auto N = sizeof(decltype(real)) / sizeof(decltype(real[0]));
  //{0, 8, 1, 9, 2, 10, 3, 11}
  static constexpr auto idx_real = generate_indices.template operator()<0 / 1, N, 0, N>();
  //{4, 12, 5, 13, 6, 14, 7, 15}
  static constexpr auto idx_imag = generate_indices.template operator()<N / 2, N, 0, N>();
  // Separate real and imaginary parts
  const auto &&res1 = permute_vectors(idx_real, real, imag);
  const auto &&res2 = permute_vectors(idx_imag, real, imag);
  return std::make_pair(res1, res2);
}
// Load two vectors of complex numbers
// it uses memcpy to load the vectors to avoid reinterpret_cast
// drawback is that it is not constexpr
// compilers are really good at optimizing memcpy not worth using union puns
// TODO: this can be more general, loading complex numbers of any size
template <typename T, const size_t elems> __attribute__((always_inline)) inline static auto load(const std::complex<T> *const __restrict__ a) {
  Vec<T, elems> va0{};
  Vec<T, elems> va1{};
  std::memcpy(&va0,  a, sizeof(T) * elems);
  std::memcpy(&va1, a + elems / 2, sizeof(T) * elems);
  return std::make_pair(va0, va1);
}

// Store two vectors of complex numbers
// it uses memcpy to store the vectors to avoid reinterpret_cast
// drawback is that it is not constexpr
// compilers are really good at optimizing memcpy not worth using union puns
__attribute__((always_inline)) inline static void store(auto*  __restrict__ ptr, const auto &v) {
  std::memcpy(ptr, &v, sizeof(decltype(v)));
}



// load two vectors of complex numbers and separate real and imaginary parts
template<class T, size_t elems>
__attribute__((always_inline)) inline constexpr static auto load_and_separate(const std::complex<T> *const __restrict__ a) {
  const auto [va0, va1] = load<T, elems>(a);
  // Separate real and imaginary parts
  const auto [real, imag] = separate_real_imaginary(va0, va1);
  return std::make_pair(real, imag);
}

// initialize two vectors of real numbers from a complex number
// one vector for real part and one for imaginary part
template <typename T> __attribute__((always_inline)) inline constexpr static auto set_vector_to_complex(const std::complex<double> a) {
  T real{};
  T imag{};
  for (size_t i = 0; i < sizeof(T) / sizeof(double); i++) {
    real[i] = a.real();
    imag[i] = a.imag();
  }
  return std::make_pair(real, imag);
}

// complex multiplication vectorized
// this is the same code that std::complex uses for multiplication when -fcx-limited-range is enabled
__attribute__((always_inline)) inline constexpr static auto complex_mul(const auto real0, const auto imag0, const auto real1, const auto imag1) {
  // check if the compiler is clang or gcc
  // (a+ib) * (c+id) = (ac - bd) + i(ad + bc)
  // (real0 + i*imag0) * (real1 + i*imag1) = (real0*real1 - imag0*imag1) + i*(real0*imag1 + imag0*real1)
  // Multiply complex numbers
  // Equivalent to normal multiplication but std library uses this algorithm.
  // Benchmark: https://quick-bench.com/q/OV56ZJWeLE7nHQxRNFXO0-kGTtU
#ifdef ENABLE_KARASUBA
   auto ac = real0 * real1;
   auto bd = imag0 * imag1;
   auto &&real = ac - bd;
   // TODO: __builtin_elementwise_fma is clang specific
   auto &&imag = __builtin_elementwise_fma(real0 + imag0, real1 + imag1, - ac - bd);
#else
   auto &&real = __builtin_elementwise_fma(real0, real1, -imag0 * imag1);
   auto &&imag = __builtin_elementwise_fma(real0, imag1, imag0 * real1);
#endif
   return std::make_pair(real, imag);
}

#endif