#include <triqs_ctint/measures/intrinsics.h>
#include <gtest/gtest.h>
#include <complex>
#include <array>
#include <random>
#include <algorithm>

class IntrinsicsTest : public ::testing::Test {
  protected:
  std::array<std::complex<double>, 4> a, b, c, d;
  std::array<std::complex<double>, 8> ab;
  std::array<std::complex<double>, 8> cd;
  __m512d v_a, v_b, v_c, v_d;
  std::mt19937_64 gen;
  std::uniform_real_distribution<double> dis{0.0, 1.0};

  void SetUp() override {
    std::random_device rd;
    const auto seed = rd();
    SCOPED_TRACE("Seed: " + std::to_string(seed));
    gen.seed(seed);
    const auto generate_complex = [this] { return std::complex{dis(gen), dis(gen)}; };

    std::generate(a.begin(), a.end(), generate_complex);
    std::generate(b.begin(), b.end(), generate_complex);
    std::generate(c.begin(), c.end(), generate_complex);
    std::generate(d.begin(), d.end(), generate_complex);

    v_a = _mm512_setr_pd(a[0].real(), a[0].imag(), a[1].real(), a[1].imag(), a[2].real(), a[2].imag(), a[3].real(), a[3].imag());
    v_b = _mm512_setr_pd(b[0].real(), b[0].imag(), b[1].real(), b[1].imag(), b[2].real(), b[2].imag(), b[3].real(), b[3].imag());
    v_c = _mm512_setr_pd(c[0].real(), c[0].imag(), c[1].real(), c[1].imag(), c[2].real(), c[2].imag(), c[3].real(), c[3].imag());
    v_d = _mm512_setr_pd(d[0].real(), d[0].imag(), d[1].real(), d[1].imag(), d[2].real(), d[2].imag(), d[3].real(), d[3].imag());
    ab  = {a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]};
    cd  = {c[0], d[0], c[1], d[1], c[2], d[2], c[3], d[3]};
  }
};

std::string print_vector(__m512d v) {
  std::string str = "{";
  for (int i = 0; i < 8; ++i) {
    str += std::to_string(v[i]) + ", ";
  }
  str += "}";
  return str;
}

TEST_F(IntrinsicsTest, SeparateRealImaginary) {
  auto [real, imag] = separate_real_imaginary(v_a, v_b);
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(a[i].real(), real[i]);
    EXPECT_DOUBLE_EQ(a[i].imag(), imag[i]);
    EXPECT_DOUBLE_EQ(b[i].real(), real[i + 4]);
    EXPECT_DOUBLE_EQ(b[i].imag(), imag[i + 4]);
  }
  // Add your assertions here
}

TEST_F(IntrinsicsTest, InterleaveVectors) {
  auto [interleave_v1, interleave_v2] = interleave_vectors(v_a, v_b);

  // print v_a and v_b and interleave_v1 and interleave_v2
  std::string v_str = "\n";
  v_str += "v_a = " + print_vector(v_a) + "\n";
  v_str += "v_b = " + print_vector(v_b) + "\n";
  v_str += "v_1 = " + print_vector(interleave_v1) + "\n";
  v_str += "v_2 = " + print_vector(interleave_v2) + "\n";
  SCOPED_TRACE(v_str);

  for (int i = 0; i < 8; ++i) {
    SCOPED_TRACE("i: " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(interleave_v1[i], v_a[i>>1]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(interleave_v1[i], v_b[i>>1]); }
  }
  for (int i = 8; i < 16; ++i) {
    SCOPED_TRACE("i: " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(interleave_v2[i-8], v_a[i>>1]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(interleave_v2[i-8], v_b[i>>1]); }
  }
  // Add your assertions here
}

TEST_F(IntrinsicsTest, Load) {
  auto [va0, va1] = load(ab.data());
  std::string v_str = "\n";
  v_str += "v_a = " + print_vector(va0) + "\n";
  v_str += "v_b = " + print_vector(va1) + "\n";
  v_str += "ab = {";
  for (const auto elem : ab) {
    v_str += std::to_string(elem.real()) + ", " + std::to_string(elem.imag()) + ", ";
  }
  v_str += "}";
  SCOPED_TRACE(v_str);
  for (int i = 0; i < 8; ++i) {
    SCOPED_TRACE("i : " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(ab[i>>1].real(), va0[i]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(ab[i>>1].imag(), va0[i]); }
  }
  for (int i = 8; i < 16; ++i) {
    SCOPED_TRACE("i : " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(ab[i>>1].real(), va1[i-8]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(ab[i>>1].imag(), va1[i-8]); }
  }
  // Add your assertions here
}

TEST_F(IntrinsicsTest, LoadAndSeparate) {
  auto [real, imag] = load_and_separate(a.data());
  for (int i = 0; i < 4; ++i) {
    EXPECT_DOUBLE_EQ(a[i].real(), real[i]);
    EXPECT_DOUBLE_EQ(a[i].imag(), imag[i]);
  }
}

TEST_F(IntrinsicsTest, ComplexMulAvx512) {
  auto [real0, imag0] = load_and_separate(ab.data());
  auto [real1, imag1] = load_and_separate(cd.data());
  auto [real, imag] = complex_mul_avx512(real0, imag0, real1, imag1);
  for (int i = 0; i < 8; ++i) {
    EXPECT_DOUBLE_EQ((ab[i]*cd[i]).real(), real[i]);
    EXPECT_DOUBLE_EQ((ab[i]*cd[i]).imag(), imag[i]);
  }
}