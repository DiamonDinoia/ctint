#include <triqs_ctint/measures/intrinsics.h>

#if USE_INTRINSICS == 1

#include <gtest/gtest.h>
#include <complex>
#include <array>
#include <random>
#include <algorithm>

template <typename T, int N>
struct VecType {
  using type = T;
  static constexpr int size = N;
};

#ifdef __AVX512F__
using MyTypes = ::testing::Types<VecType<float, 2>, VecType<float, 4>, VecType<float, 8>, VecType<float, 16>, VecType<double, 2>, VecType<double, 4>,
                                 VecType<double, 8>>;
#elifdef __AVX2__
using MyTypes = ::testing::Types<VecType<float, 2>, VecType<float, 4>, VecType<float, 8>, VecType<double, 2>, VecType<double, 4>>;
#elifdef __SSE4_2__
using MyTypes = ::testing::Types<VecType<float, 2>, VecType<float, 4>, VecType<double, 2>>;
#endif

template <typename VecType>
class IntrinsicsTest : public ::testing::Test {
  public:
  using T = typename VecType::type;
  static constexpr int N = VecType::size;
  std::array<std::complex<T>, N/2> a{}, b{}, c{}, d{};
  std::array<std::complex<T>, N> ab{};
  std::array<std::complex<T>, N> cd{};
  Vec<T, N> v_a{}, v_b{}, v_c{}, v_d{};
  std::mt19937_64 gen{};
  std::uniform_real_distribution<T> dis{0.0, 1.0};

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

    for (int i = 0; i < N/2; ++i) {
      v_a[2*i] = a[i].real();
      v_a[2*i + 1] = a[i].imag();
      v_b[2*i] = b[i].real();
      v_b[2*i + 1] = b[i].imag();
      v_c[2*i] = c[i].real();
      v_c[2*i + 1] = c[i].imag();
      v_d[2*i] = d[i].real();
      v_d[2*i + 1] = d[i].imag();
      ab[2*i] = a[i];
      ab[2*i + 1] = b[i];
      cd[2*i] = c[i];
      cd[2*i + 1] = d[i];
    }
  }
};

std::string print_vector(const auto &v) {
  static constexpr auto N = sizeof(decltype(v))/sizeof(decltype(v[0]));
  std::string str = "{";
  for (int i = 0; i < N; ++i) {
    str += std::to_string(v[i]) + ", ";
  }
  str += "}";
  return str;
}
TYPED_TEST_SUITE(IntrinsicsTest, MyTypes);

TYPED_TEST(IntrinsicsTest, SeparateRealImaginary) {
  static constexpr auto N = TypeParam::size;
  static constexpr auto N2 = N/2;
  auto [real, imag] = separate_real_imaginary(this->v_a, this->v_b);
  std::string v_str = "\n";
  v_str += "v_a = " + print_vector(this->v_a) + "\n";
  v_str += "v_b = " + print_vector(this->v_b) + "\n";
  v_str += "rea = " + print_vector(real) + "\n";
  v_str += "ima = " + print_vector(imag) + "\n";
  SCOPED_TRACE(v_str);
  for (int i = 0; i <N2; ++i) {
    EXPECT_DOUBLE_EQ(this->a[i].real(), real[i]);
    EXPECT_DOUBLE_EQ(this->a[i].imag(), imag[i]);
    EXPECT_DOUBLE_EQ(this->b[i].real(), real[i + N2]);
    EXPECT_DOUBLE_EQ(this->b[i].imag(), imag[i + N2]);
  }
  // Add your assertions here
}

TYPED_TEST(IntrinsicsTest, InterleaveVectors) {
  static constexpr auto N = TypeParam::size;
  auto [interleave_v1, interleave_v2] = interleave_vectors(this->v_a, this->v_b);

  // print v_a and v_b and interleave_v1 and interleave_v2
  std::string v_str = "\n";
  v_str += "v_a = " + print_vector(this->v_a) + "\n";
  v_str += "v_b = " + print_vector(this->v_b) + "\n";
  v_str += "v_1 = " + print_vector(interleave_v1) + "\n";
  v_str += "v_2 = " + print_vector(interleave_v2) + "\n";
  SCOPED_TRACE(v_str);

  for (int i = 0; i < N; ++i) {
    SCOPED_TRACE("i: " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(interleave_v1[i], this->v_a[i>>1]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(interleave_v1[i], this->v_b[i>>1]); }
  }
  for (int i = N; i < 2*N; ++i) {
    SCOPED_TRACE("i: " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(interleave_v2[i-N], this->v_a[i>>1]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(interleave_v2[i-N], this->v_b[i>>1]); }
  }
  // Add your assertions here
}

TYPED_TEST(IntrinsicsTest, Load) {
  using T = TypeParam::type;
  static constexpr auto N = TypeParam::size;
  auto [va0, va1] = load<T, N>(this->ab.data());
  std::string v_str = "\n";
  v_str += "v_a = " + print_vector(va0) + "\n";
  v_str += "v_b = " + print_vector(va1) + "\n";
  v_str += "ab = {";
  for (const auto elem : this->ab) {
    v_str += std::to_string(elem.real()) + ", " + std::to_string(elem.imag()) + ", ";
  }
  v_str += "}";
  SCOPED_TRACE(v_str);
  for (int i = 0; i < N; ++i) {
    SCOPED_TRACE("i : " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(this->ab[i>>1].real(), va0[i]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(this->ab[i>>1].imag(), va0[i]); }
  }
  for (int i = N; i < N*2; ++i) {
    SCOPED_TRACE("i : " + std::to_string(i));
    if (i % 2 == 0) { EXPECT_DOUBLE_EQ(this->ab[i>>1].real(), va1[i-N]); }
    if (i % 2 == 1) { EXPECT_DOUBLE_EQ(this->ab[i>>1].imag(), va1[i-N]); }
  }
}

TYPED_TEST(IntrinsicsTest, Store) {
  using T = TypeParam::type;
  static constexpr auto N = TypeParam::size;
  std::array<T, N> data{};
  store(data.data(), this->v_a);
  std::string v_str = "\n";
  v_str += "v_a = " + print_vector(this->v_a) + "\n";
  v_str += "data = " + print_vector(data) + "\n";
  SCOPED_TRACE(v_str);
  for (int i = 0; i < N; ++i) {
    SCOPED_TRACE("i : " + std::to_string(i));
    EXPECT_DOUBLE_EQ(this->v_a[i], data[i]);
  }
}

TYPED_TEST(IntrinsicsTest, LoadAndSeparate) {
  using T = TypeParam::type;
  static constexpr auto N = TypeParam::size;
  static constexpr auto N2 = N/2;
  auto [real, imag] = load_and_separate<T, N>(this->a.data());
  for (int i = 0; i < N2; ++i) {
    EXPECT_DOUBLE_EQ(this->a[i].real(), real[i]);
    EXPECT_DOUBLE_EQ(this->a[i].imag(), imag[i]);
  }
}

TYPED_TEST(IntrinsicsTest, ComplexMulAvx512) {
  using T = TypeParam::type;
  static constexpr auto N = TypeParam::size;
  auto [real0, imag0] = load_and_separate<T, N>(this->ab.data());
  auto [real1, imag1] = load_and_separate<T, N>(this->cd.data());
  auto [real, imag] = complex_mul(real0, imag0, real1, imag1);
  for (int i = 0; i < N; ++i) {
    EXPECT_NEAR((this->ab[i]*this->cd[i]).real(), real[i], 2*std::numeric_limits<T>::epsilon());
    EXPECT_NEAR((this->ab[i]*this->cd[i]).imag(), imag[i], 2*std::numeric_limits<T>::epsilon());
  }
}

TYPED_TEST(IntrinsicsTest, SetVectorToValue) {
  using T = TypeParam::type;
  static constexpr auto N = TypeParam::size;
  if constexpr (std::is_same_v<T, double>) {
    std::complex<double> value{this->dis(this->gen), this->dis(this->gen)};
    auto [real, imag] = set_vector_to_complex<Vec<T, N>>(value);
    for (int i = 0; i < TypeParam::size; ++i) {
      EXPECT_DOUBLE_EQ(real[i], value.real());
      EXPECT_DOUBLE_EQ(imag[i], value.imag());
    }
  }
}


#else
int main(){
  return 0;
}
#endif