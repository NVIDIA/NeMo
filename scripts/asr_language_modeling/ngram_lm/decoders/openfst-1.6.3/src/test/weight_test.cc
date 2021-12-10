// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Regression test for FST weights.

#include <cstdlib>
#include <ctime>

#include <fst/log.h>
#include <fst/expectation-weight.h>
#include <fst/float-weight.h>
#include <fst/lexicographic-weight.h>
#include <fst/power-weight.h>
#include <fst/product-weight.h>
#include <fst/signed-log-weight.h>
#include <fst/sparse-power-weight.h>
#include <fst/string-weight.h>
#include <fst/union-weight.h>
#include "./weight-tester.h"

DEFINE_int32(seed, -1, "random seed");
DEFINE_int32(repeat, 10000, "number of test repetitions");

namespace {

using fst::Adder;
using fst::ExpectationWeight;
using fst::GALLIC;
using fst::GallicWeight;
using fst::LexicographicWeight;
using fst::LogWeight;
using fst::LogWeightTpl;
using fst::MinMaxWeight;
using fst::MinMaxWeightTpl;
using fst::NaturalLess;
using fst::PowerWeight;
using fst::ProductWeight;
using fst::SignedLogWeight;
using fst::SignedLogWeightTpl;
using fst::SparsePowerWeight;
using fst::StringWeight;
using fst::STRING_LEFT;
using fst::STRING_RIGHT;
using fst::TropicalWeight;
using fst::TropicalWeightTpl;
using fst::UnionWeight;
using fst::WeightGenerate;
using fst::WeightTester;

template <class T>
void TestTemplatedWeights(int repeat) {
  using TropicalWeightGenerate = WeightGenerate<TropicalWeightTpl<T>>;
  TropicalWeightGenerate tropical_generate;
  WeightTester<TropicalWeightTpl<T>, TropicalWeightGenerate> tropical_tester(
      tropical_generate);
  tropical_tester.Test(repeat);

  using LogWeightGenerate = WeightGenerate<LogWeightTpl<T>>;
  LogWeightGenerate log_generate;
  WeightTester<LogWeightTpl<T>, LogWeightGenerate> log_tester(log_generate);
  log_tester.Test(repeat);

  using MinMaxWeightGenerate = WeightGenerate<MinMaxWeightTpl<T>>;
  MinMaxWeightGenerate minmax_generate(true);
  WeightTester<MinMaxWeightTpl<T>, MinMaxWeightGenerate> minmax_tester(
      minmax_generate);
  minmax_tester.Test(repeat);

  using SignedLogWeightGenerate = WeightGenerate<SignedLogWeightTpl<T>>;
  SignedLogWeightGenerate signedlog_generate;
  WeightTester<SignedLogWeightTpl<T>, SignedLogWeightGenerate>
      signedlog_tester(signedlog_generate);
  signedlog_tester.Test(repeat);
}

template <class Weight>
void TestAdder(int n) {
  Weight sum = Weight::Zero();
  Adder<Weight> adder;
  for (int i = 0; i < n; ++i) {
    sum = Plus(sum, Weight::One());
    adder.Add(Weight::One());
  }
  CHECK(ApproxEqual(sum, adder.Sum()));
}

template <class Weight>
void TestSignedAdder(int n) {
  Weight sum = Weight::Zero();
  Adder<Weight> adder;
  const Weight minus_one = Minus(Weight::Zero(), Weight::One());
  for (int i = 0; i < n; ++i) {
    if (i < n/4 || i > 3*n/4) {
      sum = Plus(sum, Weight::One());
      adder.Add(Weight::One());
    } else {
      sum = Minus(sum, Weight::One());
      adder.Add(minus_one);
    }
  }
  CHECK(ApproxEqual(sum, adder.Sum()));
}

}  // namespace

int main(int argc, char **argv) {
  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(argv[0], &argc, &argv, true);

  LOG(INFO) << "Seed = " << FLAGS_seed;
  srand(FLAGS_seed);

  TestTemplatedWeights<float>(FLAGS_repeat);
  TestTemplatedWeights<double>(FLAGS_repeat);
  FLAGS_fst_weight_parentheses = "()";
  TestTemplatedWeights<float>(FLAGS_repeat);
  TestTemplatedWeights<double>(FLAGS_repeat);
  FLAGS_fst_weight_parentheses = "";

  // Makes sure type names for templated weights are consistent.
  CHECK(TropicalWeight::Type() == "tropical");
  CHECK(TropicalWeightTpl<double>::Type() != TropicalWeightTpl<float>::Type());
  CHECK(LogWeight::Type() == "log");
  CHECK(LogWeightTpl<double>::Type() != LogWeightTpl<float>::Type());
  TropicalWeightTpl<double> w(15.0);
  TropicalWeight tw(15.0);

  TestAdder<TropicalWeight>(1000);
  TestAdder<LogWeight>(1000);
  TestSignedAdder<SignedLogWeight>(1000);

  return 0;

  using LeftStringWeight = StringWeight<int>;
  using LeftStringWeightGenerate = WeightGenerate<LeftStringWeight>;
  LeftStringWeightGenerate left_string_generate;
  WeightTester<LeftStringWeight, LeftStringWeightGenerate> left_string_tester(
      left_string_generate);
  left_string_tester.Test(FLAGS_repeat);

  using RightStringWeight = StringWeight<int, STRING_RIGHT>;
  using RightStringWeightGenerate = WeightGenerate<RightStringWeight>;
  RightStringWeightGenerate right_string_generate;
  WeightTester<RightStringWeight, RightStringWeightGenerate>
      right_string_tester(right_string_generate);
  right_string_tester.Test(FLAGS_repeat);

  // COMPOSITE WEIGHTS AND TESTERS - DEFINITIONS

  using TropicalGallicWeight = GallicWeight<int, TropicalWeight>;
  using TropicalGallicWeightGenerate = WeightGenerate<TropicalGallicWeight>;
  TropicalGallicWeightGenerate tropical_gallic_generate(true);
  WeightTester<TropicalGallicWeight, TropicalGallicWeightGenerate>
      tropical_gallic_tester(tropical_gallic_generate);

  using TropicalGenGallicWeight = GallicWeight<int, TropicalWeight, GALLIC>;
  using TropicalGenGallicWeightGenerate =
      WeightGenerate<TropicalGenGallicWeight>;
  TropicalGenGallicWeightGenerate tropical_gen_gallic_generate(false);
  WeightTester<TropicalGenGallicWeight, TropicalGenGallicWeightGenerate>
      tropical_gen_gallic_tester(tropical_gen_gallic_generate);

  using TropicalProductWeight = ProductWeight<TropicalWeight, TropicalWeight>;
  using TropicalProductWeightGenerate = WeightGenerate<TropicalProductWeight>;
  TropicalProductWeightGenerate tropical_product_generate;
  WeightTester<TropicalProductWeight, TropicalProductWeightGenerate>
      tropical_product_tester(tropical_product_generate);

  using TropicalLexicographicWeight =
      LexicographicWeight<TropicalWeight, TropicalWeight>;
  using TropicalLexicographicWeightGenerate =
      WeightGenerate<TropicalLexicographicWeight>;
  TropicalLexicographicWeightGenerate tropical_lexicographic_generate;
  WeightTester<TropicalLexicographicWeight,
               TropicalLexicographicWeightGenerate>
      tropical_lexicographic_tester(tropical_lexicographic_generate);

  using TropicalCubeWeight = PowerWeight<TropicalWeight, 3>;
  using TropicalCubeWeightGenerate = WeightGenerate<TropicalCubeWeight>;
  TropicalCubeWeightGenerate tropical_cube_generate;
  WeightTester<TropicalCubeWeight, TropicalCubeWeightGenerate>
      tropical_cube_tester(tropical_cube_generate);

  using FirstNestedProductWeight =
      ProductWeight<TropicalProductWeight, TropicalWeight>;
  using FirstNestedProductWeightGenerate =
      WeightGenerate<FirstNestedProductWeight>;
  FirstNestedProductWeightGenerate first_nested_product_generate;
  WeightTester<FirstNestedProductWeight, FirstNestedProductWeightGenerate>
      first_nested_product_tester(first_nested_product_generate);

  using SecondNestedProductWeight =
      ProductWeight<TropicalWeight, TropicalProductWeight>;
  using SecondNestedProductWeightGenerate =
      WeightGenerate<SecondNestedProductWeight>;
  SecondNestedProductWeightGenerate second_nested_product_generate;
  WeightTester<SecondNestedProductWeight, SecondNestedProductWeightGenerate>
      second_nested_product_tester(second_nested_product_generate);

  using NestedProductCubeWeight = PowerWeight<FirstNestedProductWeight, 3>;
  using NestedProductCubeWeightGenerate =
      WeightGenerate<NestedProductCubeWeight>;
  NestedProductCubeWeightGenerate nested_product_cube_generate;
  WeightTester<NestedProductCubeWeight, NestedProductCubeWeightGenerate>
      nested_product_cube_tester(nested_product_cube_generate);

  using SparseNestedProductCubeWeight =
      SparsePowerWeight<NestedProductCubeWeight, size_t>;
  using SparseNestedProductCubeWeightGenerate =
      WeightGenerate<SparseNestedProductCubeWeight>;
  SparseNestedProductCubeWeightGenerate sparse_nested_product_cube_generate;
  WeightTester<SparseNestedProductCubeWeight,
               SparseNestedProductCubeWeightGenerate>
      sparse_nested_product_cube_tester(sparse_nested_product_cube_generate);

  using LogSparsePowerWeight = SparsePowerWeight<LogWeight, size_t>;
  using LogSparsePowerWeightGenerate = WeightGenerate<LogSparsePowerWeight>;
  LogSparsePowerWeightGenerate log_sparse_power_generate;
  WeightTester<LogSparsePowerWeight, LogSparsePowerWeightGenerate>
      log_sparse_power_tester(log_sparse_power_generate);

  using LogLogExpectationWeight = ExpectationWeight<LogWeight, LogWeight>;
  using LogLogExpectationWeightGenerate =
      WeightGenerate<LogLogExpectationWeight>;
  LogLogExpectationWeightGenerate log_log_expectation_generate;
  WeightTester<LogLogExpectationWeight, LogLogExpectationWeightGenerate>
      log_log_expectation_tester(log_log_expectation_generate);

  using LogLogSparseExpectationWeight =
      ExpectationWeight<LogWeight, LogSparsePowerWeight>;
  using LogLogSparseExpectationWeightGenerate =
      WeightGenerate<LogLogSparseExpectationWeight>;
  LogLogSparseExpectationWeightGenerate log_log_sparse_expectation_generate;
  WeightTester<LogLogSparseExpectationWeight,
               LogLogSparseExpectationWeightGenerate>
      log_log_sparse_expectation_tester(log_log_sparse_expectation_generate);

  struct UnionWeightOptions {
    using Compare = NaturalLess<TropicalWeight>;

    struct Merge {
      TropicalWeight operator()(const TropicalWeight &w1,
                                const TropicalWeight &w2) const {
        return w1;
      }
    };

    using ReverseOptions = UnionWeightOptions;
  };

  using TropicalUnionWeight = UnionWeight<TropicalWeight, UnionWeightOptions>;
  using TropicalUnionWeightGenerate = WeightGenerate<TropicalUnionWeight>;
  TropicalUnionWeightGenerate tropical_union_generate;
  WeightTester<TropicalUnionWeight, TropicalUnionWeightGenerate>
      tropical_union_tester(tropical_union_generate);

  // COMPOSITE WEIGHTS AND TESTERS - TESTING

  // Tests composite weight I/O with parentheses.
  FLAGS_fst_weight_parentheses = "()";

  // Unnested composite.
  tropical_gallic_tester.Test(FLAGS_repeat);
  tropical_gen_gallic_tester.Test(FLAGS_repeat);
  tropical_product_tester.Test(FLAGS_repeat);
  tropical_lexicographic_tester.Test(FLAGS_repeat);
  tropical_cube_tester.Test(FLAGS_repeat);
  log_sparse_power_tester.Test(FLAGS_repeat);
  log_log_expectation_tester.Test(FLAGS_repeat, false);
  tropical_union_tester.Test(FLAGS_repeat, false);

  // Nested composite.
  first_nested_product_tester.Test(FLAGS_repeat);
  second_nested_product_tester.Test(5);
  nested_product_cube_tester.Test(FLAGS_repeat);
  sparse_nested_product_cube_tester.Test(FLAGS_repeat);
  log_log_sparse_expectation_tester.Test(FLAGS_repeat, false);

  // ... and tests composite weight I/O without parentheses.
  FLAGS_fst_weight_parentheses = "";

  // Unnested composite.
  tropical_gallic_tester.Test(FLAGS_repeat);
  tropical_product_tester.Test(FLAGS_repeat);
  tropical_lexicographic_tester.Test(FLAGS_repeat);
  tropical_cube_tester.Test(FLAGS_repeat);
  log_sparse_power_tester.Test(FLAGS_repeat);
  log_log_expectation_tester.Test(FLAGS_repeat, false);
  tropical_union_tester.Test(FLAGS_repeat, false);

  // Nested composite.
  second_nested_product_tester.Test(FLAGS_repeat);
  log_log_sparse_expectation_tester.Test(FLAGS_repeat, false);

  std::cout << "PASS" << std::endl;

  return 0;
}
