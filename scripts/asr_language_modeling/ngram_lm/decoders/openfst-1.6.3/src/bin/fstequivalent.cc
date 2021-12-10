// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Two DFAs are equivalent iff their exit status is zero.

#include <unistd.h>

#include <climits>
#include <cstring>
#include <ctime>

#include <memory>
#include <string>

#include <fst/log.h>
#include <fst/script/equivalent.h>
#include <fst/script/getters.h>
#include <fst/script/randequivalent.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_bool(random, false,
            "Test equivalence by randomly selecting paths in the input FSTs");
DEFINE_int32(max_length, INT32_MAX, "Maximum path length");
DEFINE_int32(npath, 1, "Number of paths to generate");
DEFINE_int32(seed, time(nullptr) + getpid(), "Random seed");
DEFINE_string(select, "uniform",
              "Selection type: one of: "
              " \"uniform\", \"log_prob\" (when appropriate),"
              " \"fast_log_prob\" (when appropriate)");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;

  string usage =
      "Two DFAs are equivalent iff the exit status is zero.\n\n"
      "  Usage: ";
  usage += argv[0];
  usage += " in1.fst in2.fst\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc != 3) {
    ShowUsage();
    return 1;
  }

  const string in1_name = strcmp(argv[1], "-") == 0 ? "" : argv[1];
  const string in2_name = strcmp(argv[2], "-") == 0 ? "" : argv[2];

  if (in1_name.empty() && in2_name.empty()) {
    LOG(ERROR) << argv[0] << ": Can't take both inputs from standard input";
    return 1;
  }

  std::unique_ptr<FstClass> ifst1(FstClass::Read(in1_name));
  if (!ifst1) return 1;

  std::unique_ptr<FstClass> ifst2(FstClass::Read(in2_name));
  if (!ifst2) return 1;

  if (!FLAGS_random) {
    bool result = s::Equivalent(*ifst1, *ifst2, FLAGS_delta);
    if (!result) VLOG(1) << "FSTs are not equivalent";
    return result ? 0 : 2;
  } else {
    s::RandArcSelection ras;
    if (!s::GetRandArcSelection(FLAGS_select, &ras)) {
      LOG(ERROR) << argv[0] << ": Unknown or unsupported select type "
                            << FLAGS_select;
      return 1;
    }
    bool result = s::RandEquivalent(
        *ifst1, *ifst2, FLAGS_npath, FLAGS_delta, FLAGS_seed,
        fst::RandGenOptions<s::RandArcSelection>(ras, FLAGS_max_length));
    if (!result) VLOG(1) << "FSTs are not equivalent";
    return result ? 0 : 2;
  }
}
