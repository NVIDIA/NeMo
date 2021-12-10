// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Prunes states and arcs of an FST w.r.t. the shortest path weight.

#include <cstring>

#include <memory>
#include <string>

#include <fst/script/prune.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_int64(nstate, fst::kNoStateId, "State number threshold");
DEFINE_string(weight, "", "Weight threshold");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::MutableFstClass;
  using fst::script::WeightClass;

  string usage = "Prunes states and arcs of an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<MutableFstClass> fst(MutableFstClass::Read(in_name, true));
  if (!fst) return 1;

  const auto weight_threshold =
      FLAGS_weight.empty() ? WeightClass::Zero(fst->WeightType())
                           : WeightClass(fst->WeightType(), FLAGS_weight);

  const s::PruneOptions opts(weight_threshold, FLAGS_nstate, nullptr,
                             FLAGS_delta);

  s::Prune(fst.get(), opts);

  return !fst->Write(out_name);
}
