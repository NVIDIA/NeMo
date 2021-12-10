// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Find shortest path(s) in an FST.

#include <cstring>

#include <memory>
#include <string>
#include <vector>

#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/shortest-path.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_int32(nshortest, 1, "Return N-shortest paths");
DEFINE_bool(unique, false, "Return unique strings");
DEFINE_string(weight, "", "Weight threshold");
DEFINE_int64(nstate, fst::kNoStateId, "State number threshold");
DEFINE_string(queue_type, "auto",
              "Queue type: one of \"auto\", "
              "\"fifo\", \"lifo\", \"shortest\', \"state\", \"top\"");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::WeightClass;
  using fst::script::VectorFstClass;

  string usage = "Finds shortest path(s) in an FST.\n\n  Usage: ";
  usage += argv[0];
  usage += " [in.fst [out.fst]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name =
      (argc > 1 && (strcmp(argv[1], "-") != 0)) ? argv[1] : "";
  const string out_name = argc > 2 ? argv[2] : "";

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  const auto weight_threshold =
      FLAGS_weight.empty() ? WeightClass::Zero(ifst->WeightType())
                           : WeightClass(ifst->WeightType(), FLAGS_weight);

  VectorFstClass ofst(ifst->ArcType());

  fst::QueueType queue_type;
  if (!s::GetQueueType(FLAGS_queue_type, &queue_type)) {
    LOG(ERROR) << "Unknown or unsupported queue type: " << FLAGS_queue_type;
    return 1;
  }

  s::ShortestPathOptions opts(queue_type, FLAGS_nshortest, FLAGS_unique, false,
                              FLAGS_delta, false, weight_threshold,
                              FLAGS_nstate);

  std::vector<WeightClass> distance;

  s::ShortestPath(*ifst, &ofst, &distance, opts);

  return !ofst.Write(out_name);
}
