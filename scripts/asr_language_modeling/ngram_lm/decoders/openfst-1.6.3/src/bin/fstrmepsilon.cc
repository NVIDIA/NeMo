// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Removes epsilons from an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/rmepsilon.h>

DEFINE_bool(connect, true, "Trim output");
DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_int64(nstate, fst::kNoStateId, "State number threshold");
DEFINE_bool(reverse, false, "Perform in the reverse direction");
DEFINE_string(weight, "", "Weight threshold");
DEFINE_string(queue_type, "auto",
              "Queue type: one of: \"auto\", "
              "\"fifo\", \"lifo\", \"shortest\", \"state\", \"top\"");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;
  using fst::script::WeightClass;

  string usage = "Removes epsilons from an FST.\n\n  Usage: ";
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

  std::unique_ptr<FstClass> ifst(FstClass::Read(in_name));
  if (!ifst) return 1;

  const auto weight_threshold =
      FLAGS_weight.empty() ? WeightClass::Zero(ifst->WeightType())
                           : WeightClass(ifst->WeightType(), FLAGS_weight);

  fst::QueueType queue_type;
  if (!s::GetQueueType(FLAGS_queue_type, &queue_type)) {
    LOG(ERROR) << argv[0]
               << ": Unknown or unsupported queue type: " << FLAGS_queue_type;
    return 1;
  }

  const s::RmEpsilonOptions opts(queue_type, FLAGS_delta, FLAGS_connect,
                                 weight_threshold, FLAGS_nstate);

  VectorFstClass ofst(ifst->ArcType());

  s::RmEpsilon(*ifst, &ofst, FLAGS_reverse, opts);

  return !ofst.Write(out_name);
}
