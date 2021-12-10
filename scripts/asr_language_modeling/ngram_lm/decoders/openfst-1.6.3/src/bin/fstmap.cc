// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Applies an operation to each arc of an FST.

#include <cstring>

#include <memory>
#include <string>

#include <fst/log.h>
#include <fst/script/getters.h>
#include <fst/script/map.h>

DEFINE_double(delta, fst::kDelta, "Comparison/quantization delta");
DEFINE_string(map_type, "identity",
              "Map operation, one of: \"arc_sum\", \"arc_unique\", "
              "\"identity\", \"input_epsilon\", \"invert\", "
              "\"output_epsilon\", \"plus (--weight)\", "
              "\"quantize (--delta)\", \"rmweight\", \"superfinal\", "
              "\"times (--weight)\", \"to_log\", \"to_log64\", \"to_std\"");
DEFINE_string(weight, "", "Weight parameter");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::WeightClass;

  string usage = "Applies an operation to each arc of an FST.\n\n  Usage: ";
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

  s::MapType map_type;
  if (!s::GetMapType(FLAGS_map_type, &map_type)) {
    LOG(ERROR) << argv[0] << ": Unknown or unsupported map type "
               << FLAGS_map_type;
    return 1;
  }

  const auto weight_param =
      !FLAGS_weight.empty()
          ? WeightClass(ifst->WeightType(), FLAGS_weight)
          : (FLAGS_map_type == "times" ? WeightClass::One(ifst->WeightType())
                                       : WeightClass::Zero(ifst->WeightType()));

  std::unique_ptr<FstClass> ofst(s::Map(*ifst, map_type, FLAGS_delta,
                                        weight_param));

  return !ofst->Write(out_name);
}
