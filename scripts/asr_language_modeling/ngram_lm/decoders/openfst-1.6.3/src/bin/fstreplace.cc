// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Performs the dynamic replacement of arcs in one FST with another FST,
// allowing for the definition of FSTs analogous to RTNs.

#include <cstring>

#include <string>
#include <vector>

#include <fst/script/getters.h>
#include <fst/script/replace.h>

DEFINE_string(call_arc_labeling, "input",
              "Which labels to make non-epsilon on the call arc. "
              "One of: \"input\" (default), \"output\", \"both\", \"neither\"");
DEFINE_string(return_arc_labeling, "neither",
              "Which labels to make non-epsilon on the return arc. "
              "One of: \"input\", \"output\", \"both\", \"neither\" (default)");
DEFINE_int64(return_label, 0, "Label to put on return arc");
DEFINE_bool(epsilon_on_replace, false, "Call/return arcs are epsilon arcs?");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::script::VectorFstClass;
  using fst::ReplaceLabelType;

  string usage = "Recursively replaces FST arcs with other FST(s).\n\n"
                 "  Usage: ";
  usage += argv[0];
  usage += " root.fst rootlabel [rule1.fst label1 ...] [out.fst]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc < 4) {
    ShowUsage();
    return 1;
  }

  const string in_name = argv[1];
  const string out_name = argc % 2 == 0 ? argv[argc - 1] : "";

  // Replace takes ownership of the pointer of FST arrays, deleting all such
  // pointers when the underlying ReplaceFst is destroyed.
  auto *ifst = FstClass::Read(in_name);
  if (!ifst) return 1;

  std::vector<s::LabelFstClassPair> pairs;
  // Note that if the root label is beyond the range of the underlying FST's
  // labels, truncation will occur.
  const auto root = atoll(argv[2]);
  pairs.emplace_back(root, ifst);

  for (auto i = 3; i < argc - 1; i += 2) {
    ifst = FstClass::Read(argv[i]);
    if (!ifst) return 1;
    // Note that if the root label is beyond the range of the underlying FST's
    // labels, truncation will occur.
    const auto label = atoll(argv[i + 1]);
    pairs.emplace_back(label, ifst);
  }

  ReplaceLabelType call_label_type;
  if (!s::GetReplaceLabelType(FLAGS_call_arc_labeling, FLAGS_epsilon_on_replace,
                              &call_label_type)) {
    LOG(ERROR) << argv[0] << ": Unknown or unsupported call arc replace "
               << "label type: " << FLAGS_call_arc_labeling;
  }
  ReplaceLabelType return_label_type;
  if (!s::GetReplaceLabelType(FLAGS_return_arc_labeling,
                              FLAGS_epsilon_on_replace, &return_label_type)) {
    LOG(ERROR) << argv[0] << ": Unknown or unsupported return arc replace "
               << "label type: " << FLAGS_return_arc_labeling;
  }

  s::ReplaceOptions opts(root, call_label_type, return_label_type,
                         FLAGS_return_label);

  VectorFstClass ofst(ifst->ArcType());
  s::Replace(pairs, &ofst, opts);

  return !ofst.Write(out_name);
}
