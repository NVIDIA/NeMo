// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/disambiguate.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1: Full signature.
void Disambiguate(const FstClass &ifst, MutableFstClass *ofst,
                  const DisambiguateOptions &opts) {
  if (!ArcTypesMatch(ifst, *ofst, "Disambiguate") ||
      !ofst->WeightTypesMatch(opts.weight_threshold, "Disambiguate")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DisambiguateArgs1 args(ifst, ofst, opts);
  Apply<Operation<DisambiguateArgs1>>("Disambiguate", ifst.ArcType(), &args);
}

// 2: Signature with default WeightClass argument.
void Disambiguate(const FstClass &ifst, MutableFstClass *ofst,
                  int64 n, int64 l) {
  if (!ArcTypesMatch(ifst, *ofst, "Disambiguate")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  DisambiguateArgs2 args(ifst, ofst, n, l);
  Apply<Operation<DisambiguateArgs2>>("Disambiguate", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(Disambiguate, StdArc, DisambiguateArgs1);
REGISTER_FST_OPERATION(Disambiguate, LogArc, DisambiguateArgs1);
REGISTER_FST_OPERATION(Disambiguate, Log64Arc, DisambiguateArgs1);

REGISTER_FST_OPERATION(Disambiguate, StdArc, DisambiguateArgs2);
REGISTER_FST_OPERATION(Disambiguate, LogArc, DisambiguateArgs2);
REGISTER_FST_OPERATION(Disambiguate, Log64Arc, DisambiguateArgs2);

}  // namespace script
}  // namespace fst
