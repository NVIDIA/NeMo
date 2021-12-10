// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/rmepsilon.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1: Full signature with RmEpsilonOptions.
void RmEpsilon(const FstClass &ifst, MutableFstClass *ofst, bool reverse,
               const RmEpsilonOptions &opts) {
  if (!ArcTypesMatch(ifst, *ofst, "RmEpsilon") ||
      !ofst->WeightTypesMatch(opts.weight_threshold, "RmEpsilon")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  RmEpsilonArgs1 args(ifst, ofst, reverse, opts);
  Apply<Operation<RmEpsilonArgs1>>("RmEpsilon", ifst.ArcType(), &args);
}

// 2: Full signature with flat arguments.
void RmEpsilon(MutableFstClass *fst, bool connect,
               const WeightClass &weight_threshold, int64 state_threshold,
               float delta) {
  if (!fst->WeightTypesMatch(weight_threshold, "RmEpsilon")) {
    fst->SetProperties(kError, kError);
    return;
  }
  RmEpsilonArgs2 args(fst, connect, weight_threshold, state_threshold, delta);
  Apply<Operation<RmEpsilonArgs2>>("RmEpsilon", fst->ArcType(), &args);
}

// #2 signature with default WeightClass argument.
void RmEpsilon(MutableFstClass *fst, bool connect, int64 state_threshold,
               float delta) {
  const WeightClass weight_threshold = WeightClass::Zero(fst->WeightType());
  RmEpsilon(fst, connect, weight_threshold, state_threshold, delta);
}

// 3: Full signature with RmEpsilonOptions and weight vector.
void RmEpsilon(MutableFstClass *fst, std::vector<WeightClass> *distance,
               const RmEpsilonOptions &opts) {
  if (distance) {
    for (auto it = distance->begin(); it != distance->end(); ++it) {
      if (!fst->WeightTypesMatch(*it, "RmEpsilon")) {
        fst->SetProperties(kError, kError);
        return;
      }
    }
  }
  if (!fst->WeightTypesMatch(opts.weight_threshold, "RmEpsilon")) {
    fst->SetProperties(kError, kError);
    return;
  }
  RmEpsilonArgs3 args(fst, distance, opts);
  Apply<Operation<RmEpsilonArgs3>>("RmEpsilon", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(RmEpsilon, StdArc, RmEpsilonArgs1);
REGISTER_FST_OPERATION(RmEpsilon, LogArc, RmEpsilonArgs1);
REGISTER_FST_OPERATION(RmEpsilon, Log64Arc, RmEpsilonArgs1);

REGISTER_FST_OPERATION(RmEpsilon, StdArc, RmEpsilonArgs2);
REGISTER_FST_OPERATION(RmEpsilon, LogArc, RmEpsilonArgs2);
REGISTER_FST_OPERATION(RmEpsilon, Log64Arc, RmEpsilonArgs2);

REGISTER_FST_OPERATION(RmEpsilon, StdArc, RmEpsilonArgs3);
REGISTER_FST_OPERATION(RmEpsilon, LogArc, RmEpsilonArgs3);
REGISTER_FST_OPERATION(RmEpsilon, Log64Arc, RmEpsilonArgs3);

}  // namespace script
}  // namespace fst
