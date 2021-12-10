// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/randequivalent.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
bool RandEquivalent(const FstClass &fst1, const FstClass &fst2, int32 npath,
                    float delta, time_t seed,
                    const RandGenOptions<RandArcSelection> &opts, bool *error) {
  if (!ArcTypesMatch(fst1, fst2, "RandEquivalent")) return false;
  RandEquivalentInnerArgs1 iargs(fst1, fst2, npath, delta, seed, opts, error);
  RandEquivalentArgs1 args(iargs);
  Apply<Operation<RandEquivalentArgs1>>("RandEquivalent", fst1.ArcType(),
                                        &args);
  return args.retval;
}

// 2
bool RandEquivalent(const FstClass &fst1, const FstClass &fst2, int32 npath,
                    float delta, time_t seed, int32 max_length, bool *error) {
  if (!ArcTypesMatch(fst1, fst2, "RandEquivalent")) return false;
  RandEquivalentInnerArgs2 iargs(fst1, fst2, npath, delta, seed, max_length,
                                 error);
  RandEquivalentArgs2 args(iargs);
  Apply<Operation<RandEquivalentArgs2>>("RandEquivalent", fst1.ArcType(),
                                        &args);
  return args.retval;
}

REGISTER_FST_OPERATION(RandEquivalent, StdArc, RandEquivalentArgs1);
REGISTER_FST_OPERATION(RandEquivalent, LogArc, RandEquivalentArgs1);
REGISTER_FST_OPERATION(RandEquivalent, Log64Arc, RandEquivalentArgs1);

REGISTER_FST_OPERATION(RandEquivalent, StdArc, RandEquivalentArgs2);
REGISTER_FST_OPERATION(RandEquivalent, LogArc, RandEquivalentArgs2);
REGISTER_FST_OPERATION(RandEquivalent, Log64Arc, RandEquivalentArgs2);

}  // namespace script
}  // namespace fst
