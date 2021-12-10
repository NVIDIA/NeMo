// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RANDEQUIVALENT_H_
#define FST_SCRIPT_RANDEQUIVALENT_H_

#include <climits>
#include <ctime>

#include <fst/randequivalent.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1
using RandEquivalentInnerArgs1 =
    args::Package<const FstClass &, const FstClass &, int32, float, time_t,
                  const RandGenOptions<RandArcSelection> &, bool *>;

using RandEquivalentArgs1 =
    args::WithReturnValue<bool, RandEquivalentInnerArgs1>;

template <class Arc>
void RandEquivalent(RandEquivalentArgs1 *args) {
  const Fst<Arc> &fst1 = *(args->args.arg1.GetFst<Arc>());
  const Fst<Arc> &fst2 = *(args->args.arg2.GetFst<Arc>());
  const auto seed = args->args.arg5;
  const auto &opts = args->args.arg6;
  if (opts.selector == UNIFORM_ARC_SELECTOR) {
    const UniformArcSelector<Arc> selector(seed);
    const RandGenOptions<UniformArcSelector<Arc>> ropts(selector,
                                                        opts.max_length);
    args->retval = RandEquivalent(fst1, fst2, args->args.arg3, args->args.arg4,
                                  ropts, args->args.arg7);
  } else if (opts.selector == FAST_LOG_PROB_ARC_SELECTOR) {
    const FastLogProbArcSelector<Arc> selector(seed);
    const RandGenOptions<FastLogProbArcSelector<Arc>> ropts(selector,
                                                            opts.max_length);
    args->retval = RandEquivalent(fst1, fst2, args->args.arg3, args->args.arg4,
                                  ropts, args->args.arg7);
  } else {
    const LogProbArcSelector<Arc> selector(seed);
    const RandGenOptions<LogProbArcSelector<Arc>> ropts(selector,
                                                        opts.max_length);
    args->retval = RandEquivalent(fst1, fst2, args->args.arg3, args->args.arg4,
                                  ropts, args->args.arg7);
  }
}

// 2
using RandEquivalentInnerArgs2 =
    args::Package<const FstClass &, const FstClass &, int32, float, time_t,
                  int32, bool *>;

using RandEquivalentArgs2 =
    args::WithReturnValue<bool, RandEquivalentInnerArgs2>;

template <class Arc>
void RandEquivalent(RandEquivalentArgs2 *args) {
  const Fst<Arc> &fst1 = *(args->args.arg1.GetFst<Arc>());
  const Fst<Arc> &fst2 = *(args->args.arg2.GetFst<Arc>());
  args->retval = fst::RandEquivalent(fst1, fst2, args->args.arg3,
                                         args->args.arg4, args->args.arg5,
                                         args->args.arg6, args->args.arg7);
}

// 1
bool RandEquivalent(const FstClass &fst1, const FstClass &fst2, int32 npath = 1,
                    float delta = kDelta, time_t seed = time(nullptr),
                    const RandGenOptions<RandArcSelection> &opts =
                        RandGenOptions<RandArcSelection>(UNIFORM_ARC_SELECTOR),
                    bool *error = nullptr);

// 2
bool RandEquivalent(const FstClass &fst1, const FstClass &fst2, int32 npath = 1,
                    float delta = kDelta, time_t seed = time(nullptr),
                    int32 max_length = INT32_MAX, bool *error = nullptr);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RANDEQUIVALENT_H_
