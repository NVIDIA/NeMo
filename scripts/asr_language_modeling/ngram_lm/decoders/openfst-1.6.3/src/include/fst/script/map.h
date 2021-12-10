// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_MAP_H_
#define FST_SCRIPT_MAP_H_

#include <memory>

#include <fst/arc-map.h>
#include <fst/state-map.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

template <class M>
Fst<typename M::ToArc> *ArcMap(const Fst<typename M::FromArc> &fst,
                               const M &mapper) {
  using ToArc = typename M::ToArc;
  auto *ofst = new VectorFst<ToArc>;
  ArcMap(fst, ofst, mapper);
  return ofst;
}

template <class M>
Fst<typename M::ToArc> *StateMap(const Fst<typename M::FromArc> &fst,
                                 const M &mapper) {
  using ToArc = typename M::ToArc;
  auto *ofst = new VectorFst<ToArc>;
  StateMap(fst, ofst, mapper);
  return ofst;
}

enum MapType {
  ARC_SUM_MAPPER,
  ARC_UNIQUE_MAPPER,
  IDENTITY_MAPPER,
  INPUT_EPSILON_MAPPER,
  INVERT_MAPPER,
  OUTPUT_EPSILON_MAPPER,
  PLUS_MAPPER,
  QUANTIZE_MAPPER,
  RMWEIGHT_MAPPER,
  SUPERFINAL_MAPPER,
  TIMES_MAPPER,
  TO_LOG_MAPPER,
  TO_LOG64_MAPPER,
  TO_STD_MAPPER
};

using MapInnerArgs =
    args::Package<const FstClass &, MapType, float, const WeightClass &>;

using MapArgs = args::WithReturnValue<FstClass *, MapInnerArgs>;

template <class Arc>
void Map(MapArgs *args) {
  const Fst<Arc> &ifst = *(args->args.arg1.GetFst<Arc>());
  MapType map_type = args->args.arg2;
  float delta = args->args.arg3;
  const auto weight = *(args->args.arg4.GetWeight<typename Arc::Weight>());
  std::unique_ptr<Fst<Arc>> fst;
  std::unique_ptr<Fst<LogArc>> lfst;
  std::unique_ptr<Fst<Log64Arc>> l64fst;
  std::unique_ptr<Fst<StdArc>> sfst;
  if (map_type == ARC_SUM_MAPPER) {
    fst.reset(script::StateMap(ifst, ArcSumMapper<Arc>(ifst)));
    args->retval = new FstClass(*fst);
  } else if (map_type == ARC_UNIQUE_MAPPER) {
    fst.reset(script::StateMap(ifst, ArcUniqueMapper<Arc>(ifst)));
    args->retval = new FstClass(*fst);
  } else if (map_type == IDENTITY_MAPPER) {
    fst.reset(script::ArcMap(ifst, IdentityArcMapper<Arc>()));
    args->retval = new FstClass(*fst);
  } else if (map_type == INPUT_EPSILON_MAPPER) {
    fst.reset(script::ArcMap(ifst, InputEpsilonMapper<Arc>()));
    args->retval = new FstClass(*fst);
  } else if (map_type == INVERT_MAPPER) {
    fst.reset(script::ArcMap(ifst, InvertWeightMapper<Arc>()));
    args->retval = new FstClass(*fst);
  } else if (map_type == OUTPUT_EPSILON_MAPPER) {
    fst.reset(script::ArcMap(ifst, OutputEpsilonMapper<Arc>()));
    args->retval = new FstClass(*fst);
  } else if (map_type == PLUS_MAPPER) {
    fst.reset(script::ArcMap(ifst, PlusMapper<Arc>(weight)));
    args->retval = new FstClass(*fst);
  } else if (map_type == QUANTIZE_MAPPER) {
    fst.reset(script::ArcMap(ifst, QuantizeMapper<Arc>(delta)));
    args->retval = new FstClass(*fst);
  } else if (map_type == RMWEIGHT_MAPPER) {
    fst.reset(script::ArcMap(ifst, RmWeightMapper<Arc>()));
    args->retval = new FstClass(*fst);
  } else if (map_type == SUPERFINAL_MAPPER) {
    fst.reset(script::ArcMap(ifst, SuperFinalMapper<Arc>()));
    args->retval = new FstClass(*fst);
  } else if (map_type == TIMES_MAPPER) {
    fst.reset(script::ArcMap(ifst, TimesMapper<Arc>(weight)));
    args->retval = new FstClass(*fst);
  } else if (map_type == TO_LOG_MAPPER) {
    lfst.reset(script::ArcMap(ifst, WeightConvertMapper<Arc, LogArc>()));
    args->retval = new FstClass(*lfst);
  } else if (map_type == TO_LOG64_MAPPER) {
    l64fst.reset(script::ArcMap(ifst, WeightConvertMapper<Arc, Log64Arc>()));
    args->retval = new FstClass(*l64fst);
  } else if (map_type == TO_STD_MAPPER) {
    sfst.reset(script::ArcMap(ifst, WeightConvertMapper<Arc, StdArc>()));
    args->retval = new FstClass(*sfst);
  } else {
    FSTERROR() << "Unknown mapper type: " << map_type;
    VectorFst<Arc> *ofst = new VectorFst<Arc>;
    ofst->SetProperties(kError, kError);
    args->retval = new FstClass(*ofst);
  }
}

FstClass *Map(const FstClass &fst, MapType map_type, float delta,
              const WeightClass &weight);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_MAP_H_
