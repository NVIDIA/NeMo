// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/script/script-impl.h>
#include <fst/script/shortest-path.h>

namespace fst {
namespace script {

// 1
void ShortestPath(const FstClass &ifst, MutableFstClass *ofst,
                  std::vector<WeightClass> *distance,
                  const ShortestPathOptions &opts) {
  if (!ArcTypesMatch(ifst, *ofst, "ShortestPath")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ShortestPathArgs1 args(ifst, ofst, distance, opts);
  Apply<Operation<ShortestPathArgs1>>("ShortestPath", ifst.ArcType(), &args);
}

// 2
void ShortestPath(const FstClass &ifst, MutableFstClass *ofst, int32 n,
                  bool unique, bool first_path,
                  const WeightClass &weight_threshold, int64 state_threshold) {
  if (!ArcTypesMatch(ifst, *ofst, "ShortestPath")) {
    ofst->SetProperties(kError, kError);
    return;
  }
  ShortestPathArgs2 args(ifst, ofst, n, unique, first_path, weight_threshold,
                         state_threshold);
  Apply<Operation<ShortestPathArgs2>>("ShortestPath", ifst.ArcType(), &args);
}

REGISTER_FST_OPERATION(ShortestPath, StdArc, ShortestPathArgs1);
REGISTER_FST_OPERATION(ShortestPath, LogArc, ShortestPathArgs1);
REGISTER_FST_OPERATION(ShortestPath, Log64Arc, ShortestPathArgs1);

REGISTER_FST_OPERATION(ShortestPath, StdArc, ShortestPathArgs2);
REGISTER_FST_OPERATION(ShortestPath, LogArc, ShortestPathArgs2);
REGISTER_FST_OPERATION(ShortestPath, Log64Arc, ShortestPathArgs2);

}  // namespace script
}  // namespace fst
