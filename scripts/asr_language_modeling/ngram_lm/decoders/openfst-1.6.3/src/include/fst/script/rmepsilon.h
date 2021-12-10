// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RMEPSILON_H_
#define FST_SCRIPT_RMEPSILON_H_

#include <vector>

#include <fst/queue.h>
#include <fst/rmepsilon.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/shortest-distance.h>  // for ShortestDistanceOptions
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

struct RmEpsilonOptions : public fst::script::ShortestDistanceOptions {
  bool connect;
  const WeightClass &weight_threshold;
  int64 state_threshold;

  RmEpsilonOptions(QueueType qt, float d, bool c, const WeightClass &w,
                   int64 n = kNoStateId)
      : ShortestDistanceOptions(qt, EPSILON_ARC_FILTER, kNoStateId, d),
        connect(c), weight_threshold(w), state_threshold(n) {}
};

// This function transforms a script-land RmEpsilonOptions into a lib-land
// RmEpsilonOptions, and then calls the operation.
template <class Arc>
void RmEpsilonHelper(MutableFst<Arc> *fst,
                     std::vector<typename Arc::Weight> *distance,
                     const RmEpsilonOptions &opts) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  const Weight weight_threshold = *(opts.weight_threshold.GetWeight<Weight>());
  switch (opts.queue_type) {
    case AUTO_QUEUE: {
      AutoQueue<StateId> queue(*fst, distance, EpsilonArcFilter<Arc>());
      fst::RmEpsilonOptions<Arc, AutoQueue<StateId>> ropts(
          &queue, opts.delta, opts.connect, weight_threshold,
          opts.state_threshold);
      RmEpsilon(fst, distance, ropts);
      break;
    }
    case FIFO_QUEUE: {
      FifoQueue<StateId> queue;
      fst::RmEpsilonOptions<Arc, FifoQueue<StateId>> ropts(
          &queue, opts.delta, opts.connect, weight_threshold,
          opts.state_threshold);
      RmEpsilon(fst, distance, ropts);
      break;
    }
    case LIFO_QUEUE: {
      LifoQueue<StateId> queue;
      fst::RmEpsilonOptions<Arc, LifoQueue<StateId>> ropts(
          &queue, opts.delta, opts.connect, weight_threshold,
          opts.state_threshold);
      RmEpsilon(fst, distance, ropts);
      break;
    }
    case SHORTEST_FIRST_QUEUE: {
      NaturalShortestFirstQueue<StateId, Weight> queue(*distance);
      fst::RmEpsilonOptions<Arc, NaturalShortestFirstQueue<StateId, Weight>>
          ropts(&queue, opts.delta, opts.connect, weight_threshold,
                opts.state_threshold);
      RmEpsilon(fst, distance, ropts);
      break;
    }
    case STATE_ORDER_QUEUE: {
      StateOrderQueue<StateId> queue;
      fst::RmEpsilonOptions<Arc, StateOrderQueue<StateId>> ropts(
          &queue, opts.delta, opts.connect, weight_threshold,
          opts.state_threshold);
      RmEpsilon(fst, distance, ropts);
      break;
    }
    case TOP_ORDER_QUEUE: {
      TopOrderQueue<StateId> queue(*fst, EpsilonArcFilter<Arc>());
      fst::RmEpsilonOptions<Arc, TopOrderQueue<StateId>> ropts(
          &queue, opts.delta, opts.connect, weight_threshold,
          opts.state_threshold);
      RmEpsilon(fst, distance, ropts);
      break;
    }
    default:
      FSTERROR() << "Unknown queue type: " << opts.queue_type;
      fst->SetProperties(kError, kError);
  }
}

// 1: Full signature with RmEpsilonOptions.
using RmEpsilonArgs1 = args::Package<const FstClass &, MutableFstClass *, bool,
                                     const RmEpsilonOptions &>;

template <class Arc>
void RmEpsilon(RmEpsilonArgs1 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  std::vector<typename Arc::Weight> distance;
  if (args->arg3) {
    VectorFst<Arc> rfst;
    Reverse(ifst, &rfst, false);
    RmEpsilonHelper(&rfst, &distance, args->arg4);
    Reverse(rfst, ofst, false);
    if (rfst.NumStates() != ofst->NumStates())
      RmEpsilonHelper(ofst, &distance, args->arg4);
  } else {
    *ofst = ifst;
    RmEpsilonHelper(ofst, &distance, args->arg4);
  }
}

// 2: Full signature with flat arguments.
using RmEpsilonArgs2 =
    args::Package<MutableFstClass *, bool, const WeightClass, int64, float>;

template <class Arc>
void RmEpsilon(RmEpsilonArgs2 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  typename Arc::Weight weight = *(args->arg3.GetWeight<typename Arc::Weight>());
  RmEpsilon(fst, args->arg2, weight, args->arg4, args->arg5);
}

// 3: Full signature with RmEpsilonOptions and weight vector.
using RmEpsilonArgs3 =
    args::Package<MutableFstClass *, std::vector<WeightClass> *,
                  const RmEpsilonOptions &>;

template <class Arc>
void RmEpsilon(RmEpsilonArgs3 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  const RmEpsilonOptions &opts = args->arg3;
  std::vector<typename Arc::Weight> weights;
  RmEpsilonHelper(fst, &weights, opts);
  // Copy the weights back....
  args->arg2->resize(weights.size());
  for (auto i = 0; i < weights.size(); ++i) {
    (*args->arg2)[i] = WeightClass(weights[i]);
  }
}

// 1
void RmEpsilon(const FstClass &ifst, MutableFstClass *ofst, bool reverse,
               const RmEpsilonOptions &opts);

// 2
void RmEpsilon(MutableFstClass *fst, bool connect,
               const WeightClass &weight_threshold,
               int64 state_threshold = fst::kNoStateId,
               float delta = fst::kDelta);

// #2 signature with default WeightClass argument.
void RmEpsilon(MutableFstClass *fst, bool connect,
               int64 state_threshold = fst::kNoStateId,
               float delta = fst::kDelta);

// 3
void RmEpsilon(MutableFstClass *fst, std::vector<WeightClass> *distance,
               const RmEpsilonOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RMEPSILON_H_
