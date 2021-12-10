// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_SHORTEST_PATH_H_
#define FST_SCRIPT_SHORTEST_PATH_H_

#include <memory>
#include <vector>

#include <fst/shortest-path.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/shortest-distance.h>  // for ShortestDistanceOptions
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

struct ShortestPathOptions : public fst::script::ShortestDistanceOptions {
  const int32 nshortest;
  const bool unique;
  const bool has_distance;
  const bool first_path;
  const WeightClass &weight_threshold;
  const int64 state_threshold;

  ShortestPathOptions(QueueType qt, int32 n, bool u, bool hasdist, float d,
                      bool fp, const WeightClass &w,
                      int64 s = fst::kNoStateId)
      : ShortestDistanceOptions(qt, ANY_ARC_FILTER, kNoStateId, d),
        nshortest(n), unique(u), has_distance(hasdist), first_path(fp),
        weight_threshold(w), state_threshold(s) {}
};

// 1
using ShortestPathArgs1 =
    args::Package<const FstClass &, MutableFstClass *,
                  std::vector<WeightClass> *, const ShortestPathOptions &>;

template <class Arc>
void ShortestPath(ShortestPathArgs1 *args) {
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  const ShortestPathOptions &opts = args->arg4;
  using ArcFilter = AnyArcFilter<Arc>;
  std::vector<Weight> weights;
  const Weight &weight_threshold = *opts.weight_threshold.GetWeight<Weight>();
  switch (opts.queue_type) {
    case AUTO_QUEUE: {
      using Queue = AutoQueue<StateId>;
      std::unique_ptr<Queue> queue(
          QueueConstructor<Queue, Arc, ArcFilter>::Construct(ifst, &weights));
      fst::ShortestPathOptions<Arc, Queue, ArcFilter> spopts(
          queue.get(), ArcFilter(), opts.nshortest, opts.unique,
          opts.has_distance, opts.delta, opts.first_path, weight_threshold,
          opts.state_threshold);
      ShortestPath(ifst, ofst, &weights, spopts);
      return;
    }
    case FIFO_QUEUE: {
      using Queue = FifoQueue<StateId>;
      std::unique_ptr<Queue> queue(
          QueueConstructor<Queue, Arc, ArcFilter>::Construct(ifst, &weights));
      fst::ShortestPathOptions<Arc, Queue, ArcFilter> spopts(
          queue.get(), ArcFilter(), opts.nshortest, opts.unique,
          opts.has_distance, opts.delta, opts.first_path, weight_threshold,
          opts.state_threshold);
      ShortestPath(ifst, ofst, &weights, spopts);
      return;
    }
    case LIFO_QUEUE: {
      using Queue = LifoQueue<StateId>;
      std::unique_ptr<Queue> queue(
          QueueConstructor<Queue, Arc, ArcFilter>::Construct(ifst, &weights));
      fst::ShortestPathOptions<Arc, Queue, ArcFilter> spopts(
          queue.get(), ArcFilter(), opts.nshortest, opts.unique,
          opts.has_distance, opts.delta, opts.first_path, weight_threshold,
          opts.state_threshold);
      ShortestPath(ifst, ofst, &weights, spopts);
      return;
    }
    case SHORTEST_FIRST_QUEUE: {
      using Queue = NaturalShortestFirstQueue<StateId, Weight>;
      std::unique_ptr<Queue> queue(
          QueueConstructor<Queue, Arc, ArcFilter>::Construct(ifst, &weights));
      fst::ShortestPathOptions<Arc, Queue, ArcFilter> spopts(
          queue.get(), ArcFilter(), opts.nshortest, opts.unique,
          opts.has_distance, opts.delta, opts.first_path, weight_threshold,
          opts.state_threshold);
      ShortestPath(ifst, ofst, &weights, spopts);
      return;
    }
    case STATE_ORDER_QUEUE: {
      using Queue = StateOrderQueue<StateId>;
      std::unique_ptr<Queue> queue(
          QueueConstructor<Queue, Arc, ArcFilter>::Construct(ifst, &weights));
      fst::ShortestPathOptions<Arc, Queue, ArcFilter> spopts(
          queue.get(), ArcFilter(), opts.nshortest, opts.unique,
          opts.has_distance, opts.delta, opts.first_path, weight_threshold,
          opts.state_threshold);
      ShortestPath(ifst, ofst, &weights, spopts);
      return;
    }
    case TOP_ORDER_QUEUE: {
      using Queue = TopOrderQueue<StateId>;
      std::unique_ptr<Queue> queue(
          QueueConstructor<Queue, Arc, ArcFilter>::Construct(ifst, &weights));
      fst::ShortestPathOptions<Arc, Queue, ArcFilter> spopts(
          queue.get(), ArcFilter(), opts.nshortest, opts.unique,
          opts.has_distance, opts.delta, opts.first_path, weight_threshold,
          opts.state_threshold);
      ShortestPath(ifst, ofst, &weights, spopts);
      return;
    }
    default:
      FSTERROR() << "Unknown queue type: " << opts.queue_type;
      ofst->SetProperties(kError, kError);
  }
  // Copies the weights back.
  args->arg3->resize(weights.size());
  for (auto i = 0; i < weights.size(); ++i) {
    (*args->arg3)[i] = WeightClass(weights[i]);
  }
}

// 2
using ShortestPathArgs2 =
    args::Package<const FstClass &, MutableFstClass *, int32, bool, bool,
                  const WeightClass &, int64>;

template <class Arc>
void ShortestPath(ShortestPathArgs2 *args) {
  const Fst<Arc> &ifst = *(args->arg1.GetFst<Arc>());
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  const typename Arc::Weight &weight_threshold =
      *args->arg6.GetWeight<typename Arc::Weight>();
  ShortestPath(ifst, ofst, args->arg3, args->arg4, args->arg5,
               weight_threshold, args->arg7);
}

// 1
void ShortestPath(const FstClass &ifst, MutableFstClass *ofst,
                  std::vector<WeightClass> *distance,
                  const ShortestPathOptions &opts);

// 2
void ShortestPath(const FstClass &ifst, MutableFstClass *ofst, int32 n,
    bool unique, bool first_path, const WeightClass &weight_threshold,
    int64 state_threshold = fst::kNoStateId);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_SHORTEST_PATH_H_
