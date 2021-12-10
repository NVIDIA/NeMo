// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_SHORTEST_DISTANCE_H_
#define FST_SCRIPT_SHORTEST_DISTANCE_H_

#include <vector>

#include <fst/queue.h>  // for QueueType
#include <fst/shortest-distance.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/prune.h>  // for ArcFilterType
#include <fst/script/weight-class.h>

namespace fst {
namespace script {

enum ArcFilterType {
  ANY_ARC_FILTER,
  EPSILON_ARC_FILTER,
  INPUT_EPSILON_ARC_FILTER,
  OUTPUT_EPSILON_ARC_FILTER
};

// See nlp/fst/lib/shortest-distance.h for the template options class
// that this one shadows
struct ShortestDistanceOptions {
  const QueueType queue_type;
  const ArcFilterType arc_filter_type;
  const int64 source;
  const float delta;
  const bool first_path;

  ShortestDistanceOptions(QueueType qt, ArcFilterType aft, int64 s, float d)
      : queue_type(qt), arc_filter_type(aft), source(s), delta(d),
        first_path(false) {}
};

// 1
using ShortestDistanceArgs1 =
    args::Package<const FstClass &, std::vector<WeightClass> *,
                  const ShortestDistanceOptions &>;

template <class Queue, class Arc, class ArcFilter>
struct QueueConstructor {
  static Queue *Construct(const Fst<Arc> &,
                          const std::vector<typename Arc::Weight> *) {
    return new Queue();
  }
};

// Specializations to deal with AutoQueue, NaturalShortestFirstQueue,
// and TopOrderQueue's different constructors
template <class Arc, class ArcFilter>
struct QueueConstructor<AutoQueue<typename Arc::StateId>, Arc, ArcFilter> {
  //  template<class Arc, class ArcFilter>
  static AutoQueue<typename Arc::StateId> *Construct(
      const Fst<Arc> &fst, const std::vector<typename Arc::Weight> *distance) {
    return new AutoQueue<typename Arc::StateId>(fst, distance, ArcFilter());
  }
};

template <class Arc, class ArcFilter>
struct QueueConstructor<
    NaturalShortestFirstQueue<typename Arc::StateId, typename Arc::Weight>, Arc,
    ArcFilter> {
  //  template<class Arc, class ArcFilter>
  static NaturalShortestFirstQueue<typename Arc::StateId, typename Arc::Weight>
      *Construct(const Fst<Arc> &fst,
                 const std::vector<typename Arc::Weight> *distance) {
    return new NaturalShortestFirstQueue<typename Arc::StateId,
                                         typename Arc::Weight>(*distance);
  }
};

template <class Arc, class ArcFilter>
struct QueueConstructor<TopOrderQueue<typename Arc::StateId>, Arc, ArcFilter> {
  //  template<class Arc, class ArcFilter>
  static TopOrderQueue<typename Arc::StateId> *Construct(
      const Fst<Arc> &fst, const std::vector<typename Arc::Weight> *weights) {
    return new TopOrderQueue<typename Arc::StateId>(fst, ArcFilter());
  }
};

template <class Arc, class Queue>
void ShortestDistanceHelper(ShortestDistanceArgs1 *args) {
  const Fst<Arc> &fst = *(args->arg1.GetFst<Arc>());
  const ShortestDistanceOptions &opts = args->arg3;
  std::vector<typename Arc::Weight> weights;
  switch (opts.arc_filter_type) {
    case ANY_ARC_FILTER: {
      std::unique_ptr<Queue> queue(QueueConstructor<Queue, Arc,
          AnyArcFilter<Arc>>::Construct(fst, &weights));
      fst::ShortestDistanceOptions<Arc, Queue, AnyArcFilter<Arc>> sdopts(
          queue.get(), AnyArcFilter<Arc>(), opts.source, opts.delta);
      ShortestDistance(fst, &weights, sdopts);
      break;
    }
    case EPSILON_ARC_FILTER: {
      std::unique_ptr<Queue> queue(QueueConstructor<Queue, Arc,
          AnyArcFilter<Arc>>::Construct(fst, &weights));
      fst::ShortestDistanceOptions<Arc, Queue, EpsilonArcFilter<Arc>>
          sdopts(queue.get(), EpsilonArcFilter<Arc>(), opts.source, opts.delta);
      ShortestDistance(fst, &weights, sdopts);
      break;
    }
    case INPUT_EPSILON_ARC_FILTER: {
      std::unique_ptr<Queue> queue(QueueConstructor<Queue, Arc,
          InputEpsilonArcFilter<Arc>>::Construct(fst, &weights));
      fst::ShortestDistanceOptions<Arc, Queue, InputEpsilonArcFilter<Arc>>
          sdopts(queue.get(), InputEpsilonArcFilter<Arc>(), opts.source,
                 opts.delta);
      ShortestDistance(fst, &weights, sdopts);
      break;
    }
    case OUTPUT_EPSILON_ARC_FILTER: {
      std::unique_ptr<Queue> queue(QueueConstructor<Queue, Arc,
          OutputEpsilonArcFilter<Arc>>::Construct(fst, &weights));
      fst::ShortestDistanceOptions<Arc, Queue, OutputEpsilonArcFilter<Arc>>
          sdopts(queue.get(), OutputEpsilonArcFilter<Arc>(), opts.source,
                 opts.delta);
      ShortestDistance(fst, &weights, sdopts);
      break;
    }
  }
  // Copies the weights back.
  args->arg2->resize(weights.size());
  for (auto i = 0; i < weights.size(); ++i) {
    (*args->arg2)[i] = WeightClass(weights[i]);
  }
}

template <class Arc>
void ShortestDistance(ShortestDistanceArgs1 *args) {
  const ShortestDistanceOptions &opts = args->arg3;
  using StateId = typename Arc::StateId;
  using Weight = typename Arc::Weight;
  // Must consider (opts.queue_type x opts.filter_type) options.
  switch (opts.queue_type) {
    default:
      FSTERROR() << "Unknown queue type: " << opts.queue_type;
    case AUTO_QUEUE:
      ShortestDistanceHelper<Arc, AutoQueue<StateId>>(args);
      return;
    case FIFO_QUEUE:
      ShortestDistanceHelper<Arc, FifoQueue<StateId>>(args);
      return;
    case LIFO_QUEUE:
      ShortestDistanceHelper<Arc, LifoQueue<StateId>>(args);
      return;
    case SHORTEST_FIRST_QUEUE:
      ShortestDistanceHelper<Arc, NaturalShortestFirstQueue<StateId, Weight>>(
          args);
      return;
    case STATE_ORDER_QUEUE:
      ShortestDistanceHelper<Arc, StateOrderQueue<StateId>>(args);
      return;
    case TOP_ORDER_QUEUE:
      ShortestDistanceHelper<Arc, TopOrderQueue<StateId>>(args);
      return;
  }
}

// 2
using ShortestDistanceArgs2 =
    args::Package<const FstClass &, std::vector<WeightClass> *, bool, double>;

template <class Arc>
void ShortestDistance(ShortestDistanceArgs2 *args) {
  const Fst<Arc> &fst = *(args->arg1.GetFst<Arc>());
  std::vector<typename Arc::Weight> distance;
  ShortestDistance(fst, &distance, args->arg3, args->arg4);
  // Converts the typed weights back to WeightClass instances.
  std::vector<WeightClass> *retval = args->arg2;
  retval->resize(distance.size());
  for (size_t i = 0; i < distance.size(); ++i)
    (*retval)[i] = WeightClass(distance[i]);
}

// 1
void ShortestDistance(const FstClass &fst, std::vector<WeightClass> *distance,
                      const ShortestDistanceOptions &opts);

// 2
void ShortestDistance(const FstClass &ifst, std::vector<WeightClass> *distance,
                      bool reverse = false, double delta = fst::kDelta);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_SHORTEST_DISTANCE_H_
