// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_REPLACE_H_
#define FST_SCRIPT_REPLACE_H_

#include <utility>
#include <vector>

#include <fst/replace.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

struct ReplaceOptions {
  int64 root;                                   // Root rule for expansion.
  fst::ReplaceLabelType call_label_type;    // How to label call arc.
  fst::ReplaceLabelType return_label_type;  // How to label return arc.
  int64 return_label;                           // Specifies return arc label.

  explicit ReplaceOptions(int64 r,
      fst::ReplaceLabelType c = fst::REPLACE_LABEL_INPUT,
      fst::ReplaceLabelType t = fst::REPLACE_LABEL_NEITHER,
      int64 l = 0)
      : root(r), call_label_type(c), return_label_type(t), return_label(l) {}
};

using LabelFstClassPair = std::pair<int64, const FstClass *>;

using ReplaceArgs = args::Package<const std::vector<LabelFstClassPair> &,
                                  MutableFstClass *, const ReplaceOptions &>;

template <class Arc>
void Replace(ReplaceArgs *args) {
  using LabelFstPair = std::pair<typename Arc::Label, const Fst<Arc> *>;
  // Now that we know the arc type, we construct a vector of
  // std::pair<real label, real fst> that the real Replace will use.
  const std::vector<LabelFstClassPair> &untyped_pairs = args->arg1;
  auto size = untyped_pairs.size();
  std::vector<LabelFstPair> typed_pairs(size);
  for (auto i = 0; i < size; ++i) {
    typed_pairs[i].first = untyped_pairs[i].first;  // Converts label.
    typed_pairs[i].second = untyped_pairs[i].second->GetFst<Arc>();
  }
  MutableFst<Arc> *ofst = args->arg2->GetMutableFst<Arc>();
  const ReplaceOptions &opts = args->arg3;
  ReplaceFstOptions<Arc> typed_opts(opts.root, opts.call_label_type,
                                    opts.return_label_type, opts.return_label);
  ReplaceFst<Arc> rfst(typed_pairs, typed_opts);
  // Checks for cyclic dependencies before attempting expansino.
  if (rfst.CyclicDependencies()) {
    FSTERROR() << "Replace: Cyclic dependencies detected; cannot expand";
    ofst->SetProperties(kError, kError);
    return;
  }
  typed_opts.gc = true;     // Caching options to speed up batch copy.
  typed_opts.gc_limit = 0;
  *ofst = rfst;
}

void Replace(const std::vector<LabelFstClassPair> &pairs,
             MutableFstClass *ofst, const ReplaceOptions &opts);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_REPLACE_H_
