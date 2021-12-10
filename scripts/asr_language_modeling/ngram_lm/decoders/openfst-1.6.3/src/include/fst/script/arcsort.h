// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ARCSORT_H_
#define FST_SCRIPT_ARCSORT_H_

#include <fst/arcsort.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

enum ArcSortType { ILABEL_SORT, OLABEL_SORT };

using ArcSortArgs = args::Package<MutableFstClass *, ArcSortType>;

template <class Arc>
void ArcSort(ArcSortArgs *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  if (args->arg2 == ILABEL_SORT) {
    ILabelCompare<Arc> icomp;
    ArcSort(fst, icomp);
  } else {  // args->arg2 == OLABEL_SORT.
    OLabelCompare<Arc> ocomp;
    ArcSort(fst, ocomp);
  }
}

void ArcSort(MutableFstClass *ofst, ArcSortType);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ARCSORT_H_
