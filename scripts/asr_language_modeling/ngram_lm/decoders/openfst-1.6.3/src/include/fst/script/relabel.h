// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_RELABEL_H_
#define FST_SCRIPT_RELABEL_H_

#include <algorithm>
#include <utility>
#include <vector>

#include <fst/relabel.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

// 1
using RelabelArgs1 =
    args::Package<MutableFstClass *,
                  const SymbolTable *, const SymbolTable *,
                  const string &, bool,
                  const SymbolTable *, const SymbolTable *,
                  const string &, bool>;

template <class Arc>
void Relabel(RelabelArgs1 *args) {
  MutableFst<Arc> *ofst = args->arg1->GetMutableFst<Arc>();
  Relabel(ofst, args->arg2, args->arg3, args->arg4, args->arg5, args->arg6,
          args->arg7, args->arg8, args->arg9);
}

using LabelPair = std::pair<int64, int64>;

// 2
using RelabelArgs2 =
    args::Package<MutableFstClass *, const std::vector<LabelPair> &,
                  const std::vector<LabelPair> &>;

template <class Arc>
void Relabel(RelabelArgs2 *args) {
  MutableFst<Arc> *ofst = args->arg1->GetMutableFst<Arc>();
  using LabelPair = std::pair<typename Arc::Label, typename Arc::Label>;
  // In case the MutableFstClass::Label is not the same as Arc::Label,
  // make a copy.
  std::vector<LabelPair> typed_ipairs(args->arg2.size());
  std::copy(args->arg2.begin(), args->arg2.end(), typed_ipairs.begin());
  std::vector<LabelPair> typed_opairs(args->arg3.size());
  std::copy(args->arg3.begin(), args->arg3.end(), typed_opairs.begin());
  Relabel(ofst, typed_ipairs, typed_opairs);
}

// 3
using RelabelArgs3 =
    args::Package<MutableFstClass *, const SymbolTable *, const SymbolTable *>;
template <class Arc>
void Relabel(args::Package<MutableFstClass *, const SymbolTable *,
                           const SymbolTable *> *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  Relabel(fst, args->arg2, args->arg3);
}

// 1
void Relabel(MutableFstClass *ofst,
             const SymbolTable *old_isymbols, const SymbolTable *new_isymbols,
             const string &unknown_isymbol,  bool attach_new_isymbols,
             const SymbolTable *old_osymbols, const SymbolTable *new_osymbols,
             const string &unknown_osymbol, bool attach_new_osymbols);

// 2
void Relabel(MutableFstClass *ofst, const std::vector<LabelPair> &ipairs,
             const std::vector<LabelPair> &opairs);

// 3
void Relabel(MutableFstClass *fst, const SymbolTable *new_isymbols,
             const SymbolTable *new_osymbols);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_RELABEL_H_
