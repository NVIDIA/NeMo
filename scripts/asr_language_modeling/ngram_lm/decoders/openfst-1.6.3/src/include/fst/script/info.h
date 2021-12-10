// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_INFO_H_
#define FST_SCRIPT_INFO_H_

#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>
#include <fst/script/info-impl.h>

namespace fst {
namespace script {

using InfoArgs = args::Package<const FstClass &, bool, const string &,
                               const string &, bool, bool>;

template <class Arc>
void PrintFstInfo(InfoArgs *args) {
  const Fst<Arc> &fst = *(args->arg1.GetFst<Arc>());
  FstInfo fstinfo(fst, args->arg2, args->arg3, args->arg4, args->arg5);
  PrintFstInfoImpl(fstinfo, args->arg6);
  if (args->arg6) fst.Write("");
}

void PrintFstInfo(const FstClass &f, bool test_properties,
                  const string &arc_filter, const string &info_type, bool pipe,
                  bool verify);

using GetInfoArgs = args::Package<const FstClass &, bool, const string &,
                                  const string &, bool, FstInfo *>;

template <class Arc>
void GetFstInfo(GetInfoArgs *args) {
  const Fst<Arc> &fst = *(args->arg1.GetFst<Arc>());
  *(args->arg6) = FstInfo(fst, args->arg2, args->arg3, args->arg4, args->arg5);
}

void GetFstInfo(const FstClass &f, bool test_properties,
                const string &arc_filter, const string &info_type, bool verify,
                FstInfo *info);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_INFO_H_
