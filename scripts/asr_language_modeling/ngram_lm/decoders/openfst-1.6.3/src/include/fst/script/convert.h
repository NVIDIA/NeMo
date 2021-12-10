// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_CONVERT_H_
#define FST_SCRIPT_CONVERT_H_

#include <memory>
#include <string>

#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ConvertInnerArgs = args::Package<const FstClass &, const string &>;

using ConvertArgs = args::WithReturnValue<FstClass *, ConvertInnerArgs>;

template <class Arc>
void Convert(ConvertArgs *args) {
  const Fst<Arc> &fst = *(args->args.arg1.GetFst<Arc>());
  const string &new_type = args->args.arg2;
  std::unique_ptr<Fst<Arc>> result(Convert(fst, new_type));
  args->retval = result ? new FstClass(*result) : nullptr;
}

FstClass *Convert(const FstClass &f, const string &new_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_CONVERT_H_
