// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_DECODE_H_
#define FST_SCRIPT_DECODE_H_

#include <memory>
#include <string>

#include <fst/encode.h>
#include <fst/script/arg-packs.h>
#include <fst/script/encodemapper-class.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

// 1: Decode using encoder on disk.
using DecodeArgs1 = args::Package<MutableFstClass *, const string &>;

template <class Arc>
void Decode(DecodeArgs1 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  std::unique_ptr<EncodeMapper<Arc>> decoder(EncodeMapper<Arc>::Read(args->arg2,
                                                                     DECODE));
  if (!decoder) {
    fst->SetProperties(kError, kError);
    return;
  }
  Decode(fst, *decoder);
}

void Decode(MutableFstClass *fst, const string &coder_fname);

// 2: Decode using an EncodeMapperClass.
using DecodeArgs2 = args::Package<MutableFstClass *, const EncodeMapperClass &>;

template <class Arc>
void Decode(DecodeArgs2 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  const EncodeMapper<Arc> &encoder = *(args->arg2.GetEncodeMapper<Arc>());
  Decode(fst, encoder);
}

void Decode(MutableFstClass *fst, const EncodeMapperClass &encoder);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_DECODE_H_
