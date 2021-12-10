// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_ENCODE_H_
#define FST_SCRIPT_ENCODE_H_

#include <memory>
#include <string>

#include <fst/encode.h>
#include <fst/script/arg-packs.h>
#include <fst/script/encodemapper-class.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

// 1: Encode using encoder on disk.
using EncodeArgs1 =
    args::Package<MutableFstClass *, uint32, bool, const string &>;

template <class Arc>
void Encode(EncodeArgs1 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  const string &coder_fname = args->arg4;
  // If true, reuse encode from disk. If false, make a new encoder and just use
  // the filename argument as the destination state.
  std::unique_ptr<EncodeMapper<Arc>> encoder(args->arg3 ?
      EncodeMapper<Arc>::Read(coder_fname, ENCODE) :
      new EncodeMapper<Arc>(args->arg2, ENCODE));
  Encode(fst, encoder.get());
  if (!args->arg3) encoder->Write(coder_fname);
}

void Encode(MutableFstClass *fst, uint32 flags, bool reuse_encoder,
            const string &coder_fname);

// 2: Encode using an EncodeMapperClass object.
using EncodeArgs2 = args::Package<MutableFstClass *, EncodeMapperClass *>;

template <class Arc>
void Encode(EncodeArgs2 *args) {
  MutableFst<Arc> *fst = args->arg1->GetMutableFst<Arc>();
  EncodeMapper<Arc> *encoder = args->arg2->GetEncodeMapper<Arc>();
  Encode(fst, encoder);
}

void Encode(MutableFstClass *fst, EncodeMapperClass *encoder);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ENCODE_H_
