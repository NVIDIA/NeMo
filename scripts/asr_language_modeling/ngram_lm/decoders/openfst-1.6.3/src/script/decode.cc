// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/script/fst-class.h>
#include <fst/encode.h>
#include <fst/script/decode.h>
#include <fst/script/script-impl.h>

namespace fst {
namespace script {

// 1: Decode using encoder on disk.
void Decode(MutableFstClass *fst, const string &coder_fname) {
  DecodeArgs1 args(fst, coder_fname);
  Apply<Operation<DecodeArgs1>>("Decode", fst->ArcType(), &args);
}

// 2: Decode using an EncodeMapperClass.
void Decode(MutableFstClass *fst, const EncodeMapperClass &encoder) {
  if (fst->ArcType() != encoder.ArcType()) {
    FSTERROR() << "FST and encoder with non-matching arc types passed to "
               << "Decode:\n\t" << fst->ArcType() << " and "
               << encoder.ArcType();
    fst->SetProperties(kError, kError);
    return;
  }
  DecodeArgs2 args(fst, encoder);
  Apply<Operation<DecodeArgs2>>("Decode", fst->ArcType(), &args);
}

REGISTER_FST_OPERATION(Decode, StdArc, DecodeArgs1);
REGISTER_FST_OPERATION(Decode, LogArc, DecodeArgs1);
REGISTER_FST_OPERATION(Decode, Log64Arc, DecodeArgs1);

REGISTER_FST_OPERATION(Decode, StdArc, DecodeArgs2);
REGISTER_FST_OPERATION(Decode, LogArc, DecodeArgs2);
REGISTER_FST_OPERATION(Decode, Log64Arc, DecodeArgs2);

}  // namespace script
}  // namespace fst
