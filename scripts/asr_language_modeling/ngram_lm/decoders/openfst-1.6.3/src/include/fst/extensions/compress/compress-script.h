// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Declarations of 'scriptable' versions of compression operations, that is,
// those that can be called with FstClass-type arguments.

#ifndef FST_EXTENSIONS_COMPRESS_COMPRESS_SCRIPT_H_
#define FST_EXTENSIONS_COMPRESS_COMPRESS_SCRIPT_H_

#include <utility>
#include <vector>

#include <fst/extensions/compress/compress.h>

#include <fst/log.h>
#include <fst/util.h>
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

typedef args::Package<const FstClass &, const string &, const bool>
    CompressArgs;

template <class Arc>
void Compress(CompressArgs *args) {
  const Fst<Arc> &fst = *(args->arg1.GetFst<Arc>());
  const string &filename = args->arg2;
  const bool gzip = args->arg3;

  if (!fst::Compress(fst, filename, gzip)) FSTERROR() << "Compress: failed";
}

void Compress(const FstClass &fst, const string &filename, const bool gzip);

typedef args::Package<const string &, MutableFstClass *, const bool>
    DecompressArgs;

template <class Arc>
void Decompress(DecompressArgs *args) {
  const string &filename = args->arg1;
  MutableFst<Arc> *fst = args->arg2->GetMutableFst<Arc>();
  const bool gzip = args->arg3;

  if (!fst::Decompress(filename, fst, gzip))
    FSTERROR() << "Decompress: failed";
}

void Decompress(const string &filename, MutableFstClass *fst, const bool gzip);

}  // namespace script
}  // namespace fst

#endif  // FST_EXTENSIONS_COMPRESS_COMPRESS_SCRIPT_H_
