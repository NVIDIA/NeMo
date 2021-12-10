// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#include <fst/extensions/ngram/ngram-fst.h>

#include <sys/types.h>

#include <fst/fstlib.h>

using fst::NGramFst;
using fst::StdArc;
using fst::LogArc;

REGISTER_FST(NGramFst, StdArc);
REGISTER_FST(NGramFst, LogArc);
