%module swig_decoders
%{
#include "scorer.h"
#include "ctc_greedy_decoder.h"
#include "ctc_beam_search_decoder.h"
#include "decoder_utils.h"
%}

%include "std_vector.i"
%include "std_pair.i"
%include "std_string.i"
%import "decoder_utils.h"

namespace std {
    %template(DoubleVector) std::vector<double>;
    %template(IntVector) std::vector<int>;
    %template(StringVector) std::vector<std::string>;
    %template(VectorOfStructVector) std::vector<std::vector<double> >;
    %template(FloatVector) std::vector<float>;
    %template(Pair) std::pair<float, std::string>;
    %template(PairFloatStringVector)  std::vector<std::pair<float, std::string> >;
    %template(PairDoubleStringVector) std::vector<std::pair<double, std::string> >;
    %template(PairDoubleStringVector2) std::vector<std::vector<std::pair<double, std::string> > >;
    %template(DoubleVector3) std::vector<std::vector<std::vector<double> > >;
}

%template(IntDoublePairCompSecondRev) pair_comp_second_rev<int, double>;
%template(StringDoublePairCompSecondRev) pair_comp_second_rev<std::string, double>;
%template(DoubleStringPairCompFirstRev) pair_comp_first_rev<double, std::string>;

%include "scorer.h"
%include "ctc_greedy_decoder.h"
%include "ctc_beam_search_decoder.h"
