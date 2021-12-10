// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Draws a binary FSTs in the Graphviz dot text format.

#include <cstring>

#include <fstream>
#include <memory>
#include <ostream>
#include <string>

#include <fst/log.h>
#include <fst/script/draw.h>

DEFINE_bool(acceptor, false, "Input in acceptor format");
DEFINE_string(isymbols, "", "Input label symbol table");
DEFINE_string(osymbols, "", "Output label symbol table");
DEFINE_string(ssymbols, "", "State label symbol table");
DEFINE_bool(numeric, false, "Print numeric labels");
DEFINE_int32(precision, 5, "Set precision (number of char/float)");
DEFINE_string(float_format, "g",
              "Floating-point format, one of: \"e\", \"f\", or \"g\"");
DEFINE_bool(show_weight_one, false,
            "Print/draw arc weights and final weights equal to Weight::One()");
DEFINE_string(title, "", "Set figure title");
DEFINE_bool(portrait, false, "Portrait mode (def: landscape)");
DEFINE_bool(vertical, false, "Draw bottom-to-top instead of left-to-right");
DEFINE_int32(fontsize, 14, "Set fontsize");
DEFINE_double(height, 11, "Set height");
DEFINE_double(width, 8.5, "Set width");
DEFINE_double(nodesep, 0.25,
              "Set minimum separation between nodes (see dot documentation)");
DEFINE_double(ranksep, 0.40,
              "Set minimum separation between ranks (see dot documentation)");
DEFINE_bool(allow_negative_labels, false,
            "Allow negative labels (not recommended; may cause conflicts)");

int main(int argc, char **argv) {
  namespace s = fst::script;
  using fst::script::FstClass;
  using fst::SymbolTable;
  using fst::SymbolTableTextOptions;

  string usage = "Prints out binary FSTs in dot text format.\n\n  Usage: ";
  usage += argv[0];
  usage += " [binary.fst [text.dot]]\n";

  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);
  if (argc > 3) {
    ShowUsage();
    return 1;
  }

  const string in_name = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : "";

  std::unique_ptr<FstClass> fst(FstClass::Read(in_name));
  if (!fst) return 1;

  string dest = "stdout";
  std::ofstream fstrm;
  if (argc == 3) {
    fstrm.open(argv[2]);
    if (!fstrm) {
      LOG(ERROR) << argv[0] << ": Open failed, file = " << argv[2];
      return 1;
    }
    dest = argv[2];
  }
  std::ostream &ostrm = fstrm.is_open() ? fstrm : std::cout;

  const SymbolTableTextOptions opts(FLAGS_allow_negative_labels);

  std::unique_ptr<const SymbolTable> isyms;
  if (!FLAGS_isymbols.empty() && !FLAGS_numeric) {
    isyms.reset(SymbolTable::ReadText(FLAGS_isymbols, opts));
    if (!isyms) return 1;
  }

  std::unique_ptr<const SymbolTable> osyms;
  if (!FLAGS_osymbols.empty() && !FLAGS_numeric) {
    osyms.reset(SymbolTable::ReadText(FLAGS_osymbols, opts));
    if (!osyms) return 1;
  }

  std::unique_ptr<const SymbolTable> ssyms;
  if (!FLAGS_ssymbols.empty() && !FLAGS_numeric) {
    ssyms.reset(SymbolTable::ReadText(FLAGS_ssymbols));
    if (!ssyms) return 1;
  }

  if (!isyms && !FLAGS_numeric && fst->InputSymbols()) {
    isyms.reset(fst->InputSymbols()->Copy());
  }

  if (!osyms && !FLAGS_numeric && fst->OutputSymbols()) {
    osyms.reset(fst->OutputSymbols()->Copy());
  }

  s::DrawFst(*fst, isyms.get(), osyms.get(), ssyms.get(), FLAGS_acceptor,
             FLAGS_title, FLAGS_width, FLAGS_height, FLAGS_portrait,
             FLAGS_vertical, FLAGS_ranksep, FLAGS_nodesep, FLAGS_fontsize,
             FLAGS_precision, FLAGS_float_format, FLAGS_show_weight_one,
             &ostrm, dest);

  return 0;
}
