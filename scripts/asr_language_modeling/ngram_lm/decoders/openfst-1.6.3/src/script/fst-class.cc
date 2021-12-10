// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// These classes are only recommended for use in high-level scripting
// applications. Most users should use the lower-level templated versions
// corresponding to these classes.

#include <istream>

#include <fst/log.h>

#include <fst/equal.h>
#include <fst/fst-decl.h>
#include <fst/reverse.h>
#include <fst/union.h>
#include <fst/script/fst-class.h>
#include <fst/script/register.h>

namespace fst {
namespace script {

// Registration.

REGISTER_FST_CLASSES(StdArc);
REGISTER_FST_CLASSES(LogArc);
REGISTER_FST_CLASSES(Log64Arc);

// FstClass methods.

template <class FstT>
FstT *ReadFst(std::istream &istrm, const string &fname) {
  if (!istrm) {
    LOG(ERROR) << "ReadFst: Can't open file: " << fname;
    return nullptr;
  }
  FstHeader hdr;
  if (!hdr.Read(istrm, fname)) return nullptr;
  FstReadOptions read_options(fname, &hdr);
  const auto arc_type = hdr.ArcType();
  const auto reader =
      IORegistration<FstT>::Register::GetRegister()->GetReader(arc_type);
  if (!reader) {
    LOG(ERROR) << "ReadFst: Unknown arc type: " << arc_type;
    return nullptr;
  }
  return reader(istrm, read_options);
}

FstClass *FstClass::Read(const string &fname) {
  if (!fname.empty()) {
    std::ifstream istrm(fname, std::ios_base::in | std::ios_base::binary);
    return ReadFst<FstClass>(istrm, fname);
  } else {
    return ReadFst<FstClass>(std::cin, "standard input");
  }
}

FstClass *FstClass::Read(std::istream &istrm, const string &source) {
  return ReadFst<FstClass>(istrm, source);
}

FstClass *FstClass::ReadFromString(const string &fst_string) {
  std::istringstream istrm(fst_string);
  return ReadFst<FstClass>(istrm, "StringToFst");
}

const string FstClass::WriteToString() const {
  std::ostringstream ostrm;
  Write(ostrm, FstWriteOptions("StringToFst"));
  return ostrm.str();
}

bool FstClass::WeightTypesMatch(const WeightClass &weight,
                                const string &op_name) const {
  if (WeightType() != weight.Type()) {
    FSTERROR() << "FST and weight with non-matching weight types passed to "
               << op_name << ": " << WeightType() << " and " << weight.Type();
    return false;
  }
  return true;
}

// MutableFstClass methods.

MutableFstClass *MutableFstClass::Read(const string &fname, bool convert) {
  if (convert == false) {
    if (!fname.empty()) {
      std::ifstream in(fname, std::ios_base::in | std::ios_base::binary);
      return ReadFst<MutableFstClass>(in, fname);
    } else {
      return ReadFst<MutableFstClass>(std::cin, "standard input");
    }
  } else {  // Converts to VectorFstClass if not mutable.
    FstClass *ifst = FstClass::Read(fname);
    if (!ifst) return nullptr;
    if (ifst->Properties(fst::kMutable, false)) {
      return static_cast<MutableFstClass *>(ifst);
    } else {
      MutableFstClass *ofst = new VectorFstClass(*ifst);
      delete ifst;
      return ofst;
    }
  }
}

// VectorFstClass methods.

VectorFstClass *VectorFstClass::Read(const string &fname) {
  if (!fname.empty()) {
    std::ifstream in(fname, std::ios_base::in | std::ios_base::binary);
    return ReadFst<VectorFstClass>(in, fname);
  } else {
    return ReadFst<VectorFstClass>(std::cin, "standard input");
  }
}

IORegistration<VectorFstClass>::Entry GetVFSTRegisterEntry(
    const string &arc_type) {
  return IORegistration<VectorFstClass>::Register::GetRegister()->GetEntry(
      arc_type);
}

VectorFstClass::VectorFstClass(const string &arc_type)
    : MutableFstClass(GetVFSTRegisterEntry(arc_type).creator()) {
  if (Properties(kError, true) == kError) {
    FSTERROR() << "VectorFstClass: Unknown arc type: " << arc_type;
  }
}

VectorFstClass::VectorFstClass(const FstClass &other)
    : MutableFstClass(GetVFSTRegisterEntry(other.ArcType()).converter(other)) {}

}  // namespace script
}  // namespace fst
