// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_SCRIPT_PROJECT_H_
#define FST_SCRIPT_PROJECT_H_

#include <fst/project.h>  // for ProjectType
#include <fst/script/arg-packs.h>
#include <fst/script/fst-class.h>

namespace fst {
namespace script {

using ProjectArgs = args::Package<MutableFstClass *, ProjectType>;

template <class Arc>
void Project(ProjectArgs *args) {
  MutableFst<Arc> *ofst = args->arg1->GetMutableFst<Arc>();
  Project(ofst, args->arg2);
}

void Project(MutableFstClass *ofst, ProjectType project_type);

}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_PROJECT_H_
