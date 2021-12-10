// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Functions and classes to project an FST on to its domain or range.

#ifndef FST_LIB_PROJECT_H_
#define FST_LIB_PROJECT_H_

#include <fst/arc-map.h>
#include <fst/mutable-fst.h>


namespace fst {

// This specifies whether to project on input or output.
enum ProjectType { PROJECT_INPUT = 1, PROJECT_OUTPUT = 2 };

// Mapper to implement projection per arc.
template <class A>
class ProjectMapper {
 public:
  using FromArc = A;
  using ToArc = A;

  explicit ProjectMapper(ProjectType project_type)
      : project_type_(project_type) {}

  ToArc operator()(const FromArc &arc) {
    const auto label = project_type_ == PROJECT_INPUT ? arc.ilabel : arc.olabel;
    return ToArc(label, label, arc.weight, arc.nextstate);
  }

  MapFinalAction FinalAction() const { return MAP_NO_SUPERFINAL; }

  MapSymbolsAction InputSymbolsAction() const {
    return project_type_ == PROJECT_INPUT ? MAP_COPY_SYMBOLS
                                          : MAP_CLEAR_SYMBOLS;
  }

  MapSymbolsAction OutputSymbolsAction() const {
    return project_type_ == PROJECT_OUTPUT ? MAP_COPY_SYMBOLS
                                           : MAP_CLEAR_SYMBOLS;
  }

  uint64 Properties(uint64 props) {
    return ProjectProperties(props, project_type_ == PROJECT_INPUT);
  }

 private:
  ProjectType project_type_;
};

// Projects an FST onto its domain or range by either copying each arcs' input
// label to the output label or vice versa. This version modifies its input.
//
// Complexity:
//
//   Time: O(V + E)
//   Space: O(1)
//
// where V is the number of states and E is the number of arcs.
template <class Arc>
inline void Project(MutableFst<Arc> *fst, ProjectType project_type) {
  ArcMap(fst, ProjectMapper<Arc>(project_type));
  if (project_type == PROJECT_INPUT) fst->SetOutputSymbols(fst->InputSymbols());
  if (project_type == PROJECT_OUTPUT) {
    fst->SetInputSymbols(fst->OutputSymbols());
  }
}

// Projects an FST onto its domain or range by either copying each arc's input
// label to the output label or vice versa. This version is a delayed FST.
//
// Complexity:
//
//   Time: O(v + e)
//   Space: O(1)
//
// where v is the number of states visited and e is the number of arcs visited.
// Constant time and to visit an input state or arc is assumed and exclusive of
// caching.
template <class A>
class ProjectFst : public ArcMapFst<A, A, ProjectMapper<A>> {
 public:
  using FromArc = A;
  using ToArc = A;

  using Impl = internal::ArcMapFstImpl<A, A, ProjectMapper<A>>;

  ProjectFst(const Fst<A> &fst, ProjectType project_type)
      : ArcMapFst<A, A, ProjectMapper<A>>(fst, ProjectMapper<A>(project_type)) {
    if (project_type == PROJECT_INPUT) {
      GetMutableImpl()->SetOutputSymbols(fst.InputSymbols());
    }
    if (project_type == PROJECT_OUTPUT) {
      GetMutableImpl()->SetInputSymbols(fst.OutputSymbols());
    }
  }

  // See Fst<>::Copy() for doc.
  ProjectFst(const ProjectFst<A> &fst, bool safe = false)
      : ArcMapFst<A, A, ProjectMapper<A>>(fst, safe) {}

  // Gets a copy of this ProjectFst. See Fst<>::Copy() for further doc.
  ProjectFst<A> *Copy(bool safe = false) const override {
    return new ProjectFst(*this, safe);
  }

 private:
  using ImplToFst<Impl>::GetMutableImpl;
};

// Specialization for ProjectFst.
template <class A>
class StateIterator<ProjectFst<A>>
    : public StateIterator<ArcMapFst<A, A, ProjectMapper<A>>> {
 public:
  explicit StateIterator(const ProjectFst<A> &fst)
      : StateIterator<ArcMapFst<A, A, ProjectMapper<A>>>(fst) {}
};

// Specialization for ProjectFst.
template <class A>
class ArcIterator<ProjectFst<A>>
    : public ArcIterator<ArcMapFst<A, A, ProjectMapper<A>>> {
 public:
  using StateId = typename A::StateId;

  ArcIterator(const ProjectFst<A> &fst, StateId s)
      : ArcIterator<ArcMapFst<A, A, ProjectMapper<A>>>(fst, s) {}
};

// Useful alias when using StdArc.
using StdProjectFst = ProjectFst<StdArc>;

}  // namespace fst

#endif  // FST_LIB_PROJECT_H_
