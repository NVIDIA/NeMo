// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.
//
// Convenience templates for defining arg packs for the FstClass operations.
//
// See operation-templates.h for a discussion about why these are needed; the
// short story is that all FstClass operations must be implemented by a version
// that takes one argument, most likely a struct bundling all the logical
// arguments together. These template structs provide convenient ways to specify
// these bundles (e.g., by means of appropriate typedefs).
//
// The ArgPack template is sufficient for bundling together all the args for
// a particular function. The function is assumed to be void-returning. If
// you want a space for a return value, use the WithReturnValue template
// as follows:
//
// WithReturnValue<bool, ArgPack<...>>

#ifndef FST_SCRIPT_ARG_PACKS_H_
#define FST_SCRIPT_ARG_PACKS_H_

namespace fst {
namespace script {
namespace args {

// Sentinel value that means "no arg here."
class none_type {};

// Base arg pack template class. Specializations follow that allow
// fewer numbers of arguments (down to 2). If the maximum number of arguments
// increases, you will need to change three things:
//
//   1) Add more template parameters to this template
//   2) Add more specializations to allow fewer numbers of parameters than
//      the new max.
//   3) Add extra none_types to all existing specializations to fill
//      the new slots.

// 9 args (the maximum).
template <class T1, class T2 = none_type, class T3 = none_type,
          class T4 = none_type, class T5 = none_type, class T6 = none_type,
          class T7 = none_type, class T8 = none_type, class T9 = none_type>
struct Package {
  T1 arg1;
  T2 arg2;
  T3 arg3;
  T4 arg4;
  T5 arg5;
  T6 arg6;
  T7 arg7;
  T8 arg8;
  T9 arg9;

  Package(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7,
          T8 arg8, T9 arg9)
      : arg1(arg1),
        arg2(arg2),
        arg3(arg3),
        arg4(arg4),
        arg5(arg5),
        arg6(arg6),
        arg7(arg7),
        arg8(arg8),
        arg9(arg9) {}
};

// 8 args.
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7,
          class T8>
struct Package<T1, T2, T3, T4, T5, T6, T7, T8, none_type> {
  T1 arg1;
  T2 arg2;
  T3 arg3;
  T4 arg4;
  T5 arg5;
  T6 arg6;
  T7 arg7;
  T8 arg8;

  Package(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7,
          T8 arg8)
      : arg1(arg1),
        arg2(arg2),
        arg3(arg3),
        arg4(arg4),
        arg5(arg5),
        arg6(arg6),
        arg7(arg7),
        arg8(arg8) {}
};

// 7 args.
template <class T1, class T2, class T3, class T4, class T5, class T6, class T7>
struct Package<T1, T2, T3, T4, T5, T6, T7, none_type, none_type> {
  T1 arg1;
  T2 arg2;
  T3 arg3;
  T4 arg4;
  T5 arg5;
  T6 arg6;
  T7 arg7;

  Package(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7)
      : arg1(arg1),
        arg2(arg2),
        arg3(arg3),
        arg4(arg4),
        arg5(arg5),
        arg6(arg6),
        arg7(arg7) {}
};

// 6 args.
template <class T1, class T2, class T3, class T4, class T5, class T6>
struct Package<T1, T2, T3, T4, T5, T6, none_type, none_type, none_type> {
  T1 arg1;
  T2 arg2;
  T3 arg3;
  T4 arg4;
  T5 arg5;
  T6 arg6;

  Package(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
      : arg1(arg1),
        arg2(arg2),
        arg3(arg3),
        arg4(arg4),
        arg5(arg5),
        arg6(arg6) {}
};

// 5 args.
template <class T1, class T2, class T3, class T4, class T5>
struct Package<T1, T2, T3, T4, T5, none_type, none_type, none_type, none_type> {
  T1 arg1;
  T2 arg2;
  T3 arg3;
  T4 arg4;
  T5 arg5;

  Package(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
      : arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4), arg5(arg5) {}
};

// 4 args.
template <class T1, class T2, class T3, class T4>
struct Package<T1, T2, T3, T4, none_type, none_type, none_type, none_type,
               none_type> {
  T1 arg1;
  T2 arg2;
  T3 arg3;
  T4 arg4;

  Package(T1 arg1, T2 arg2, T3 arg3, T4 arg4)
      : arg1(arg1), arg2(arg2), arg3(arg3), arg4(arg4) {}
};

// 3 args.
template <class T1, class T2, class T3>
struct Package<T1, T2, T3, none_type, none_type, none_type, none_type,
               none_type, none_type> {
  T1 arg1;
  T2 arg2;
  T3 arg3;

  Package(T1 arg1, T2 arg2, T3 arg3) : arg1(arg1), arg2(arg2), arg3(arg3) {}
};

// 2 args (the minimum).
template <class T1, class T2>
struct Package<T1, T2, none_type, none_type, none_type, none_type, none_type,
               none_type, none_type> {
  T1 arg1;
  T2 arg2;

  Package(T1 arg1, T2 arg2) : arg1(arg1), arg2(arg2) {}
};

// Tack this on to an existing arg pack to add a return value. The syntax for
// accessing the args is then slightly more stilted, as you must do an extra
// member access (since the args are stored as a member of this class). The
// alternative is to declare another slew of templates for functions that return
// a value, analogous to the above.

template <class Retval, class ArgPackage>
struct WithReturnValue {
  Retval retval;
  const ArgPackage &args;

  explicit WithReturnValue(const ArgPackage &args) : args(args) {}
};

// We don't want to store a reference to a reference, if ArgPackage is already
// some reference type.
template <class Retval, class ArgPackage>
struct WithReturnValue<Retval, ArgPackage &> {
  Retval retval;
  const ArgPackage &args;

  explicit WithReturnValue(const ArgPackage &args) : args(args) {}
};

}  // namespace args
}  // namespace script
}  // namespace fst

#endif  // FST_SCRIPT_ARG_PACKS_H_
