
from nemo.core.neural_types import AxisKind, NeuralType    
from collections import defaultdict

def get_io_names(types, disabled_names):
    names = list(types.keys())
    for name in disabled_names:
        if name in names:
            names.remove(name)
    return names

def extract_dynamic_axes(name: str, ntype: NeuralType):
    """
        This method will extract BATCH and TIME dimension ids from each provided input/output name argument.
    
        For example, if module/model accepts argument named "input_signal" with type corresponding to [Batch, Time, Dim]
        shape, then the returned result should contain "input_signal" -> [0, 1] because Batch and Time are dynamic axes
        as they can change from call to call during inference.
    
        Args:
            name: Name of input or output parameter
            ntype: Corresponding Neural Type
    
        Returns:

        """
    def unpack_nested_neural_type(neural_type):
        if type(neural_type) in (list, tuple):
            return unpack_nested_neural_type(neural_type[0])
        return neural_type

    dynamic_axes = defaultdict(list)
    if type(ntype) in (list, tuple):
        ntype = unpack_nested_neural_type(ntype)

    if ntype.axes:
        for ind, axis in enumerate(ntype.axes):
            if axis.kind in [AxisKind.Batch, AxisKind.Time, AxisKind.Width, AxisKind.Height]:
                dynamic_axes[name].append(ind)
    return dynamic_axes

def get_dynamic_axes(types, names):
    dynamic_axes = defaultdict(list)
    for name in names:
        if name in types:
            dynamic_axes.update(extract_dynamic_axes(name, types[name]))
    return dynamic_axes

def to_onnxrt_input(input_names, input_list, input_dict):
    odict = {}
    for k, v in input_dict.items():
        odict[k] = v.cpu().numpy()
    for i, input in enumerate(input_list):
        if type(input) in (list, tuple):
            odict[input_names[i]] = tuple([ip.cpu().numpy() for ip in input])
        else:
            odict[input_names[i]] = input.cpu().numpy()
    return odict

