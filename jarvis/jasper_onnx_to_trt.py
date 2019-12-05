import tensorrt as trt

def build_engine_from_parser(onnx_path, max_seq_len=256, trt_fp16=True, verbose=True, 
                             engine_batch_size=8, max_batch_size=64, max_workspace_size=0,
                             dynamic_shape=True, transpose=False):
    '''Builds TRT engine from an ONNX file
    Note that network output 1 is unmarked so that the engine will not use
    vestigial length calculations associated with masked_fill
    '''
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = max_batch_size

    if trt_fp16:
        builder.fp16_mode = True
        print("Optimizing for FP16")
        config_flags = 1 << int(trt.BuilderFlag.FP16) # | 1 << int(trt.BuilderFlag.STRICT_TYPES)
        max_size = 4*1024*1024*1024
        max_len = max_seq_len
    else:
        config_flags = 0
        max_size = 4*1024*1024*1024
        max_len = max_seq_len
    if max_workspace_size > 0:
        builder.max_workspace_size = max_workspace_size
    else:
        builder.max_workspace_size = max_size
        
    config = builder.create_builder_config()
    config.flags = config_flags
    
    if dynamic_shape:
        profile = builder.create_optimization_profile()
        if transpose:
            profile.set_shape("FEATURES", min=(1,192,64), opt=(engine_batch_size,256,64), max=(builder.max_batch_size, max_len, 64))
        else:
            profile.set_shape("FEATURES", min=(1,64,192), opt=(engine_batch_size,64,256), max=(builder.max_batch_size, 64, max_len))        
        config.add_optimization_profile(profile)    
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(onnx_path, 'rb') as model:
            parsed = parser.parse(model.read())
            print ("Parsing returned ", parsed, "dynamic_shape= " , dynamic_shape, "\n")
            return builder.build_engine(network, config=config)
