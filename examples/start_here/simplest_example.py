# Copyright (c) 2019 NVIDIA Corporation
import nemo

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    backend=nemo.core.Backend.PyTorch, local_rank=None)

# instantiate necessary neural modules
dl = neural_factory.get_module(name="RealFunctionDataLayer", collection="toys",
                               params={"n": 10000, "batch_size": 128})
fx = neural_factory.get_module(name="TaylorNet", collection="toys",
                               params={"dim": 4})
loss = neural_factory.get_module(name="MSELoss", collection="toys",
                                 params={})

# describe activation's flow
x, y = dl()
p = fx(x=x)
lss = loss(predictions=p, target=y)

# SimpleLossLoggerCallback will print loss values to console.
# It should receive function to convert a list of
# backend-specific tensors into string
callback = nemo.core.SimpleLossLoggerCallback(
    tensor_list2string=lambda x: str(x[0].item()))
# Instantiate an optimizer to perform `train` action
optimizer = neural_factory.get_trainer(
    params={"optimization_params": {"num_epochs": 3, "lr": 0.0003}})
optimizer.train([lss], callbacks=[callback])
