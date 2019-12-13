Complex Training Pipelines (GAN Example)
========================================

So far, training examples have utilized one optimizer to optimize one loss
across all Trainable Neural Modules. NeMo further extends to uses cases that
require multiple losses and multiple optimizers.

.. note::
    All of our pipelines only support one datalayer.

Multiple Losses
---------------
Taking our Hello World example from earlier, let's say that we now want to
optimize for both a square error loss and a l1 loss. We can pass both the
square error loss tensor and the l1 loss tensor to
:meth:`NeuralFactory.train()<nemo.core.neural_factory.NeuralModuleFactory.train>`.
Here's an example:

.. code-block:: python

    ### Same as previous example ###
    import nemo

    # instantiate Neural Factory with supported backend
    nf = nemo.core.NeuralModuleFactory()

    # instantiate necessary neural modules
    dl = nemo.tutorials.RealFunctionDataLayer(
        n=10000, batch_size=128)
    fx = nemo.tutorials.TaylorNet(dim=4)
    mse_loss = nemo.tutorials.MSELoss()

    # describe activation's flow
    x, y = dl()
    p = fx(x=x)
    mse_loss_tensor = mse_loss(predictions=p, target=y)

    ### New code starts here ###
    # We define our new LossNM and as well as our new loss tensor
    l1_loss = nemo.tutorials.L1Loss()
    l1_loss_tensor = l1_loss(predictions=p, target=y)

    # SimpleLossLoggerCallback will print loss values to console.
    # Update printing function to add both losses
    callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[l1_loss_tensor, mse_loss_tensor],
        print_func=lambda x: print(
            f'Train Loss: {str(x[0].item() + x[1].item())}')
    )

    # Invoke "train" action with both loss tensors
    nf.train([mse_loss_tensor, l1_loss_tensor], callbacks=[callback],
             optimization_params={"num_epochs": 3, "lr": 0.0003},
             optimizer="sgd")

We can further extend this to optimize one loss at a time. Let's say that
instead of computing derivatives and gradients with respect to
mse_loss + l1_loss, we want to first compute gradients with respect to
mse_loss, do a weight update, and then compute gradients with respect to
l1_loss, and do another weight update. Here we have to define our own training
loop.

.. code-block:: python

    ### Same as previous example ###
    import nemo

    # instantiate Neural Factory with supported backend
    nf = nemo.core.NeuralModuleFactory()

    # instantiate necessary neural modules
    dl = nemo.tutorials.RealFunctionDataLayer(
        n=10000, batch_size=128)
    fx = nemo.tutorials.TaylorNet(dim=4)
    mse_loss = nemo.tutorials.MSELoss()
    l1_loss = nemo.tutorials.L1Loss()

    # describe activation's flow
    x, y = dl()
    p = fx(x=x)
    mse_loss_tensor = mse_loss(predictions=p, target=y)
    l1_loss_tensor = l1_loss(predictions=p, target=y)

    # SimpleLossLoggerCallback will print loss values to console.
    callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[l1_loss_tensor, mse_loss_tensor],
        print_func=lambda x: print(
            f'L1 Loss: {str(x[0].item())}'
            f'MSE Loss: {str(x[1].item())}')
    )

    ### New code starts here ###
    # We need to create optimizers manually to enable complex training pipelines
    optimizer = nf.create_optimizer(
        optimizer="sgd",
        # Note we have to specify the neural modules or nmtensors that we want
        # to optimize for
        things_to_optimize=[l1_loss_tensor, mse_loss_tensor],
        optimizer_params={"lr": 0.0003})

    # Now we define our training_loop, which is a list of tuples
    # Each tuple should have two elements
    # The first element is the optimizer to use
    # The second element is the loss we want to optimize
    training_loop = [
        # Optimizer MSE first and do a weight update
        (optimizer, [mse_loss_tensor]),
        # Optimizer L1 second and do a weight update
        (optimizer, [l1_loss_tensor]),
    ]

    # Invoke "train" action
    # Note, we no longer need to pass optimizer since we have a training_loop
    nf.train(training_loop, callbacks=[callback],
             optimization_params={"num_epochs": 3})

Multiple Optimizers and Multiple Losses
---------------------------------------
NeMo additionally supports use cases where a user would want to create more
than one optimizer. One example of such a use case would be a GAN where
we want to create an optimizer for the generator and an optimizer for the
discriminator. We also want to optimize for different losses in both cases.
Here are the highlights from examples/images/gan.py that enable such behaviour.

.. code-block:: python

    ...

    # Creation of Neural Modules
    generator = nemo_simple_gan.SimpleGenerator(
        batch_size=batch_size)
    discriminator = nemo_simple_gan.SimpleDiscriminator()

    ...

    # Creation of Loss NM Tensors
    # Loss 1: Interpolated image loss
    interpolated_loss = disc_loss(decision=interpolated_decision)
    # Loss 2: Real image loss
    real_loss = neg_disc_loss(decision=real_decision)
    # Loss 3: WGAN Gradient Penalty
    grad_penalty = disc_grad_penalty(
        interpolated_image=interpolated_image,
        interpolated_decision=interpolated_decision)

    ...

    # Create optimizers
    # Note that we only want one optimizer to optimize either the generator
    # or the discriminator
    optimizer_G = neural_factory.create_optimizer(
        things_to_optimize=[generator],
        ...)
    optimizer_D = neural_factory.create_optimizer(
        things_to_optimize=[discriminator],
        ...)

    # Define training_loop
    # Note in our training loop, we want to optimize the discriminator
    # 3x more compared to our generator
    losses_G = [generator_loss]
    losses_D = [interpolated_loss, real_loss, grad_penalty]
    training_loop = [
        (optimizer_D, losses_D),
        (optimizer_D, losses_D),
        (optimizer_D, losses_D),
        (optimizer_G, losses_G),
    ]

    neural_factory.train(
        tensors_to_optimize=training_loop,
        ...)
