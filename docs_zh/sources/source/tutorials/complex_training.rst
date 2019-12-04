复杂训练流程（GAN例子）
========================================

目前为止，训练样本在所有可训练的神经模块中用了一个优化器来优化一个损失函数。
NeMo进一步扩充了用例，这些用例会用到多个损失函数和多个优化器。

.. note::
    我们所有的流程都只支持一个数据层。

多个损失函数
---------------
以我们之前的Hello World为例子。假设我们现在想要优化一个平方误差损失函数和l1损失函数。
我们可以把这代表着两个损失函数的张量传给
:meth:`NeuralFactory.train()<nemo.core.neural_factory.NeuralModuleFactory.train>`。
下面是一个例子：

.. code-block:: python

    ### 和前面的例子一样 ###
    import nemo

    # 用支持的后端初始化Neural Factory
    nf = nemo.core.NeuralModuleFactory()

    # 初始化必要的神经模块
    dl = nemo.tutorials.RealFunctionDataLayer(
        n=10000, batch_size=128)
    fx = nemo.tutorials.TaylorNet(dim=4)
    mse_loss = nemo.tutorials.MSELoss()

    # 描述激活流
    x, y = dl()
    p = fx(x=x)
    mse_loss_tensor = mse_loss(predictions=p, target=y)

    ### 新的代码从这里开始 ###
    # 我们定义新的LossNM和新的损失张量
    l1_loss = nemo.tutorials.L1Loss()
    l1_loss_tensor = l1_loss(predictions=p, target=y)

    # SimpleLossLoggerCallback会把损失函数的值打印到控制台
    # 更新打印函数加入两个损失函数的张量
    callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[l1_loss_tensor, mse_loss_tensor],
        print_func=lambda x: print(
            f'Train Loss: {str(x[0].item() + x[1].item())}')
    )

    # 用两个损失函数张量激活“训练”
    nf.train([mse_loss_tensor, l1_loss_tensor], callbacks=[callback],
             optimization_params={"num_epochs": 3, "lr": 0.0003},
             optimizer="sgd")

我们可以进一步拓展这个优化器使得每次优化一个损失函数。比如说，我们不想
根据mse_loss + l1_loss计算梯度，我们想先根据mse_loss计算梯度，做一个
权重更新，然后根据l1_loss求导，再做另一个权重更新。那么我们必须要定义我们
的训练循环：

.. code-block:: python

    ### 和前面的例子一样 ###
    import nemo

    # 用支持的后端做初始化
    nf = nemo.core.NeuralModuleFactory()

    # 初始化必要的神经模块
    dl = nemo.tutorials.RealFunctionDataLayer(
        n=10000, batch_size=128)
    fx = nemo.tutorials.TaylorNet(dim=4)
    mse_loss = nemo.tutorials.MSELoss()
    l1_loss = nemo.tutorials.L1Loss()

    # 描绘激活流
    x, y = dl()
    p = fx(x=x)
    mse_loss_tensor = mse_loss(predictions=p, target=y)
    l1_loss_tensor = l1_loss(predictions=p, target=y)

    # SimpleLossLoggerCallback把损失函数值打印到控制台
    callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[l1_loss_tensor, mse_loss_tensor],
        print_func=lambda x: print(
            f'L1 Loss: {str(x[0].item())}'
            f'MSE Loss: {str(x[1].item())}')
    )

    ### 新代码从这里开始 ###
    # 我们需要创建优化器手动开启复杂的训练流程
    optimizer = nf.create_optimizer(
        optimizer="sgd",
        # 注意我们需要指定我们想要优化的神经模块和神经模块张量
        things_to_optimize=[l1_loss_tensor, mse_loss_tensor],
        optimizer_params={"lr": 0.0003})

    # 现在我们定义training_loop,这是一个二元组的list,
    # 每个二元组有两个元素
    # 第一个是优化器
    # 第二个是我们想要优化的张量
    training_loop = [
        # 首先是优化MSE，做一个权重更新
        (optimizer, [mse_loss_tensor]),
        # 然后是优化L1，做一个权重更新
        (optimizer, [l1_loss_tensor]),
    ]

    # 触发"训练"
    # 注意，我们不在需要传奇优化器，因为我们已经有了一个training_loop
    nf.train(training_loop, callbacks=[callback],
             optimization_params={"num_epochs": 3})

多个优化器和多个损失函数
---------------------------------------
NeMo也支持用户想要定义多个优化器的用例。一个这样的例子是GAN，我们想要给生成器
一个优化器，给判别器一个优化器。我们也想要优化不同的损失函数。
这个是来自examples/images/gan.py下面的支持这种操作的代码：

.. code-block:: python

    ...

    # 创建神经元模块
    generator = nemo_simple_gan.SimpleGenerator(
        batch_size=batch_size)
    discriminator = nemo_simple_gan.SimpleDiscriminator()

    ...

    # 创建损失函数张量
    # Loss 1: 插值的图像的损失函数
    interpolated_loss = disc_loss(decision=interpolated_decision)
    # Loss 2: 真实图片的损失函数
    real_loss = neg_disc_loss(decision=real_decision)
    # Loss 3: WGAN梯度惩罚项
    grad_penalty = disc_grad_penalty(
        interpolated_image=interpolated_image,
        interpolated_decision=interpolated_decision)

    ...

    # 创建优化器
    # 注意我们对于生成器和判别器分别想要一个优化器
    optimizer_G = neural_factory.create_optimizer(
        things_to_optimize=[generator],
        ...)
    optimizer_D = neural_factory.create_optimizer(
        things_to_optimize=[discriminator],
        ...)

    # 定义training_loop
    # 注意在我们的训练循环中，我们想要优化三次判别器再优化一次生成器
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
