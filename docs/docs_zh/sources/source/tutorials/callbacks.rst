回调
=========
NeMo 用回调的方法在训练过程中运行一系列的帮助函数。
NeMo 有三个非常有用的回调：SimpleLossLoggerCallback， 
CheckpointCallback，和 EvaluatorCallback。
回调在 train() 函数之前就要定义好，然后传给 train() 函数。
例如，一个常见的训练脚本是这样的：

.. code-block:: python

    loggercallback = nemo.core.SimpleLossLoggerCallback(...)
    savercallback = nemo.core.CheckpointCallback(...)
    evalcallback = nemo.core.EvaluatorCallback(...)

    nf.train(
        callbacks=[loggercallback, savercallback, evalcallback],
        ...)

SimpleLossLoggerCallback
------------------------
SimpleLossLoggerCallback 是用来记录训练过程中的一些指标数据比如 loss 以及打印到屏幕
或者 tensorboard 上两个时间步的间隔。SimpleLossLoggerCallback 有一个必须的参数和两个我们建议
重写的参数。它接受一个 list 的 NMTensors, 这些NMTensors会在训练过程中作为 print_func()，
get_tb_values() 和 log_to_tb_func() 函数的输入。两个推荐重写的参数是 print_func() 和
get_tb_values() 或者 log_to_tb_func() 任选其一。

print_func() 应该用来记录打印到屏幕上的值。我们推荐使用 nemo.logging.info()
来取代 print() 函数。比如，可以这么打印 loss 值：

.. code-block:: python

    def my_print_func(tensors):
        nemo.logging.info(f"Loss {tensors[0]}")

我们提供了两个方法来打印到 tensorboard: get_tb_values() 和
log_to_tb_func()。对于记录标量的简单用例，我们推荐使用 get_tb_values()。
对于高级用例，像是图片或者音频，我们推荐用 log_to_tb_func() 函数。

get_tb_values() 用来返回需要打印到 tensorboard 的值。它应该返回一个 list，其中每个元素是一个二元组。
二元组的第一个元素是一个字符串，表示 tensorbard 标签，第二个元素是要记录的标量值。
注意我们当前只支持标量值。注意如果要用 get_tb_values()，tb_writer 也需要定义。

.. code-block:: python

    def my_get_tb_values(tensors):
        return [("Train_Loss", tensors[0])]

log_to_tb_func() 接收三个参数:
`tensorboardX.SummaryWriter <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_
，1个list的张量和当前步。接着用户可以用
SummaryWriter类来加图片，音频和其它。比如：

.. code-block:: python

    def log_to_tb_func(swriter, tensors, step):
        swriter.add_scalar("Train_Loss", tensors[0], step)
        swriter.add_audio("Train_Sample", tensors[1][0], step)

SimpleLossLoggerCallback可以像下面这样创建:

.. code-block:: python

    from functools import partial

    loggercallback = nemo.core.SimpleLossLoggerCallback(
        # 定义我们想要传给print_func和get_tb_values的张量
        tensors=[train_loss],
        # 传入我们想要用的打印函数
        print_func=partial(my_print_func),
        # 传入可以返回tensorboard标签和张量的函数
        get_tb_values=my_get_tb_values,
        # 我们想要回调这个函数的频次
        step_freq=500,
        # 我们想要用的 tensorboard writer, 如果 create_tb_writer 在 neural_factory
        # 中设置为True, 那么它会自动在 neural_factory 创建的时候被创建
        tb_writer=neural_factory.tb_writer)
    )

CheckpointCallback
------------------
CheckpointCallback 用于在训练过程中对 checkpoint 模型进行的操作，这样他们后面
就可以重新加载来做推理或者微调。CheckpointCallback 用起来很简单:

.. code-block:: python

    from functools import partial

    loggercallback = nemo.core.CheckpointCallback(
        # 保存 checkpoints 的目录
        # 注意: Neural Factory 会自动创建 checkpoint 目录
        folder=neural_factory.checkpoint_dir,
        # 如果是None, CheckpointCallback 在训练开始的时候回从 folder
        # 中加载模型
        # 否则的话, CheckpointCallback 会尝试从 load_from_folder 中加载
        load_from_folder=None,
        # Checkpointing 回调频次(步数)
        step_freq=-1,
        # Checkpointing 回调频次(轮数)
        epoch_freq=-1,
        # 需要保存的 checkpoint 数
        checkpoints_to_keep=4,
        # 设置为 True, CheckpointCallback 会触发 error 如果加载失败
        force_load=False
    )

EvaluatorCallback
-----------------
EvaluatorCallback 在评估验证中记录指标等参数到屏幕或者 tensorboard。
EvaluatorCallback 需要三个参数:
eval_tensors, user_iter_callback, user_epochs_done_callback。类似于
SimpleLossLoggerCallback，eval_tensors 是一个 list 的 NMTensors，包含了我们
想在评估验证中获取到的值。

user_iter_callback 是在评估验证中每个 batch 后都会调用的一个函数。
它总是接收两个参数 values_dict 和 global_var_dict。
values_dict 是个字典，NMTensor 的名字作为这个 batch 的 keys，计算得到的张量值作为
这个 batch 的 values。它的主要作用是把已经计算过的张量值从 values_dict 拷贝到
global_var_dict，因为 global_var_dict 是保存了 batch 之间的值，并且会最后传递给
user_epochs_done_callback 函数。

user_epochs_done_callback 是个接收 global_var_dict 为参数的函数。它的作用是
记录要打印到屏幕的相关信息，比如像是验证集上的 loss。

像是把简单的标量值打印到 tensorboard 上，user_epochs_done_callback 应该返回一个字典，
字符串是keys,标量值是 values。这个 tag 到 value 的字典会被解析，每个元素都会被记录到
tensorboard上 (需要 tensorboard writer 定义好)。

如果想使用更复杂的 tensorboard 打印记录像是图像或者音频，
EvaluatorCallback 必须要在初始化的时候传递给 tb_writer_func 函数。这个函数必须要接收一个
`tensorboardX.SummaryWriter <https://tensorboardx.readthedocs.io/en/latest/tensorboard.html>`_
参数，以及 user_epochs_done_callback 需要的参数和当前步。

我们推荐用 user_epochs_done_callback 来简单返回 global_var_dict 
从而给到 tb_writer_func 函数来处理。用户必须在 tb_writer_func 中记录所有需要的数据，
包括标量。

例如，可以参考 <nemo_dir>/examples 下面的例子。
