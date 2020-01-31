# Copyright (c) 2019 NVIDIA Corporation
import argparse
import os
from shutil import copyfile

from PIL import Image
from tensorboardX import SummaryWriter

import nemo
import nemo.collections.simple_gan as nemo_simple_gan
from nemo.backends.pytorch.torchvision.helpers import compute_accuracy, eval_epochs_done_callback, eval_iter_callback

parser = argparse.ArgumentParser(description='MNIST')
parser.add_argument("--local_rank", default=None, type=int)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--num_epochs", default=5000, type=int)
parser.add_argument("--work_dir", default=None, type=str)
parser.add_argument(
    "--train_dataset",
    # set default=os.getcwd() unless your are running test
    default="~/data/mnist",
    type=str,
)
parser.add_argument("--amp_opt_level", choices=['O0', 'O1', 'O2', 'O3'], default='O0')

args = parser.parse_args()

batch_size = args.batch_size

work_dir = f"GAN_{args.amp_opt_level}"
if args.work_dir:
    work_dir = os.path.join(args.work_dir, work_dir)

# instantiate Neural Factory with supported backend
neural_factory = nemo.core.NeuralModuleFactory(
    local_rank=args.local_rank,
    optimization_level=args.amp_opt_level,
    log_dir=work_dir,
    create_tb_writer=True,
    files_to_copy=[__file__],
)

mnist_data = nemo_simple_gan.MnistGanDataLayer(
    batch_size=batch_size, shuffle=True, train=True, root=args.train_dataset
)

generator = nemo_simple_gan.SimpleGenerator()
discriminator = nemo_simple_gan.SimpleDiscriminator()
neg_disc_loss = nemo_simple_gan.DiscriminatorLoss(neg=True)
disc_loss = nemo_simple_gan.DiscriminatorLoss()
disc_grad_penalty = nemo_simple_gan.GradientPenalty(lambda_=3)
interpolater = nemo_simple_gan.InterpolateImage()

# Create generator DAG
latents, real_data, _ = mnist_data()
generated_image = generator(latents=latents)
generator_decision = discriminator(image=generated_image)
generator_loss = neg_disc_loss(decision=generator_decision)

# Create discriminator DAG
interpolated_image = interpolater(image1=real_data, image2=generated_image)
interpolated_decision = discriminator(image=interpolated_image)
real_decision = discriminator(image=real_data)
interpolated_loss = disc_loss(decision=interpolated_decision)
real_loss = neg_disc_loss(decision=real_decision)
grad_penalty = disc_grad_penalty(interpolated_image=interpolated_image, interpolated_decision=interpolated_decision,)

# Create Eval DAG
random_data = nemo_simple_gan.RandomDataLayer(batch_size=batch_size)
latents_e = random_data()
generated_image_e = generator(latents=latents_e)

losses_G = [generator_loss]
losses_D = [interpolated_loss, real_loss, grad_penalty]

# Since we want optimizers to only operate on a subset of the model, we need
# to manually create optimizers
# For single loss and single optimizer, the following steps can be skipped
# and an optimizer will be created in trainer.train()
optimizer_G = neural_factory.create_optimizer(
    optimizer="adam", things_to_optimize=[generator], optimizer_params={"lr": 1e-4, "betas": (0.5, 0.9),},
)
optimizer_D = neural_factory.create_optimizer(
    optimizer="adam", things_to_optimize=[discriminator], optimizer_params={"lr": 1e-4, "betas": (0.5, 0.9),},
)


def save_image(global_vars):
    images = global_vars["image"]
    image = images[0].squeeze(0).detach().cpu().numpy() * 255
    im = Image.fromarray(image.astype('uint8'))
    im.save(os.path.join(neural_factory.work_dir, "generated.jpeg"))


def put_tensor_in_dict(tensors, global_vars):
    global_vars["image"] = tensors[generated_image_e.unique_name][0]


eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[generated_image_e],
    user_iter_callback=put_tensor_in_dict,
    user_epochs_done_callback=lambda x: save_image(x),
    eval_step=2500,
)


def print_losses(tensors):
    g_loss, i_loss, r_loss, grad_p = tensors
    nemo.logging.info(f"Generator Loss: {g_loss}")
    nemo.logging.info(f"Interpolated Loss: {i_loss}")
    nemo.logging.info(f"Real Loss: {r_loss}")
    nemo.logging.info(f"Grad Penalty: {grad_p}")


def get_tb_name_value(tensors):
    g_loss = tensors[0]
    return [["G_LOSS", g_loss]]


logger_callback = nemo.core.SimpleLossLoggerCallback(
    tensors=[generator_loss, interpolated_loss, real_loss, grad_penalty],
    print_func=print_losses,
    get_tb_values=get_tb_name_value,
    step_freq=500,
    tb_writer=neural_factory.tb_writer,
)

checkpoint_callback = nemo.core.CheckpointCallback(folder=neural_factory.checkpoint_dir, step_freq=1000)

tensors_to_optimize = [
    (optimizer_D, losses_D),
    (optimizer_D, losses_D),
    (optimizer_D, losses_D),
    (optimizer_G, losses_G),
]
neural_factory.train(
    tensors_to_optimize=tensors_to_optimize,
    callbacks=[eval_callback, logger_callback, checkpoint_callback],
    optimization_params={"num_epochs": args.num_epochs},
)
