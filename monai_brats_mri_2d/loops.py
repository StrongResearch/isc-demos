import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
import utils
from cycling_utils import atomic_torch_save

from generative.losses.adversarial_loss import PatchAdversarialLoss

from torch.utils.tensorboard import SummaryWriter

## -- AUTO-ENCODER - ##

def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, 
        dim=list(range(1, len(z_sigma.shape)))
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]

intensity_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")

def generator_loss(gen_images, real_images, z_mu, z_sigma, disc_net, perceptual_loss, kl_weight, perceptual_weight, adv_weight):
    # Image intrinsic qualities
    recons_loss = intensity_loss(gen_images, real_images)
    kl_loss = compute_kl_loss(z_mu, z_sigma)
    p_loss = perceptual_loss(gen_images.float(), real_images.float())
    loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
    # Discrimnator-based loss
    logits_fake = disc_net(gen_images)[-1]
    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    loss_g = loss_g + (adv_weight * generator_loss)
    return loss_g

def discriminator_loss(gen_images, real_images, disc_net, adv_weight):
    logits_fake = disc_net(gen_images.contiguous().detach())[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    logits_real = disc_net(real_images.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
    loss_d = adv_weight * discriminator_loss
    return loss_d


def train_generator_one_epoch(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
        metrics
    ):

    # Obtained from scripts.losses.generator_loss
    kl_weight = 1e-6
    perceptual_weight = 1.0
    adv_weight = 0.5
    # From tutorial ?
    # generator_warm_up_n_epochs = 10

    generator.train()
    discriminator.train()

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for step, batch in enumerate(train_loader):

        images = batch["image"].to(device)
        timer.report(f'train batch {train_step} to device')

        # TRAIN GENERATOR

        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=True):

            reconstruction, z_mu, z_sigma = generator(images)
            timer.report(f'train batch {train_step} generator forward')

            loss_g = generator_loss(
                reconstruction, images, z_mu, z_sigma, discriminator, perceptual_loss, 
                kl_weight, perceptual_weight, adv_weight
            )
            timer.report(f'train batch {train_step} generator loss: {loss_g.item():.3f}')

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        timer.report(f'train batch {train_step} generator backward')

        # TRAIN DISCRIMINATOR

        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=True):

            loss_d = discriminator_loss(
                reconstruction, images, discriminator, adv_weight
            )
            timer.report(f'train batch {train_step} discriminator loss {loss_d.item():.3f}')

        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()
        timer.report(f'train batch {train_step} discriminator backward')

        # Reduce metrics accross nodes
        metrics["train"].update({"train_images_seen":len(images), "loss_g":loss_g.item(), "loss_d": loss_d.item()})
        metrics["train"].reduce()

        gen_loss = metrics["train"].local["loss_g"] / metrics["train"].local["train_images_seen"]
        disc_loss = metrics["train"].local["loss_d"] / metrics["train"].local["train_images_seen"]
        print("Epoch [{}] Step [{}/{}], gen_loss: {:.3f}, disc_loss: {:.3f}".format(epoch, train_step, total_steps, gen_loss, disc_loss))

        metrics["train"].reset_local()

        timer.report(f'train batch {train_step} metrics update')

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size
        
        if train_step == total_steps:
            metrics["train"].end_epoch()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
            writer = SummaryWriter(log_dir=args.tboard_path)
            writer.add_scalar("Train/gen_loss", gen_loss, train_step + epoch * total_steps)
            writer.add_scalar("Train/disc_loss", disc_loss, train_step + epoch * total_steps)
            writer.flush()
            writer.close()
            checkpoint = {
                # Universals
                "args": args,
                "epoch": epoch,
                # State variables
                "generator": generator.module.state_dict(),
                "discriminator": discriminator.module.state_dict(),
                "optimizer_g": optimizer_g.state_dict(),
                "optimizer_d": optimizer_d.state_dict(),
                "scaler_g": scaler_g.state_dict(),
                "scaler_d": scaler_d.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                # Metrics
                "metrics": metrics,
            }
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    gen_loss = metrics["train"].epoch_reports[-1]["loss_g"] / metrics["train"].epoch_reports[-1]["train_images_seen"]
    disc_loss = metrics["train"].epoch_reports[-1]["loss_d"] / metrics["train"].epoch_reports[-1]["train_images_seen"]
    print("Epoch [{}] :: gen_loss: {:,.3f}, disc_loss: {:,.3f}".format(epoch, gen_loss, disc_loss))
    return generator, timer, metrics


def evaluate_generator(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
        metrics
    ):

    generator.eval()

    val_step = val_sampler.progress // val_loader.batch_size
    total_steps = int(len(val_sampler) / val_loader.batch_size)
    print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

    with torch.no_grad():
        for batch in val_loader:

            images = batch["image"].to(device)
            timer.report(f'eval batch {val_step} to device')

            with autocast(enabled=True):

                reconstruction, _, _ = generator(images)
                timer.report(f'eval batch {val_step} forward')
                recons_loss = F.l1_loss(images.float(), reconstruction.float())
                timer.report(f'eval batch {val_step} recons_loss')

            metrics["val"].update({"val_images_seen": len(images), "val_loss": recons_loss.item()})
            metrics["val"].reduce()
            metrics["val"].reset_local()

            timer.report(f'eval batch {val_step} metrics update')

            ## Checkpointing
            print(f"Saving checkpoint at epoch {epoch} val batch {val_step}")
            val_sampler.advance(len(images))
            val_step = val_sampler.progress // val_loader.batch_size

            if val_step == total_steps:
                val_loss = metrics["val"].agg["val_loss"] / metrics["val"].agg["val_images_seen"]
                if utils.is_main_process():
                    writer = SummaryWriter(log_dir=args.tboard_path)
                    writer.add_scalar("Val/loss", val_loss, epoch)
                    writer.flush()
                    writer.close()
                print(f"Epoch {epoch} val loss: {val_loss:.4f}")
                metrics["val"].end_epoch()

            if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch
                checkpoint = {
                    # Universals
                    "args": args,
                    "epoch": epoch,
                    # State variables
                    "generator": generator.module.state_dict(),
                    "discriminator": discriminator.module.state_dict(),
                    "optimizer_g": optimizer_g.state_dict(),
                    "optimizer_d": optimizer_d.state_dict(),
                    "scaler_g": scaler_g.state_dict(),
                    "scaler_d": scaler_d.state_dict(),
                    "train_sampler": train_sampler.state_dict(),
                    "val_sampler": val_sampler.state_dict(),
                    # Metrics
                    "metrics": metrics,
                }
                timer = atomic_torch_save(checkpoint, args.resume, timer)

    return timer, metrics


## -- DIFFUSION MODEL - ##

def train_diffusion_one_epoch(
        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
        train_sampler, val_sampler, lr_scheduler, device, timer, metrics
    ):

    unet.train()
    generator.eval()

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = len(train_sampler) // train_loader.batch_size
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for step, batch in enumerate(train_loader):

        images = batch["image"].to(device)
        timer.report(f'train batch {train_step} to device')

        optimizer_u.zero_grad(set_to_none=True)

        with autocast(enabled=True):

            z_mu, z_sigma = generator.encode(images)
            timer.report(f'train batch {train_step} generator encoded')
            z = generator.sampling(z_mu, z_sigma)
            timer.report(f'train batch {train_step} generator sampling')
            noise = torch.randn_like(z).to(device)
            timer.report(f'train batch {train_step} noise')
            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
            timer.report(f'train batch {train_step} timesteps')
            noise_pred = inferer(inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=generator)
            timer.report(f'train batch {train_step} noise_pred')
            loss = F.mse_loss(noise_pred.float(), noise.float())
            timer.report(f'train batch {train_step} loss')

        scaler_u.scale(loss).backward()
        scaler_u.step(optimizer_u)
        scaler_u.update()
        lr_scheduler.step()
        timer.report(f'train batch {train_step} unet backward')

        # Reduce metrics accross nodes
        metrics["train"].update({"images_seen":len(images), "loss":loss.item()})
        metrics["train"].reduce()

        recons_loss = metrics["train"].local["loss"] / metrics["train"].local["images_seen"]
        print("Epoch [{}] Step [{}/{}] :: loss: {:,.3f}".format(epoch, train_step, total_steps, recons_loss))

        metrics["train"].reset_local()

        timer.report(f'train batch {train_step} metrics update')

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size

        if train_step == total_steps:
            metrics["train"].end_epoch()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch

            writer = SummaryWriter(log_dir=args.tboard_path)
            writer.add_scalar("Train/loss", recons_loss, train_step + epoch * total_steps)
            writer.flush()
            writer.close()

            checkpoint = {
                # Universals
                "args": args,
                "epoch": epoch,
                # State variables
                "unet": unet.module.state_dict(),
                "optimizer_u": optimizer_u.state_dict(),
                "scaler_u": scaler_u.state_dict(),
                "train_sampler": train_sampler.state_dict(),
                "val_sampler": val_sampler.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                # Metrics
                "metrics": metrics,
            }
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    train_loss = metrics["train"].epoch_reports[-1]["loss"] / metrics["train"].epoch_reports[-1]["images_seen"]
    print("Epoch [{}] :: epoch_loss: {:,.3f}".format(epoch, train_loss))
    return unet, timer, metrics


def evaluate_diffusion(
        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
        train_sampler, val_sampler, lr_scheduler, device, timer, metrics
    ):

        unet.eval()

        val_step = val_sampler.progress // val_loader.batch_size
        total_steps = int(len(val_sampler) / val_loader.batch_size)
        print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

        with torch.no_grad():
            for step, batch in enumerate(val_loader):

                images = batch["image"].to(device)
                timer.report(f'eval batch {val_step} to device')

                with autocast(enabled=True):

                    z_mu, z_sigma = generator.encode(images)
                    timer.report(f'eval batch {val_step} generator encoded')
                    z = generator.sampling(z_mu, z_sigma)
                    timer.report(f'eval batch {val_step} generator sampling')
                    noise = torch.randn_like(z).to(device)
                    timer.report(f'eval batch {val_step} noise')
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                    timer.report(f'eval batch {val_step} timesteps')
                    noise_pred = inferer(inputs=images,diffusion_model=unet,noise=noise,timesteps=timesteps,autoencoder_model=generator)
                    timer.report(f'eval batch {val_step} noise_pred')
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    timer.report(f'eval batch {val_step} loss')

                metrics["val"].update({"images_seen": len(images), "loss": loss.item()})
                metrics["val"].reduce()
                metrics["val"].reset_local()

                timer.report(f'eval batch {val_step} metrics update')

                ## Checkpointing
                print(f"Saving checkpoint at epoch {epoch} val batch {val_step}")
                val_sampler.advance(len(images))
                val_step = val_sampler.progress // val_loader.batch_size

                if val_step == total_steps:
                    metrics["val"].end_epoch()

                if utils.is_main_process() and val_step % 1 == 0: # Checkpointing every batch
                    print(f"Saving checkpoint at epoch {epoch} train batch {val_step}")
                    checkpoint = {
                        # Universals
                        "args": args,
                        "epoch": epoch,
                        # State variables
                        "unet": unet.module.state_dict(),
                        "optimizer_u": optimizer_u.state_dict(),
                        "scaler_u": scaler_u.state_dict(),
                        "train_sampler": train_sampler.state_dict(),
                        "val_sampler": val_sampler.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        # Metrics
                        "metrics": metrics,
                    }
                    timer = atomic_torch_save(checkpoint, args.resume, timer)

        # val_loss = metrics["val"].agg[metrics["val"].map["val_loss"]] / metrics["val"].agg[metrics["val"].map["val_images_seen"]]
        val_loss = metrics["val"].epoch_reports[-1]["loss"] / metrics["val"].epoch_reports[-1]["images_seen"]
        if utils.is_main_process():
            writer = SummaryWriter(log_dir=args.tboard_path)
            writer.add_scalar("Val/loss", val_loss, epoch)
            writer.flush()
            writer.close()
        print(f"Epoch [{epoch}] :: diff val loss: {val_loss:.4f}")

        return timer, metrics
