import torch, utils #, math, os
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch.distributed as dist
from cycling_utils import atomic_torch_save
from operator import itemgetter
from generative.losses.adversarial_loss import PatchAdversarialLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

#BLAH
## -- AUTO-ENCODER - ##

def compute_kl_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu**2 + z_sigma**2 - torch.log(z_sigma**2) - 1,
        dim=list(range(1, len(z_sigma.shape)))
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]

intensity_loss = torch.nn.L1Loss()
adv_loss = PatchAdversarialLoss(criterion="least_squares")

def generator_loss(reconstruction, images, z_mu, z_sigma, discriminator, perceptual_loss, kl_weight, perceptual_weight, adv_weight, apply_gan=True):
    recons_loss = intensity_loss(reconstruction, images)
    pcp_loss = perceptual_weight * perceptual_loss(reconstruction.float(), images.float())
    kl_loss = kl_weight * compute_kl_loss(z_mu, z_sigma)
    if apply_gan:
        logits_fake = discriminator(reconstruction)[-1]
        gan_loss = adv_weight * adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
    else:
        gan_loss = torch.tensor(0.0)
    return recons_loss, pcp_loss, kl_loss, gan_loss

def discriminator_loss(reconstruction, images, discriminator, adv_weight):
    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
    logits_real = discriminator(images.contiguous().detach())[-1]
    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
    loss_d = adv_weight * (loss_d_fake + loss_d_real) * 0.5
    return loss_d

def train_generator_one_epoch(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, train_loader, perceptual_loss, device, timer, metrics
    ):

    generator.train()
    discriminator.eval()
    apply_gan = epoch > args.warmup_epochs

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = len(train_loader)
    print(f'\nTraining / resuming epoch {epoch} from training step {train_step}\n')

    for images in train_loader:

        images = images.to(device)
        timer.report(f'train batch {train_step} to device')

        # UPDATE GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with autocast(enabled=False):

            reconstruction, z_mu, z_sigma = generator(images)
            timer.report(f'train batch {train_step} generator forward')

            if train_step == 1:
                torch.save({"images": images, "reconstruction": reconstruction}, "failing_example.pt")
                print('Saved example.')

            recons_loss, pcp_loss, kl_loss, gan_loss = generator_loss(
                reconstruction, images, z_mu, z_sigma, discriminator, perceptual_loss, 
                args.kl_weight, args.perceptual_weight, args.adv_weight, apply_gan=apply_gan
            )
            loss_g = recons_loss + pcp_loss + kl_loss + gan_loss

            timer.report(f'train batch {train_step} generator loss: {loss_g.item():.3f}, [recons: {recons_loss.item():.3f}, pcp: {pcp_loss.item():.3f}, kl: {kl_loss.item():.3f}, gan: {gan_loss.item():.3f}]')

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        timer.report(f'train batch {train_step} generator backward')

        # UPDATE DISCRIMINATOR
        if apply_gan:
            
            discriminator.train()
            optimizer_d.zero_grad(set_to_none=True)
            with autocast(enabled=False):
                
                loss_d = discriminator_loss(
                    reconstruction, images, discriminator, args.adv_weight
                )
                timer.report(f'train batch {train_step} discriminator loss {loss_d.item():.3f}')
            
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
                timer.report(f'train batch {train_step} discriminator backward')
        else:
            loss_d = torch.tensor(0.0)

        metrics["train"].update({
            "train_images_seen":len(images), "recons_loss":recons_loss.item(), "kl_loss":kl_loss.item(), 
            "p_loss":pcp_loss.item(), "gan_loss":gan_loss.item(), "loss_g":loss_g.item(), "loss_d": loss_d.item()
        })
        metrics["train"].reduce() # Reduce metrics accross nodes

        train_images_seen = metrics["train"].local["train_images_seen"]
        metric_names = ("recons_loss",  "kl_loss", "p_loss", "gan_loss", "loss_g", "loss_d")
        metric_vals = torch.tensor(itemgetter(*metric_names)(metrics["train"].local)) / train_images_seen
        metrics["train"].reset_local()
        print("Epoch [{}] Step [{}/{}], gen_loss: {:.3f}, disc_loss: {:.3f}".format(epoch, train_step, total_steps, metric_vals[-2], metric_vals[-1]))

        timer.report(f'train batch {train_step} metrics update')

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size

        if (train_step + 1) == total_steps:
            metrics["train"].end_epoch()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
            writer = SummaryWriter(log_dir=args.tboard_path)
            total_progress = train_step + epoch * total_steps
            writer.add_scalar("LossComponents/Reconstruction", recons_loss, total_progress)
            writer.add_scalar("LossComponents/KL_Divergence", kl_loss, total_progress)
            writer.add_scalar("LossComponents/Perceptual", pcp_loss, total_progress)
            writer.add_scalar("LossComponents/Adversarial", gan_loss, total_progress)
            writer.add_scalar("SummaryLoss/Generator", loss_g, total_progress)
            writer.add_scalar("SummaryLoss/Discriminator", loss_d, total_progress)
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
            atomic_torch_save(checkpoint, args.resume)

    gen_loss = metrics["train"].epoch_reports[-1]["loss_g"] / metrics["train"].epoch_reports[-1]["train_images_seen"]
    disc_loss = metrics["train"].epoch_reports[-1]["loss_d"] / metrics["train"].epoch_reports[-1]["train_images_seen"]
    print("Epoch [{}] :: gen_loss: {:,.3f}, disc_loss: {:,.3f}".format(epoch, gen_loss, disc_loss))


def evaluate_generator(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, val_loader, device, timer, metrics
    ):

    generator.eval()
    val_step = val_sampler.progress // val_loader.batch_size
    last_step = len(val_loader) - 1
    print(f'\nEvaluating / resuming epoch {epoch} from eval step {val_step}\n')

    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
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

            if utils.is_main_process(): # Checkpointing every batch

                if val_step == last_step:

                    val_loss = metrics["val"].agg["val_loss"] / metrics["val"].agg["val_images_seen"]
                    historic_losses = [r["val_loss"]/r["val_images_seen"] for r in metrics["val"].epoch_reports]
                    if historic_losses:
                        new_best = val_loss < min(historic_losses)
                    else:
                        new_best = True
                    if new_best:
                        save_path = "models/generator.pt"
                        atomic_torch_save({
                            "best_epoch": epoch, 
                            "best_val_loss": val_loss, 
                            "best_model": generator.module.state_dict()
                        }, save_path)
                        timer.report(f"Saving best model")
                
                    writer = SummaryWriter(log_dir=args.tboard_path)
                    writer.add_scalar("Val/Reconstruction", val_loss, epoch)
                    plottable = torch.cat((images, reconstruction))
                    grid = make_grid(plottable, nrow=len(images))
                    writer.add_image('Val/ImageBatch', grid, epoch)

                    writer.flush()
                    writer.close()
                    metrics["val"].end_epoch()

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
                atomic_torch_save(checkpoint, args.resume)


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
