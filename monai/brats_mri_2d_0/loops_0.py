from tqdm import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import torch.distributed as dist
import utils
from cycling_utils import atomic_torch_save
from torch.utils.tensorboard import SummaryWriter
tb_path = "/mnt/Client/StrongUniversity/USYD-04/usyd04_adam/output_brats_mri_2d_gen/tb"

## -- AUTO-ENCODER - ##

def train_generator_one_epoch(
        args, epoch, generator, discriminator, optimizer_g, optimizer_d, train_sampler, val_sampler,
        scaler_g, scaler_d, train_loader, val_loader, perceptual_loss, adv_loss, device, timer,
        metrics
    ):

    # Maybe pull these out into args later
    kl_weight = 1e-6
    generator_warm_up_n_epochs = 3
    perceptual_weight = 0.001
    adv_weight = 0.01

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
            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            timer.report(f'train batch {train_step} recons_loss')
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            timer.report(f'train batch {train_step} p_loss')
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            timer.report(f'train batch {train_step} kl_loss')
            loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
            timer.report(f'train batch {train_step} loss_g (1)')

            if epoch > generator_warm_up_n_epochs: # Train generator for n epochs on reconstruction, KL, and perceptual loss before introducing discriminator loss

                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                timer.report(f'train batch {train_step} logits_fake from discriminator')
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                timer.report(f'train batch {train_step} generator_loss (adv_loss)')
                loss_g += adv_weight * generator_loss
                timer.report(f'train batch {train_step} loss_g (2)')

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        timer.report(f'train batch {train_step} generator backward')

        # TRAIN DISCRIMINATOR

        if epoch > generator_warm_up_n_epochs:  # Train generator for n epochs before starting discriminator training

            optimizer_d.zero_grad(set_to_none=True)

            with autocast(enabled=True):

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                timer.report(f'train batch {train_step} discriminator forward (fake)')
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                timer.report(f'train batch {train_step} loss_d_fake')
                logits_real = discriminator(images.contiguous().detach())[-1]
                timer.report(f'train batch {train_step} discriminator forward (real)')
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                timer.report(f'train batch {train_step} loss_d_real')
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                timer.report(f'train batch {train_step} discriminator_loss')
                loss_d = adv_weight * discriminator_loss
                timer.report(f'train batch {train_step} loss_d')

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            timer.report(f'train batch {train_step} discriminator backward')

        # Reduce metrics accross nodes
        metrics["train"].update({"train_images_seen":len(images), "epoch_loss":recons_loss.item()})
        if epoch > generator_warm_up_n_epochs:
            metrics["train"].update({"gen_epoch_loss":generator_loss.item(), "disc_epoch_loss":discriminator_loss.item()})
        metrics["train"].reduce_and_reset_local()

        timer.report(f'train batch {train_step} metrics update')

        recons_loss = metrics["train"].agg[metrics["train"].map["epoch_loss"]] / metrics["train"].agg[metrics["train"].map["train_images_seen"]]
        gen_loss = metrics["train"].agg[metrics["train"].map["gen_epoch_loss"]] / metrics["train"].agg[metrics["train"].map["train_images_seen"]]
        disc_loss = metrics["train"].agg[metrics["train"].map["disc_epoch_loss"]] / metrics["train"].agg[metrics["train"].map["train_images_seen"]]
        print("Epoch [{}] Step [{}/{}] :: recons_loss: {:,.3f}, gen_loss: {:,.3f}, disc_loss: {:,.3f}".format(epoch, train_step+1, total_steps, recons_loss, gen_loss, disc_loss))

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size
        
        if train_step == total_steps:
            metrics["train"].end_epoch()

        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
            writer = SummaryWriter(log_dir=tb_path)
            writer.add_scalar("recons_loss", recons_loss, step)
            writer.add_scalar("gen_loss", recons_loss, step)
            writer.add_scalar("disc_loss", recons_loss, step)
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
            metrics["val"].reduce_and_reset_local()

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

    val_loss = metrics["val"].agg[metrics["val"].map["val_loss"]] / metrics["val"].agg[metrics["val"].map["val_images_seen"]]
    if utils.is_main_process():
        writer = SummaryWriter(log_dir=tb_path)
        writer.add_scalar("val", val_loss, epoch)
        writer.flush()
        writer.close()
    print(f"Epoch {epoch} val loss: {val_loss:.4f}")

    return timer, metrics



## -- DIFFUSION MODEL - ##

def train_diffusion_one_epoch(
        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
        train_sampler, val_sampler, train_images_seen, val_images_seen, epoch_loss, val_loss, device, timer
    ):

    unet.train()
    generator.eval()

    train_step = train_sampler.progress // train_loader.batch_size
    total_steps = int(len(train_sampler) / train_loader.batch_size)
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
        timer.report(f'train batch {train_step} unet backward')

        epoch_loss += loss.item()
        train_images_seen += len(images)
        recons_loss = epoch_loss / train_images_seen
        print("Epoch [{}] Step [{}/{}] :: recons_loss: {:,.3f}".format(epoch, train_step+1, total_steps, recons_loss))

        ## Checkpointing
        print(f"Saving checkpoint at epoch {epoch} train batch {train_step}")
        train_sampler.advance(len(images))
        train_step = train_sampler.progress // train_loader.batch_size
        if utils.is_main_process() and train_step % 1 == 0: # Checkpointing every batch
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

                # Evaluation metrics
                "train_images_seen": train_images_seen,
                "val_images_seen": val_images_seen,
                "epoch_loss": epoch_loss,
                "val_loss": val_loss,
            }
            timer = atomic_torch_save(checkpoint, args.resume, timer)

    return unet, timer


def evaluate_diffusion(
        args, epoch, unet, generator, optimizer_u, scaler_u, inferer, train_loader, val_loader, 
        train_sampler, val_sampler, train_images_seen, val_images_seen, epoch_loss, val_loss, device, timer
    ):

        unet.eval()

        val_step = val_sampler.progress // val_loader.batch_size
        print(f'\nEvaluating / resuming epoch {epoch} from training step {val_step}\n')

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

                val_loss += loss.item()
                val_images_seen += len(images)
                timer.report(f'eval batch {val_step} metrics update')

                ## Checkpointing
                print(f"Saving checkpoint at epoch {epoch} val batch {val_step}")
                val_sampler.advance(len(images))
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

                        # Evaluation metrics
                        "train_images_seen": train_images_seen,
                        "val_images_seen": val_images_seen,
                        "epoch_loss": epoch_loss,
                        "val_loss": val_loss,
                    }
                    timer = atomic_torch_save(checkpoint, args.resume, timer)

        val_loss /= val_images_seen
        print(f"Epoch {epoch} diff val loss: {val_loss:.4f}")

        return timer
