from tqdm import tqdm
import torch
import os
def train(disc, gen, BCE, disc_opt, gen_opt, l1_loss,epoch,loader, Config , make_grid, wandb, writer_real, writer_fake, step):
    loop  = tqdm(enumerate(loader), leave = False, desc="{}/{}".format(epoch,Config.EPOCHS))
    for batch_idx , (input_image,target_image) in loop:
        input_image,target_image= input_image.to(Config.DEVICE) , target_image.to(Config.DEVICE)
        # Train disc
        fake_image = gen(input_image)
        
        disc_real = disc(input_image, target_image)
        disc_fake = disc(input_image, fake_image)

        disc_real_loss = BCE(disc_real, torch.ones_like(disc_real))
        disc_fake_loss = BCE(disc_fake, torch.zeros_like(disc_fake))

        disc_loss = (disc_real_loss+disc_fake_loss)/2

        disc.zero_grad()
        disc_loss.backward()
        disc_opt.step()

    # training Generator
        fake_image      = gen(input_image) # WOuld make more fast if we remove this but want to make it more expressive
        disc_fake       = disc(input_image, fake_image)
        gen_fake_loss   = BCE(disc_fake, torch.ones_like(disc_fake))
        L1              = l1_loss(fake_image, target_image) * Config.L1_LAMBDA

        gen_loss        = gen_fake_loss+ L1
        
        gen.zero_grad()
        gen_loss.backward()
        gen_opt.step()
        
        loop.set_postfix(
            D_real=torch.sigmoid(disc_real).mean().item(),
            D_fake=torch.sigmoid(disc_fake).mean().item(),
            disc_loss = disc_loss.item(),
            gen_loss  = gen_loss.item()
        )
        if batch_idx % 2 == 0:
            fake_image = fake_image * 0.5 + 0.5 
            input_image = input_image * 0.5 + 0.5 

            img_grid_real = make_grid(input_image[:8], normalize=True)
            img_grid_fake = make_grid(fake_image[:8], normalize=True)
            step +=1
            writer_real.add_image("Real", img_grid_real, global_step=step)
            writer_fake.add_image("Fake", img_grid_fake, global_step=step)
            wandb.log({"Discremenator Loss": disc_loss.item(), "Generator Loss": gen_loss.item()})
            wandb.log({"img": [wandb.Image(img_grid_fake, caption=step)]})
            torch.save(gen.state_dict(), os.path.join(wandb.run.dir, 'dc_gan_model_gen.pt'))
            torch.save(disc.state_dict(), os.path.join(wandb.run.dir, 'dc_gan_model_disc.pt'))
    return step


    