import torch
from tqdm.auto import tqdm
from torch.nn import BCEWithLogitsLoss
import matplotlib.pyplot as plt

def plot_losses(trainer):
    #plot losses
    plt.subplots(figsize=(10, 5))

    #Plot 2 plots
    plt.subplot(1, 2, 1)
    plt.plot(trainer.losses_gen, label='Generator')
    plt.title('Generator')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(trainer.losses, label='Discriminator')
    plt.legend()
    plt.title('Discriminator')
    plt.xlabel('Iter')
    plt.ylabel('Loss')

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size, 1, 1)*2 - 1 
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size, 1, 1)
    return input_z

def sample_from_latent(n_samples, z_size=100, mode_z='normal'):
    return create_noise(n_samples, z_size, mode_z)

def plot_images(images_dl):
    batch_imgs, _labels = next(iter(images_dl))
    plt.figure(figsize=(5,5))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(batch_imgs[i].view(28,28), cmap='binary', interpolation='none')
        plt.xticks([])
        plt.yticks([])

def plot_epochs_images(epoch_samples, selected_epochs = [1, 2, 4, 10, 15, 39]):
    fig = plt.figure(figsize=(10, 14))
    for i,e in enumerate(selected_epochs):
        for j in range(5):
            ax = fig.add_subplot(len(selected_epochs), 5, i*5+j+1)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.text(
                    -0.06, 0.5, f'Epoch {e}',
                    rotation=90, size=18, color='red',
                    horizontalalignment='right',
                    verticalalignment='center', 
                    transform=ax.transAxes)
            
            image = epoch_samples[e-1][j]
            ax.imshow(image, cmap='gray_r')
    # plt.savefig('figures/ch17-dcgan-samples.pdf')
    plt.show()

def save_epochs_images(epoch_samples):
    fig = plt.figure(figsize=(6, 1.5))
    for j in range(5):
        ax = fig.add_subplot(1, 5, j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.06, 0.5, f'Epoch {len(epoch_samples)}',
                rotation=90, size=18, color='red',
                horizontalalignment='right',
                verticalalignment='center', 
                transform=ax.transAxes)
        
        image = epoch_samples[-1][j]
        ax.imshow(image, cmap='gray_r')
    plt.savefig(f'figures/e_{len(epoch_samples)}.png')

def generate_and_show(generator, device):
    z_input = sample_from_latent(3).to(device)
    fake_imgs = generator(z_input)
    img0 = fake_imgs.detach().cpu().numpy()

    #Show images
    plt.figure(figsize=(5,5))
    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.tight_layout()
        plt.imshow(img0[i].reshape(28,28), cmap='binary', interpolation='none')
        plt.xticks([])
        plt.yticks([])


bce = BCEWithLogitsLoss()
def compute_loss(pred, target):
    return bce(pred, target)
      
 
class Trainner():
    def __init__(self, generator, discriminator, train_loader, device):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.losses = []
        self.losses_gen = []
        self.epoch_samples = []
        self.train_loader = train_loader

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        # self.optimizer_G = torch.optim.AdamW(self.generator.parameters(), lr=0.0003)
        # self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters(), lr=0.0002)
        
        self.optimizer_G = torch.optim.AdamW(self.generator.parameters())
        self.optimizer_D = torch.optim.AdamW(self.discriminator.parameters())

    def discriminator_step(self, img_batch, z_mode = 'uniform'):
        batch_size = img_batch.shape[0]
        img_batch = img_batch.to(self.device)
        ones = torch.ones((batch_size, 1)).to(self.device)
        zeros = torch.zeros((batch_size, 1)).to(self.device)
        z_input = sample_from_latent(batch_size, 100,  mode_z=z_mode).to(self.device)

        fake_imgs = self.generator(z_input)

        y_pred = self.discriminator(img_batch)
        y_pred_fake = self.discriminator(fake_imgs)

        loss = compute_loss(y_pred, ones)
        loss += compute_loss(y_pred_fake, zeros)

        self.optimizer_D.zero_grad()
        loss.backward()
        self.optimizer_D.step()        

        return loss.cpu().item()

    def generator_step(self, labels_batch, img_batch, z_mode = 'uniform'):
        self.optimizer_G.zero_grad()

        ones = torch.ones((labels_batch.shape[0], 1)).to(self.device)
        z_input = sample_from_latent(img_batch.shape[0], mode_z=z_mode).to(self.device)

        fake_imgs = self.generator(z_input)
        y_pred_fake = self.discriminator(fake_imgs)
        loss = compute_loss(y_pred_fake, ones)
        loss.backward()

        self.optimizer_G.step()
        return loss.cpu().item()    


    def create_samples(self, input_z):
        g_output = self.generator(input_z)
        images = torch.reshape(g_output, (input_z.size(0), 28,28))    
        return (images+1)/2.0


    def train_both(self, epochs = 1, train_gen_every = 1, train_dis_every = 1, z_mode = 'uniform'):
        progress_bar = tqdm(range(epochs*len(self.train_loader)))
        fixed_z = sample_from_latent(16, mode_z=z_mode).to(self.device)
        self.discriminator.train()
        for epoch in range(epochs):
            self.generator.train()
            for iter, (img_batch, labels_batch) in enumerate(self.train_loader):
                if iter % train_dis_every == 0:
                    d_loss = self.discriminator_step(img_batch, z_mode=z_mode)
                if iter % train_gen_every == 0:
                    g_loss = self.generator_step(labels_batch, img_batch, z_mode)

                if d_loss:
                    self.losses.append(d_loss)
                if g_loss:
                    self.losses_gen.append(g_loss)
                progress_bar.update(1)

            self.generator.eval()
            self.epoch_samples.append(self.create_samples(fixed_z).detach().cpu().numpy())
            save_epochs_images(self.epoch_samples)
            print("Epoch: {} - D loss: {} - G loss: {}".format(epoch, d_loss, g_loss))        
                