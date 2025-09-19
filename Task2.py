import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
from torchvision.utils import save_image
import warnings
warnings.filterwarnings('ignore')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ----------------------------------------------------------------------------
# ----------------------- Dataset Setup (CIFAR-10) -------------------------
# ----------------------------------------------------------------------------

def setup_data():
    """Setup CIFAR-10 dataset and dataloaders"""
    print(f"Using device: {device}")
    print("Using CIFAR-10 default resolution: 32x32")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    batch_size = 128

    # CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")

    return train_loader, test_loader

# ----------------------------------------------------------------------------
# ------------------------- VAE Model Architecture --------------------------
# ----------------------------------------------------------------------------

class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEEncoder, self).__init__()
        #32x32 input
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)    # 16x16
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)  # 8x8
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1) # 4x4
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1) # 2x2

        # Final size: 512 * 2 * 2 = 2048
        self.fc_mu = nn.Linear(512 * 2 * 2, latent_dim)
        self.fc_var = nn.Linear(512 * 2 * 2, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)  # 16x16
        x = F.leaky_relu(self.conv2(x), 0.2)  # 8x8
        x = F.leaky_relu(self.conv3(x), 0.2)  # 4x4
        x = F.leaky_relu(self.conv4(x), 0.2)  # 2x2

        x = x.view(x.size(0), -1)  # Flatten to 2048
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAEDecoder, self).__init__()
        # 32x32 output 
        self.fc = nn.Linear(latent_dim, 512 * 2 * 2)

        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)  # 4x4
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8x8
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16x16
        self.deconv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)     # 32x32

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, 2, 2)  # Reshape to 2x2

        x = F.leaky_relu(self.deconv1(x), 0.2)  # 4x4
        x = F.leaky_relu(self.deconv2(x), 0.2)  # 8x8
        x = F.leaky_relu(self.deconv3(x), 0.2)  # 16x16
        x = torch.tanh(self.deconv4(x))         # 32x32

        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return mu, log_var

    def decode(self, z):
        return self.decoder(z)

# ----------------------------------------------------------------------------
# ------------------------ DCGAN Model Architecture ------------------------
# ----------------------------------------------------------------------------

class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super(DCGANGenerator, self).__init__()
        
        self.main = nn.Sequential(
            # latent_dim -> 1024 x 4 x 4 (doubled capacity)
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # 1024 x 4 x 4 -> 512 x 8 x 8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 8 x 8 -> 256 x 16 x 16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 16 x 16 -> 3 x 32 x 32
            nn.ConvTranspose2d(256, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        
        if input.dim() == 2:
            input = input.unsqueeze(2).unsqueeze(3)  # (batch_size, latent_dim, 1, 1)
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super(DCGANDiscriminator, self).__init__()
        # 32x32 input 
        self.main = nn.Sequential(
            # 3 x 32 x 32 -> 64 x 16 x 16
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16 -> 128 x 8 x 8
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8 -> 256 x 4 x 4
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Final convolutional layer 
        self.final_conv = nn.Conv2d(256, 1, 4, 1, 0, bias=False)

    def forward(self, input):
        features = self.main(input)
        output = self.final_conv(features)
        return output.view(output.size(0), -1).squeeze(1)

# ----------------------------------------------------------------------------
# ----------------------- Loss Functions and Training -----------------------
# ----------------------------------------------------------------------------

def vae_loss_function(recon_x, x, mu, log_var, beta=0.01):
    # Reconstruction loss 
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae(vae, train_loader, num_epochs=100, lr=1e-3, beta=1.0):
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    train_losses = []

    print("1. Training VAE...")
    for epoch in range(num_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var = vae(data)
            loss, recon_loss, kl_loss = vae_loss_function(recon_batch, data, mu, log_var, beta)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon_loss = total_recon_loss / len(train_loader.dataset)
        avg_kl_loss = total_kl_loss / len(train_loader.dataset)

        train_losses.append(avg_loss)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch}: Total Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}')

    return train_losses

def train_gan(generator, discriminator, train_loader, num_epochs=100, lr_g=2e-4, lr_d=1e-4, latent_dim=128):
    generator.to(device)
    discriminator.to(device)

    # Weight initialization - Xavier
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Hinge loss functions
    def d_hinge_loss(real_pred, fake_pred):
        return torch.mean(F.relu(1. - real_pred)) + torch.mean(F.relu(1. + fake_pred))

    def g_hinge_loss(fake_pred):
        return -torch.mean(fake_pred)

    # Optimizers 
    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Training
    generator.train()
    discriminator.train()

    g_losses = []
    d_losses = []
    d_accuracies = []

    print("Training GAN with Hinge Loss")
    for epoch in range(num_epochs):
        total_d_loss = 0
        total_g_loss = 0
        total_d_real_acc = 0
        total_d_fake_acc = 0
        num_batches = 0

        for batch_idx, (real_data, _) in enumerate(train_loader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # Train Discriminator every step
            optimizer_D.zero_grad()

            # Real data
            real_output = discriminator(real_data)

            # Fake data 
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(noise)
            fake_output = discriminator(fake_data.detach())

            # Calculate discriminator accuracy
            real_acc = (real_output > 0).float().mean().item()  # Should be > 0 for real
            fake_acc = (fake_output < 0).float().mean().item()  # Should be < 0 for fake

            total_d_real_acc += real_acc
            total_d_fake_acc += fake_acc

            # Hinge loss for discriminator
            d_loss = d_hinge_loss(real_output, fake_output)
            d_loss.backward()
            optimizer_D.step()

            # Train Generator twice
            for _ in range(2):  
                optimizer_G.zero_grad()

                # Generate fake data
                noise = torch.randn(batch_size, latent_dim, device=device)
                fake_data = generator(noise)
                fake_output = discriminator(fake_data)

                # Hinge loss for generator
                g_loss = g_hinge_loss(fake_output)
                g_loss.backward()
                optimizer_G.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1

        avg_d_loss = total_d_loss / len(train_loader)
        avg_g_loss = total_g_loss / len(train_loader)
        avg_d_real_acc = total_d_real_acc / num_batches
        avg_d_fake_acc = total_d_fake_acc / num_batches
        avg_d_acc = (avg_d_real_acc + avg_d_fake_acc) / 2  # Overall discriminator accuracy

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)
        d_accuracies.append(avg_d_acc)

        if epoch % 20 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch}: D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}, D Acc: {avg_d_acc:.3f}')

    return g_losses, d_losses, d_accuracies

# ----------------------------------------------------------------------------
# ---------------------- 2. Reconstruction vs Generation -------------------
# ----------------------------------------------------------------------------

def compare_reconstruction_vs_generation(vae, generator, test_loader, num_samples=10):
    print("2. Reconstruction vs Generation Comparison...")

    vae.eval()
    generator.eval()

 
    os.makedirs('outputs', exist_ok=True)

    with torch.no_grad():
        # Get test images
        test_images, test_labels = next(iter(test_loader))
        test_images = test_images[:num_samples].to(device)

        # VAE reconstruction
        vae_recon, _, _ = vae(test_images)

        # GAN generation with standard normal noise
        torch.manual_seed(42) 
        noise = torch.randn(num_samples, 128, device=device)  # Standard normal noise
        gan_samples = generator(noise)

        # Denormalize for visualization
        def denormalize(tensor):
            return (tensor + 1) / 2

      
        fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))

        # Convert tensors for matplotlib
        test_imgs_plot = denormalize(test_images.cpu()).permute(0, 2, 3, 1).numpy()
        vae_imgs_plot = denormalize(vae_recon.cpu()).permute(0, 2, 3, 1).numpy()
        gan_imgs_plot = denormalize(gan_samples.cpu()).permute(0, 2, 3, 1).numpy()

        for i in range(num_samples):
            # Original images (top row)
            axes[0, i].imshow(np.clip(test_imgs_plot[i], 0, 1))
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original Images', fontsize=12, pad=20)

            # VAE reconstructions (middle row)
            axes[1, i].imshow(np.clip(vae_imgs_plot[i], 0, 1))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('VAE Reconstructions', fontsize=12, pad=20)

            # GAN generations (bottom row)
            axes[2, i].imshow(np.clip(gan_imgs_plot[i], 0, 1))
            axes[2, i].axis('off')
            if i == 0:
                axes[2, i].set_title('GAN Generated', fontsize=12, pad=20)

        plt.tight_layout()
        plt.savefig('outputs/reconstruction_vs_generation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("   Saved reconstruction vs generation comparison to outputs/reconstruction_vs_generation.png")

        # Calculate reconstruction error for VAE
        recon_error = F.mse_loss(vae_recon, test_images, reduction='mean')
        print(f"   VAE Average Reconstruction Error (MSE): {recon_error.item():.6f}")

    return test_images, vae_recon, gan_samples

# ----------------------------------------------------------------------------
# ------------------- 3. Latent Space Structure Analysis -------------------
# ----------------------------------------------------------------------------

def latent_space_interpolation(vae, generator, test_loader, num_interpolations=10):
    print("3. Latent Space Structure - Interpolation Analysis...")


    vae.eval()
    generator.eval()

    with torch.no_grad():
        # Get two different test images from different classes
        all_images = []
        all_labels = []
        for images, labels in test_loader:
            all_images.append(images)
            all_labels.append(labels)
            if len(all_images) >= 5:  
                break

        all_images = torch.cat(all_images)
        all_labels = torch.cat(all_labels)

        # Find images from different classes
        unique_labels = torch.unique(all_labels)
        if len(unique_labels) >= 2:
            class1_idx = (all_labels == unique_labels[0]).nonzero()[0].item()
            class2_idx = (all_labels == unique_labels[1]).nonzero()[0].item()
        else:
            class1_idx, class2_idx = 0, 1

        img1 = all_images[class1_idx:class1_idx+1].to(device)
        img2 = all_images[class2_idx:class2_idx+1].to(device)

        print(f"   Interpolating between class {all_labels[class1_idx].item()} and class {all_labels[class2_idx].item()}")

        # VAE interpolation using actual image latents
        with torch.no_grad():
            mu1, _ = vae.encode(img1)
            mu2, _ = vae.encode(img2)

        interpolation_factors = torch.linspace(0, 1, num_interpolations, device=device)
        vae_interpolations = []

        for alpha in interpolation_factors:
            z_interp = (1 - alpha) * mu1 + alpha * mu2
            recon_interp = vae.decode(z_interp)
            vae_interpolations.append(recon_interp)

        vae_interpolations = torch.cat(vae_interpolations, dim=0)

        # Improved GAN interpolation - use spherical interpolation (SLERP)
        def slerp(val, low, high):
            """Spherical interpolation between two points on a hypersphere"""
            low_norm = low / torch.norm(low, dim=-1, keepdim=True)
            high_norm = high / torch.norm(high, dim=-1, keepdim=True)
            omega = torch.acos(torch.clamp(torch.sum(low_norm * high_norm, dim=-1, keepdim=True), -1, 1))
            so = torch.sin(omega)
            if (so == 0).any():
                # Fall back to linear interpolation
                return (1.0 - val) * low + val * high
            else:
                return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

        # Generate starting and ending points for GAN interpolation
        torch.manual_seed(42)  # For reproducible interpolation
        z1 = torch.randn(1, 128, device=device)  # Standard normal noise
        z2 = torch.randn(1, 128, device=device)

        gan_interpolations = []
        for i, alpha in enumerate(interpolation_factors):
            # Use spherical interpolation
            z_interp = slerp(alpha, z1, z2)
            gen_interp = generator(z_interp)
            gan_interpolations.append(gen_interp)

        gan_interpolations = torch.cat(gan_interpolations, dim=0)

        # Save high-resolution interpolations
        def denormalize(tensor):
            return torch.clamp((tensor + 1) / 2, 0, 1)

        vae_grid = denormalize(vae_interpolations.cpu())
        gan_grid = denormalize(gan_interpolations.cpu())

        
        save_image(vae_grid, 'outputs/vae_interpolation.png', nrow=num_interpolations)
        save_image(gan_grid, 'outputs/gan_interpolation.png', nrow=num_interpolations)


def latent_representation_analysis(vae, test_loader, num_samples=1000):
    print("3. Latent Space Structure - Representation Analysis...")

    vae.eval()

    # Collect balanced samples from each class
    class_samples = {i: [] for i in range(10)}  # CIFAR-10 has 10 classes
    class_latents = {i: [] for i in range(10)}
    samples_per_class = num_samples // 10  # 100 samples per class

    with torch.no_grad():
        for images, batch_labels in test_loader:
            images = images.to(device)
            mu, _ = vae.encode(images)

            # Collect samples for each class
            for i, (img, label) in enumerate(zip(mu.cpu().numpy(), batch_labels.numpy())):
                if len(class_latents[label]) < samples_per_class:
                    class_latents[label].append(img)

            
            if all(len(class_latents[i]) >= samples_per_class for i in range(10)):
                break

    # Create balanced dataset
    latent_vectors = []
    labels = []
    for class_id in range(10):
        if len(class_latents[class_id]) > 0:
            # Take up to samples_per_class from each class
            class_data = class_latents[class_id][:samples_per_class]
            latent_vectors.extend(class_data)
            labels.extend([class_id] * len(class_data))

    latent_vectors = np.array(latent_vectors)
    labels = np.array(labels)

    # PCA analysis
    pca = PCA(n_components=2)
    latent_2d_pca = pca.fit_transform(latent_vectors)

    # t-SNE analysis with better parameters for clustering
    try:
     
        perplexity = min(50, len(latent_vectors) // 4)  # Higher perplexity for better global structure
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                   learning_rate=200, n_iter=1000, early_exaggeration=12)
        latent_2d_tsne = tsne.fit_transform(latent_vectors)
    except Exception as e:
        print(f"   t-SNE failed: {e}, using PCA for second plot")
        pca2 = PCA(n_components=2)
        latent_2d_tsne = pca2.fit_transform(latent_vectors)

    # Plot PCA and t-SNE
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i in range(10):
        mask = labels == i
        if np.sum(mask) > 0:  # Only plot if we have samples for this class
            plt.scatter(latent_2d_pca[mask, 0], latent_2d_pca[mask, 1],
                       c=[colors[i]], label=cifar10_classes[i], alpha=0.7, s=30)
    plt.title('VAE Latent Space (PCA)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    # Plot t-SNE
    plt.subplot(1, 2, 2)
    for i in range(10):
        mask = labels == i
        if np.sum(mask) > 0:  # Only plot if we have samples for this class
            plt.scatter(latent_2d_tsne[mask, 0], latent_2d_tsne[mask, 1],
                       c=[colors[i]], label=cifar10_classes[i], alpha=0.7, s=30)
    plt.title('VAE Latent Space (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/latent_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   PCA explained variance ratio: {pca.explained_variance_ratio_}")


    print("   Performing enhanced semantic analysis")

    # Calculate variance across all latent dimensions to find most meaningful ones
    latent_variance = np.var(latent_vectors, axis=0)
    top_varying_dims = np.argsort(latent_variance)[-10:][::-1] 
    most_varying_dims = top_varying_dims[:5]  # Use top 5 most varying dimensions

    fig, axes = plt.subplots(len(most_varying_dims), 9, figsize=(20, len(most_varying_dims) * 2.5))
    if len(most_varying_dims) == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        class_samples = []
        class_labels = []

        # Collect one sample from each class if possible
        for images, labels in test_loader:
            for class_id in range(10):  # CIFAR-10 has 10 classes
                class_mask = labels == class_id
                if class_mask.any():
                    class_idx = class_mask.nonzero()[0].item()
                    if class_id not in class_labels:
                        class_samples.append(images[class_idx:class_idx+1])
                        class_labels.append(class_id)
                if len(class_samples) >= 3:  # Use 3 different classes as bases
                    break
            if len(class_samples) >= 3:
                break

        # Get base latent vectors from different classes
        base_latents = []
        for sample in class_samples:
            sample = sample.to(device)
            mu, _ = vae.encode(sample)
            base_latents.append(mu)


        dim_values = torch.linspace(-4, 4, 9)

        # Test each of the most varying dimensions
        for row_idx, dim_idx in enumerate(most_varying_dims):
            variations = []

            # Use the first class sample as base 
            base_z = base_latents[0].clone()  # Use single class, not average

            for val in dim_values:
                z_var = base_z.clone()
                z_var[0, dim_idx] = val
                decoded = vae.decode(z_var)
                variations.append(decoded)

            variations = torch.cat(variations, dim=0)
            variations_grid = torch.clamp((variations + 1) / 2, 0, 1).cpu()  # Denormalize and clamp

            # Display the variations for this dimension
            for i, img in enumerate(variations_grid):
                img_np = img.permute(1, 2, 0).numpy()
                axes[row_idx, i].imshow(img_np)

                if row_idx == 0:  # Only add value labels on top row
                    axes[row_idx, i].set_title(f'{dim_values[i]:.1f}', fontsize=10)

                axes[row_idx, i].axis('off')

            # Add dimension label with variance information
            dim_var = latent_variance[dim_idx]
            axes[row_idx, 0].text(-0.15, 0.5, f'Dim {dim_idx}\n(σ²={dim_var:.3f})',
                                 transform=axes[row_idx, 0].transAxes,
                                 rotation=0, verticalalignment='center', fontsize=10, fontweight='bold')

    plt.suptitle('VAE Semantic Factor Analysis - Most Varying Latent Dimensions', fontsize=16)
    plt.tight_layout()
    plt.savefig('outputs/semantic_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()


    print(f"   Analyzed dimensions: {most_varying_dims} with highest variance")


    with torch.no_grad():
        # Sample multiple images and their latent representations
        all_images = []
        all_latents = []
        for images, _ in test_loader:
            if len(all_images) >= 500:  # Limit for efficiency
                break
            images = images.to(device)
            mu, _ = vae.encode(images)
            all_images.append(images)
            all_latents.append(mu)

        all_latents = torch.cat(all_latents)
        latent_vars = torch.var(all_latents, dim=0)
        top_varying_dims = torch.argsort(latent_vars, descending=True)[:5]

        print(f"   Top 5 most varying latent dimensions: {top_varying_dims.cpu().numpy()}")
        print(f"   Their variances: {latent_vars[top_varying_dims].cpu().numpy()}")

# ----------------------------------------------------------------------------
# ------------------- 4. Out-of-Distribution (OOD) Inputs ------------------
# ----------------------------------------------------------------------------

def ood_analysis(vae, generator, test_loader):
    print("4. Out-of-Distribution (OOD) Analysis")

    vae.eval()
    generator.eval()

    # Load CIFAR-100
    try:
        ood_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        ood_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=ood_transform)
        ood_loader = DataLoader(ood_dataset, batch_size=64, shuffle=True)
        print("   Using CIFAR-100 as OOD dataset")
    except:
        print("   Could not load CIFAR-100, using synthetic OOD data")
        ood_loader = None

    with torch.no_grad():
        # Get normal CIFAR-10 test images
        normal_images, _ = next(iter(test_loader))
        normal_images = normal_images[:32].to(device)

        # Get OOD images
        if ood_loader is not None:
            ood_images, _ = next(iter(ood_loader))
            ood_images = ood_images[:32].to(device)
        else:
            # Create synthetic OOD data (random noise, geometric shapes)
            ood_images = torch.randn_like(normal_images)

        # VAE reconstruction of normal images
        normal_recon, normal_mu, normal_logvar = vae(normal_images)
        normal_recon_error = F.mse_loss(normal_recon, normal_images, reduction='none').mean(dim=[1,2,3])

        # VAE reconstruction of OOD images
        ood_recon, ood_mu, ood_logvar = vae(ood_images)
        ood_recon_error = F.mse_loss(ood_recon, ood_images, reduction='none').mean(dim=[1,2,3])

        print(f"   Normal images - Average reconstruction error: {normal_recon_error.mean().item():.6f}")
        print(f"   OOD images - Average reconstruction error: {ood_recon_error.mean().item():.6f}")

        # Systematic GAN latent extrapolation stress testing
        torch.manual_seed(42)  

        # Test multiple levels of latent extremeness
        latent_scales = [1.0, 2.0, 3.0, 4.0, 6.0]  # 1.0 = normal, 6.0 = very extreme
        scale_results = {}

        print("   Testing GAN latent extrapolation with extreme values:")
        scale_images = {}  # Store generated images for visualization

        for scale in latent_scales:
            noise = torch.randn(8, 128, device=device) * scale
            generated = generator(noise)

            # Store for visualization
            scale_images[scale] = generated

            # Measure image quality degradation

            has_nan = torch.isnan(generated).any()
            has_inf = torch.isinf(generated).any()

            # Measure pixel variance (low variance = mode collapse/artifacts)
            pixel_var = generated.var().item()

            scale_results[scale] = {
                'has_nan': has_nan.item(),
                'has_inf': has_inf.item(),
                'pixel_variance': pixel_var
            }

            status = "DEGRADED" if (has_nan or has_inf or pixel_var < 0.01) else "STABLE"
            print(f"     Scale {scale}x: {status} (var={pixel_var:.4f})")

        # Create GAN stress testing visualization
        def denormalize(tensor):
            return torch.clamp((tensor + 1) / 2, 0, 1)

        fig_stress, axes_stress = plt.subplots(len(latent_scales), 8, figsize=(24, len(latent_scales) * 3))

        for row_idx, scale in enumerate(latent_scales):
            images = denormalize(scale_images[scale].cpu()).permute(0, 2, 3, 1).numpy()
            status = scale_results[scale]

            # Determine row color based on quality
            if status['has_nan'] or status['has_inf'] or status['pixel_variance'] < 0.01:
                row_color = 'red'
                quality_label = 'DEGRADED'
            else:
                row_color = 'green'
                quality_label = 'STABLE'

            for col_idx in range(8):
                axes_stress[row_idx, col_idx].imshow(images[col_idx])
                axes_stress[row_idx, col_idx].axis('off')

                # Add scale label on first image of each row
                if col_idx == 0:
                    axes_stress[row_idx, col_idx].text(-0.1, 0.5,
                                                     f'{scale}x\n{quality_label}\n(σ²={status["pixel_variance"]:.3f})',
                                                     transform=axes_stress[row_idx, col_idx].transAxes,
                                                     rotation=0, verticalalignment='center', fontsize=10,
                                                     fontweight='bold', color=row_color,
                                                     bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        plt.suptitle('GAN Latent Extrapolation Stress Test: Quality vs. Latent Magnitude', fontsize=16)
        plt.tight_layout()
        plt.savefig('outputs/gan_stress_test.png', dpi=150, bbox_inches='tight')
        plt.close()

        # Use standard scales for visualization
        normal_noise = torch.randn(16, 128, device=device)  # 1.0x standard
        extreme_noise = torch.randn(16, 128, device=device) * 4.0  # 4.0x extreme

        normal_gen = generator(normal_noise)
        extreme_gen = generator(extreme_noise)


        def denormalize(tensor):
            return torch.clamp((tensor + 1) / 2, 0, 1)


        fig, axes = plt.subplots(6, 8, figsize=(20, 15))

        # Convert tensors for plotting
        normal_imgs = denormalize(normal_images[:8].cpu()).permute(0, 2, 3, 1).numpy()
        normal_recon_imgs = denormalize(normal_recon[:8].cpu()).permute(0, 2, 3, 1).numpy()
        ood_imgs = denormalize(ood_images[:8].cpu()).permute(0, 2, 3, 1).numpy()
        ood_recon_imgs = denormalize(ood_recon[:8].cpu()).permute(0, 2, 3, 1).numpy()
        normal_gen_imgs = denormalize(normal_gen.cpu()).permute(0, 2, 3, 1).numpy()
        extreme_gen_imgs = denormalize(extreme_gen.cpu()).permute(0, 2, 3, 1).numpy()

        row_labels = [
            'CIFAR-10 Original',
            'VAE Recon (Normal)',
            'CIFAR-100/OOD Original',
            'VAE Recon (OOD)',
            'GAN Normal Latent',
            'GAN Extreme Latent'
        ]

        all_images = [normal_imgs, normal_recon_imgs, ood_imgs, ood_recon_imgs, normal_gen_imgs, extreme_gen_imgs]

        for row_idx, (images, label) in enumerate(zip(all_images, row_labels)):
            for col_idx in range(8):
                if col_idx < len(images):
                    axes[row_idx, col_idx].imshow(np.clip(images[col_idx], 0, 1))
                axes[row_idx, col_idx].axis('off')

                # Add row labels
                if col_idx == 0:
                    axes[row_idx, col_idx].text(-0.1, 0.5, label, transform=axes[row_idx, col_idx].transAxes,
                                               rotation=90, verticalalignment='center', fontsize=11, fontweight='bold')

        plt.suptitle('Out-of-Distribution Analysis: Normal vs OOD Reconstruction', fontsize=16)
        plt.tight_layout()
        plt.savefig('outputs/ood_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
   

        # Anomaly detection using reconstruction error
        threshold = normal_recon_error.mean() + 2 * normal_recon_error.std()
        normal_anomalies = (normal_recon_error > threshold).sum().item()
        ood_anomalies = (ood_recon_error > threshold).sum().item()

        print(f"   Anomaly detection threshold: {threshold.item():.6f}")
        print(f"   Normal images flagged as anomalies: {normal_anomalies}/{len(normal_recon_error)}")
        print(f"   OOD images flagged as anomalies: {ood_anomalies}/{len(ood_recon_error)}")

        # GAN latent extrapolation summary
        stable_scales = [s for s, r in scale_results.items() if not (r['has_nan'] or r['has_inf'] or r['pixel_variance'] < 0.01)]
        print(f"   GAN latent extrapolation: Stable up to {max(stable_scales) if stable_scales else 0}x normal range")
        print(f"   GAN robustness: {'HIGH' if len(stable_scales) >= 4 else 'MEDIUM' if len(stable_scales) >= 2 else 'LOW'}")

# ----------------------------------------------------------------------------
# ---------------------- 5. Evaluation Metrics -----------------------------
# ----------------------------------------------------------------------------

def calculate_fid_score(real_images, generated_images, batch_size=50):
    """
    Simplified FID calculation using feature statistics
    """
    try:
        from torchvision.models import inception_v3
        from scipy.linalg import sqrtm

        # Load pre-trained Inception model
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = nn.Identity()  # Remove final layer
        inception.eval()
        inception.to(device)

        def get_features(images):
            features = []
            # Ensure images are in [0,1] range for Inception
            images = torch.clamp(images, 0, 1)

            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                # Resize to 299x299 for Inception and normalize
                batch_resized = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                # Inception expects normalized inputs
                batch_norm = (batch_resized - 0.5) / 0.5

                with torch.no_grad():
                    feat = inception(batch_norm)
                features.append(feat.cpu().numpy())
            return np.concatenate(features)

        # Get features
        real_features = get_features(real_images)
        gen_features = get_features(generated_images)

        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)

        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma_real += eps * np.eye(sigma_real.shape[0])
        sigma_gen += eps * np.eye(sigma_gen.shape[0])

        # Calculate FID using proper matrix square root
        diff = mu_real - mu_gen

        # Calculate matrix square root
        covmean = sqrtm(sigma_real.dot(sigma_gen))

        # Handle complex numbers
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.diagonal(covmean).imag)
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

        return max(0, fid)  # Ensure non-negative

    except Exception as e:
        print(f"   FID calculation failed: {e}")
        print("   Skipping FID computation")
        return None

def evaluate_models(vae, generator, test_loader, num_samples=1000):
    print("5. Model Evaluation...")

    vae.eval()
    generator.eval()

    with torch.no_grad():
        # Collect real images
        real_images = []
        sample_count = 0

        for images, _ in test_loader:
            if sample_count >= num_samples:
                break
            real_images.append(images)
            sample_count += images.size(0)

        real_images = torch.cat(real_images)[:num_samples].to(device)

        # Generate VAE reconstructions
        vae_recons, _, _ = vae(real_images)

        # Generate GAN samples with standard noise
        torch.manual_seed(42)  
        noise = torch.randn(num_samples, 128, device=device)  # Standard normal noise
        gan_samples = generator(noise)

        # Calculate reconstruction error for VAE
        recon_error = F.mse_loss(vae_recons, real_images)
        print(f"   VAE Reconstruction Error (MSE): {recon_error.item():.6f}")

        
        def normalize_for_fid(images):
            return (images + 1) / 2  # Convert from [-1,1] to [0,1]

        real_norm = normalize_for_fid(real_images)
        vae_norm = normalize_for_fid(vae_recons)
        gan_norm = normalize_for_fid(gan_samples)

        # Calculate FID scores
        vae_fid = calculate_fid_score(real_norm, vae_norm)
        gan_fid = calculate_fid_score(real_norm, gan_norm)

        if vae_fid is not None:
            print(f"   VAE FID Score: {vae_fid:.2f}")
        if gan_fid is not None:
            print(f"   GAN FID Score: {gan_fid:.2f}")

# ----------------------------------------------------------------------------
# ----------------------------- Main Execution ------------------------------
# ----------------------------------------------------------------------------

def main():
    print("Starting VAE vs GAN Comparison on CIFAR-10")
    print("=" * 50)

    # Initialize parameters
    latent_dim = 128

    train_loader, test_loader = setup_data()

    # Initialize models
    vae = VAE(latent_dim=latent_dim)
    generator = DCGANGenerator(latent_dim=latent_dim)
    discriminator = DCGANDiscriminator()

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # 1. Train models
    print("\n" + "="*50)
    print("STEP 1: TRAINING MODELS")
    print("="*50)

    # Train VAE 
    vae_losses = train_vae(vae, train_loader, num_epochs=100, lr=2e-4, beta=0.01)

    # Train GAN 
    g_losses, d_losses, d_accuracies = train_gan(generator, discriminator, train_loader, num_epochs=100, lr_g=3e-4, lr_d=8e-5, latent_dim=latent_dim)

    # Plot training curves with discriminator accuracy
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(vae_losses)
    plt.title('VAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 2)
    plt.plot(g_losses, label='Generator', color='blue')
    plt.plot(d_losses, label='Discriminator', color='red')
    plt.title('GAN Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 3)
    plt.plot(d_accuracies, label='D Accuracy', color='green', linewidth=2)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Random (0.5)')
    plt.title('Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(vae_losses, label='VAE', color='purple')
    plt.plot(g_losses, label='GAN Generator', color='blue')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.close()
    print("   Saved training curves to outputs/training_curves.png")

    # 2. Reconstruction vs Generation
    print("\n" + "="*50)
    print("STEP 2: RECONSTRUCTION VS GENERATION")
    print("="*50)
    test_images, vae_recon, gan_samples = compare_reconstruction_vs_generation(vae, generator, test_loader)

    # 3. Latent Space Structure
    print("\n" + "="*50)
    print("STEP 3: LATENT SPACE STRUCTURE")
    print("="*50)
    latent_space_interpolation(vae, generator, test_loader)
    latent_representation_analysis(vae, test_loader)

    # 4. OOD Analysis
    print("\n" + "="*50)
    print("STEP 4: OUT-OF-DISTRIBUTION ANALYSIS")
    print("="*50)
    ood_analysis(vae, generator, test_loader)

    # 5. Evaluation
    print("\n" + "="*50)
    print("STEP 5: MODEL EVALUATION")
    print("="*50)
    evaluate_models(vae, generator, test_loader)

    # Save models
    torch.save(vae.state_dict(), 'outputs/vae_model.pth')
    torch.save(generator.state_dict(), 'outputs/gan_generator.pth')
    torch.save(discriminator.state_dict(), 'outputs/gan_discriminator.pth')

if __name__ == "__main__":
    main()