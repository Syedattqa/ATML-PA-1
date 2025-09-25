# VAE vs GAN Comparison on CIFAR-10

This project implements and compares 
Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) on the CIFAR-10 dataset.
The comparison includes training, evaluation, and analysis of both models across multiple dimensions.

## Project Overview

This project performs a comprehensive comparison between:
- **VAE (Variational Autoencoder)**: Focuses on reconstruction and latent space structure
- **DCGAN (Deep Convolutional GAN)**: Focuses on generation quality and adversarial training

The analysis includes:
1. Model training with loss tracking
2. Reconstruction vs Generation comparison
3. Latent space structure analysis with interpolation
4. Out-of-distribution (OOD) robustness testing
5. Comprehensive evaluation metrics including FID scores


## Installation

### Prerequisites

- Python 3.8 or higher

### Dependencies

Install the required libraries using pip:

```bash
pip install -r requirements.txt
```


To run the full VAE vs GAN comparison:

```bash
python Task2.py
```

This will execute the following steps:

1. **Dataset Setup**: Download and prepare CIFAR-10 dataset
2. **Model Training**: Train both VAE and DCGAN models (100 epochs each)
3. **Reconstruction vs Generation**: Compare model outputs side-by-side
4. **Latent Space Analysis**: Perform interpolation and semantic analysis
5. **OOD Testing**: Test robustness with out-of-distribution data
6. **Evaluation**: Calculate metrics including FID scores


### Output

The script generates several outputs in the `outputs/` directory:

- `training_curves.png`: Training loss curves for both models
- `reconstruction_vs_generation.png`: Side-by-side comparison of reconstructions
- `vae_interpolation.png`: VAE latent space interpolation
- `gan_interpolation.png`: GAN latent space interpolation
- `latent_analysis.png`: PCA and t-SNE visualization of latent space
- `semantic_analysis.png`: Semantic factor analysis
- `ood_analysis.png`: Out-of-distribution robustness test
- `gan_stress_test.png`: GAN latent extrapolation stress test
- `vae_model.pth`: Trained VAE model weights
- `gan_generator.pth`: Trained GAN generator weights
- `gan_discriminator.pth`: Trained GAN discriminator weights


## Configuration

### Key Parameters

You can modify these parameters in the `main()` function:

```python
# Training parameters
num_epochs = 100
batch_size = 128
latent_dim = 128

# VAE parameters
vae_lr = 2e-4
beta = 0.01  # KL divergence weight

# GAN parameters
gan_lr_g = 3e-4  # Generator learning rate
gan_lr_d = 8e-5  # Discriminator learning rate
```