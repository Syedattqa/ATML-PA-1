# WORKFLOW

This project implements and compares CNN vs VIT, VAEs vs GAN and CLIP.
The comparison includes training, evaluation, and analysis of both models across multiple dimensions.



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

and similarly the rest .py files, for .ipynb, you can run it directly, in Jupyter Notebook.

This will execute the following steps:

1. **Dataset Setup**: Download and prepare CIFAR-10 dataset
2. **Model Training**: Train both VAE and DCGAN models (100 epochs each)
3. **Reconstruction vs Generation**: Compare model outputs side-by-side
4. **Latent Space Analysis**: Perform interpolation and semantic analysis
5. **OOD Testing**: Test robustness with out-of-distribution data
6. **Evaluation**: Calculate metrics including FID scores

