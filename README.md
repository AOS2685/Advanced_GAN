# Advanced Face Generation using WGAN-GP

This project implements a **Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP)** using PyTorch to generate synthetic human faces from random noise.

The model is trained on images from the CelebA face dataset and uses convolutional neural networks to learn the visual distribution of human faces.

## Project Overview

Traditional Generative Adversarial Networks consist of a Generator and a Discriminator competing against each other. However, basic GANs can suffer from unstable training, mode collapse, and loss values that do not clearly represent the quality of generated images.

This project explores an advanced GAN architecture using **Wasserstein GAN with Gradient Penalty (WGAN-GP)**.

Instead of using a Discriminator that simply classifies an image as real or fake, WGAN uses a **Critic** that assigns a realism score to an image.

The Generator learns to produce synthetic faces that receive increasingly better scores from the Critic.

## How It Works

The training process can be summarized as follows:

1. A 200-dimensional random noise vector is generated.
2. The Generator transforms the noise into a 128 × 128 RGB face image.
3. The Critic evaluates real CelebA images and generated images.
4. Real images are encouraged to receive higher Critic scores.
5. Generated images are encouraged to receive lower Critic scores during Critic training.
6. Gradient Penalty is applied to maintain smooth and stable Critic behaviour.
7. The Critic is trained multiple times before each Generator update.
8. The Generator learns to create images that receive higher scores from the Critic.
9. Through repeated adversarial training, the generated face distribution gradually moves closer to the real face distribution.

## Why WGAN-GP?

Basic GANs may suffer from several problems:

* Mode Collapse
* Unstable training
* Vanishing gradients
* Difficulty interpreting GAN loss
* Imbalance between the Generator and Discriminator

WGAN introduces the Wasserstein distance concept to provide a more meaningful learning signal.

Gradient Penalty further improves training stability by constraining the gradient magnitude of the Critic.

## Generator Architecture

The Generator transforms a 200-dimensional latent noise vector into a 128 × 128 RGB image.

Architecture:

200 × 1 × 1
↓
512 × 4 × 4
↓
256 × 8 × 8
↓
128 × 16 × 16
↓
64 × 32 × 32
↓
32 × 64 × 64
↓
3 × 128 × 128

The Generator uses:

* ConvTranspose2d
* Batch Normalization
* ReLU
* Tanh

ConvTranspose2d layers progressively increase the spatial dimensions of the latent representation until a complete RGB image is generated.

## Critic Architecture

The Critic receives a 128 × 128 RGB image and produces a single realism score.

Architecture:

3 × 128 × 128
↓
16 × 64 × 64
↓
32 × 32 × 32
↓
64 × 16 × 16
↓
128 × 8 × 8
↓
256 × 4 × 4
↓
1

The Critic uses:

* Conv2d
* Instance Normalization
* Leaky ReLU

The convolutional layers progressively extract visual features while reducing the image dimensions.

## Gradient Penalty

WGAN-GP uses Gradient Penalty to improve the stability of Critic training.

The model creates interpolated images between real and generated samples:

Mixed Image = α × Real Image + (1 - α) × Fake Image

The gradient of the Critic output with respect to these interpolated images is calculated.

A penalty is applied when the gradient norm moves away from 1.

This encourages the Critic to behave smoothly and satisfies the Lipschitz constraint required by the Wasserstein objective.

## Training Strategy

For every Generator update, the Critic is trained five times.

Training flow:

Critic → Critic → Critic → Critic → Critic → Generator

Training the Critic multiple times provides a stronger and more meaningful learning signal to the Generator.

### Critic Loss

The Critic loss is calculated using:

Critic Loss = Mean Fake Score - Mean Real Score + Gradient Penalty

The Critic learns to assign higher scores to real images and lower scores to generated images.

### Generator Loss

The Generator loss is calculated as:

Generator Loss = -Mean Critic Score for Generated Images

The Generator attempts to increase the Critic score of generated images, encouraging the creation of more realistic faces.

## Latent Space Interpolation

The project also demonstrates latent space interpolation.

Two random latent vectors are generated and intermediate points are calculated between them.

The Generator converts each intermediate latent vector into an image.

This creates a smooth visual transition from one generated face to another and demonstrates that the Generator has learned a continuous representation of facial features.

## Tech Stack

* Python
* PyTorch
* Torchvision
* CelebA Dataset
* NumPy
* Matplotlib
* Pillow
* Weights & Biases
* CUDA / GPU Training

## Key Concepts Demonstrated

* Generative Adversarial Networks
* Wasserstein GAN
* Gradient Penalty
* Convolutional Neural Networks
* Conv2d
* ConvTranspose2d
* Latent Space
* Latent Space Interpolation
* Adversarial Training
* Adam Optimizer
* Batch Normalization
* Instance Normalization
* GPU-based Deep Learning
* Model Checkpointing
* Experiment Tracking

## Simple Analogy

The Generator acts like an **artist learning to draw realistic human faces**, while the Critic acts like a **professional art reviewer who gives each image a realism score**.

The Critic is trained to provide meaningful feedback, while Gradient Penalty prevents the Critic from reacting too aggressively to small image changes.

As training continues, the Artist uses the Critic's feedback to generate increasingly realistic synthetic faces.

## Objective

The objective of this project is to understand advanced Generative Adversarial Network architectures and explore how Wasserstein Loss and Gradient Penalty can improve GAN training stability while generating high-resolution synthetic face images.
