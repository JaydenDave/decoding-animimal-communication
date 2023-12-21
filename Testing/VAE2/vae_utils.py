import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from scipy.stats import norm


def reconstructions(vae,dataset,num_to_show):
    batches_to_predict = 1
    example_images = np.array(list(dataset.take(batches_to_predict).get_single_element()))
    z_mean, z_log_var, reconstructions = vae.predict(example_images)
    num_to_show = len(example_images) if num_to_show > len(example_images) else num_to_show
    fig, axs = plt.subplots(2,num_to_show)
    plt.subplots_adjust(wspace=0, hspace=0)
    
    for i in range(num_to_show):
        axs[0,i].imshow(example_images[i])
        axs[1,i].imshow(reconstructions[i])

        # Remove x and y ticks and labels for each subplot
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_yticklabels([])
        axs[1, i].set_xticklabels([])
        axs[1, i].set_yticklabels([])

    fig.text(0.5, 0.96, "Example Images", ha='center', va='center', fontsize=12)
    fig.text(0.5, 0.48, "Reconstructions", ha='center', va='center', fontsize=12)
    plt.show()

def check_dimension_distribution(vae,dataset,dimensions = 50):
    batches_to_predict = 1
    example_images = np.array(list(dataset.take(batches_to_predict).get_single_element()))
    _, _, z = vae.encoder.predict(example_images)

    x = np.linspace(-3, 3, 100)

    fig = plt.figure(figsize=(20, 5))
    fig.subplots_adjust(hspace=0.6, wspace=0.4)

    for i in range(dimensions):
        ax = fig.add_subplot(5, 10, i + 1)
        ax.hist(z[:, i], density=True, bins=20)
        ax.axis("off")
        ax.text(
            0.5, -0.35, str(i), fontsize=10, ha="center", transform=ax.transAxes
        )
        ax.plot(x, norm.pdf(x))

    plt.show()

def generate_random(vae,num_to_generate, LATENT_DIM):
    grid_width, grid_height = (10, 3)

    # Sample some points in the latent space, from the standard normal distribution
    z_sample = np.random.normal(size=(num_to_generate, LATENT_DIM))
    
    # Decode the sampled points
    reconstructions = vae.decoder.predict(z_sample)

    # Draw a plot of decoded images
    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Output the grid of faces
    for i in range(grid_width * grid_height):
        ax = fig.add_subplot(grid_height, grid_width, i + 1)
        ax.axis("off")
        ax.imshow(reconstructions[i, :, :])
    plt.show()