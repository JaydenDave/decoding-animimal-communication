import tensorflow as tf
from tensorflow.keras import utils

#lego brick dataset
directory = r"C:\Users\Jayde\Desktop\lego-brick-data\dataset"
#also resizes the images to 64x64 (from 400x400), interpolating between pixel values
train_data = utils.image_dataset_from_directory(directory,
                                               labels=None,
                                               color_mode="grayscale",
                                               image_size=(64,64),
                                               batch_size=128,
                                               shuffle=True,
                                               seed=42,
                                               interpolation="bilinear")

sample = train_data.take(1).get_single_element()
print(sample.shape)
