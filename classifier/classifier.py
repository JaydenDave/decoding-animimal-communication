import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
    callbacks,
    losses,
    utils,
    metrics,
    optimizers,
)

DIM = 64
CHANNELS = 1

def classifier(num_cat, slice_len):   
    input_size = slice_len
    input = layers.Input(shape = (input_size, CHANNELS,), name = "classifier_input")
    x = layers.Conv1D(filters = DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(input)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv1D(filters = 2*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv1D(filters = 4*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv1D(filters = 8*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)

    x = layers.Conv1D(filters = 16*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    
    if slice_len ==32768:
        x = layers.Conv1D(filters = 32*DIM, strides = 2, kernel_size = 25, padding = "same", use_bias = True)(x) 
        x = layers.LeakyReLU(alpha = 0.2)(x)
    
    elif slice_len ==65536:
        x = layers.Conv1D(filters = 32*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x) 
        x = layers.LeakyReLU(alpha = 0.2)(x)
    x = layers.Flatten()(x)
    
    output = layers.Dense(units = num_cat, use_bias = True)(x)
    #dont need softmax activation because the softmax cross entropy on the loss function has it built in

    classifier = models.Model(input, output, name="classifier")
    return classifier


class CLASSIFICATION_MODEL(models.Model):
    def __init__(self,n_categories,slice_len):
        super(CLASSIFICATION_MODEL, self).__init__()

        self.classifier = classifier(n_categories, slice_len)
        self.n_categories = n_categories

    def compile(self, optimizer):
        super(CLASSIFICATION_MODEL, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.optimizer = optimizer

        self.loss_metric = metrics.Mean(name="loss")
        self.train_accuracy_metric = metrics.Accuracy(name= "train_acc")

        self.optimizer.build(self.classifier.trainable_variables)

        
    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.train_accuracy_metric,
            ]

      
    def train_step(self, input_data):
        data= input_data[0]
        labels = input_data[1]
        batch_size = tf.shape(data)[0]

        with tf.GradientTape() as tape:
            predictions = self.classifier(data, training = True)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = predictions))
        
        gradient = tape.gradient(loss, self.classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.classifier.trainable_variables))
        
        self.loss_metric.update_state(loss)
        self.train_accuracy_metric.update_state(labels=tf.argmax(labels, 1), 
                                  predictions=tf.argmax(predictions,1))
       
        
        return {m.name: m.result() for m in self.metrics}
