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
CHANNELS = 1 #keeping as 1? what for mono or stereo?
PHASE_PARAM = 2
LATENT_DIM = 100
DISCRIMINATOR_STEPS = 5
GP_WEIGHT = 10
LEARNING_RATE = 1e-4
ADAM_BETA_1 = 0.5
ADAM_BETA_2 = 0.9
BATCH_SIZE = 64
EPOCHS = 100


def generator(slice_len):
    dim_mul = 16 if slice_len == 16384 else 32
    gen_input = layers.Input(shape=(100,)) # input vector of legnth 100 (z sampled from uniform normal dist)
    x = layers.Dense(units = 4*4*dim_mul*DIM, use_bias = True)(gen_input)
    x = layers.Reshape((16,dim_mul*DIM))(x)
    x = layers.ReLU()(x)
    dim_mul //=2

    x = layers.Conv1DTranspose(filters = dim_mul*DIM, strides = 4, padding= "same", kernel_size = 25, use_bias = True)(x)
    x = layers.ReLU()(x)
    dim_mul //=2

    x = layers.Conv1DTranspose(filters = dim_mul*DIM, strides = 4, padding= "same", kernel_size = 25, use_bias = True)(x)
    x = layers.ReLU()(x)
    dim_mul //=2

    x = layers.Conv1DTranspose(filters = dim_mul*DIM, strides = 4, padding= "same", kernel_size = 25, use_bias = True)(x)
    x = layers.ReLU()(x)
    dim_mul //=2
   
    x = layers.Conv1DTranspose(filters = dim_mul*DIM, strides = 4, padding= "same", kernel_size = 25, use_bias = True)(x)
    x = layers.ReLU()(x)

    if slice_len ==16384:
        gen_output = layers.Conv1DTranspose(filters = CHANNELS, strides = 4, padding= "same", kernel_size = 25, use_bias = True, activation = 'tanh')(x)
    
    elif slice_len ==32768:
        x = layers.Conv1DTranspose(filters = DIM, strides = 4, padding= "same", kernel_size = 25, use_bias = True)(x)
        x = layers.ReLU()(x)
        gen_output = layers.Conv1DTranspose(filters = CHANNELS, strides = 2, padding= "same", kernel_size = 25, use_bias = True, activation = 'tanh')(x)
    
    elif slice_len ==65536:
        x = layers.Conv1DTranspose(filters = DIM, strides = 4, padding= "same", kernel_size = 25, use_bias = True)(x)
        x = layers.ReLU()(x)
        gen_output = layers.Conv1DTranspose(filters = CHANNELS, strides = 4, padding= "same", kernel_size = 25, use_bias = True, activation = 'tanh')(x)

    generator = models.Model(gen_input, gen_output, name= "generator")
    return generator

class PhaseShuffle(layers.Layer):
    def call(self, x):
        n= PHASE_PARAM
        b, x_len, nch = x.get_shape().as_list()
        phase = tf.random.uniform([], minval=-n, maxval=n + 1, dtype=tf.int32)
        pad_l = tf.maximum(phase, 0)
        pad_r = tf.maximum(-phase, 0)
        phase_start = pad_r
        x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode='reflect')

        x = x[:, phase_start:phase_start+x_len]
        x.set_shape([b, x_len, nch])
        return x

def discriminator(slice_len):
    input_size = slice_len
    dis_input = layers.Input(shape = (input_size, CHANNELS,), name = "discriminator_input")
    x = layers.Conv1D(filters = DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(dis_input)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 2*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 4*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 8*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 16*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    
    if slice_len ==32768:
        x = layers.Conv1D(filters = 32*DIM, strides = 2, kernel_size = 25, padding = "same", use_bias = True)(x) 
        x = layers.LeakyReLU(alpha = 0.2)(x)
    
    elif slice_len ==65536:
        x = layers.Conv1D(filters = 32*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x) 
        x = layers.LeakyReLU(alpha = 0.2)(x)
    x = layers.Flatten()(x)

    dis_output = layers.Dense(units = 1, use_bias = True)(x)
    discriminator = models.Model(dis_input, dis_output, name="discriminator")
    return discriminator

def auxiliary(num_cat, slice_len):   
    input_size = slice_len
    aux_input = layers.Input(shape = (input_size, CHANNELS,), name = "aux_input")
    x = layers.Conv1D(filters = DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(aux_input)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    #x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 2*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    #x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 4*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    #x = PhaseShuffle()(x)

    x = layers.Conv1D(filters = 8*DIM, strides = 4, kernel_size = 25, padding = "same", use_bias = True)(x)
    x = layers.LeakyReLU(alpha = 0.2)(x)
    #x = PhaseShuffle()(x)

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

    auxiliary = models.Model(aux_input, output, name="auxiliary")
    return auxiliary

class GAN(models.Model):
    def __init__(self, latent_dim, discriminator_steps, gp_weight, n_categories,slice_len):
        super(GAN, self).__init__()
        self.discriminator = discriminator(slice_len)
        self.generator = generator(slice_len)
        self.auxiliary = auxiliary(n_categories, slice_len)
        self.latent_dim = latent_dim
        self.discriminator_steps = discriminator_steps
        self.gp_weight = gp_weight
        self.n_categories = n_categories

    
    
    def compile(self, d_optimizer, g_optimizer, q_optimizer):
        super(GAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.q_optimizer = q_optimizer
        self.d_wass_loss_metric = metrics.Mean(name = "d_wass_loss")
        self.d_gp_metric = metrics.Mean(name = "d_gp")
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")
        self.q_loss_metric = metrics.Mean(name="q_loss")
        self.d_acc_real_metric = metrics.Accuracy(name = "d_acc_real")
        self.d_acc_gen_metric = metrics.Accuracy(name = "d_acc_gen")
        self.d_optimizer.build(self.discriminator.trainable_variables)
        self.g_optimizer.build(self.generator.trainable_variables)
        self.q_optimizer.build(self.generator.trainable_variables+ self.auxiliary.trainable_variables)
        
    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.g_loss_metric,
            self.d_gp_metric,
            self.d_wass_loss_metric,
            self.q_loss_metric,
            self.d_acc_real_metric,
            self.d_acc_gen_metric,
            ]
    
    def gradient_penalty(self, batch_size, real_data, fake_data):
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0) # each audio file in the batch gets a random number between 0 and 1, stored as the vector alpha
        diff = fake_data - real_data
        interpolated = real_data +alpha*diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.discriminator(interpolated, training = True)

        #gradient of the preds wrt the inputs
        grads = gp_tape.gradient(pred, [interpolated])[0]

        norm= tf.sqrt(tf.reduce_sum(tf.square(grads), axis = [1,2]))
        gp = tf.reduce_mean((norm -1.0)**2) #returns avg square distance between L2 norm and 1
        return gp
    
    def q_cost_tf(self, input, q):
        #q is Q(G(z,c)) output from q network
        z_dim = self.latent_dim - self.n_categories - self.n_cont
        input_cat = input[:, z_dim+self.n_cont:]
        input_con = input[:, z_dim: z_dim+self.n_cont:]
        
        q_cat = q[:, self.n_cont:]
        q_con = q[:, :self.n_cont]
        lcat = tf.nn.softmax_cross_entropy_with_logits(labels=input_cat, logits=q_cat)
        lcon =  0.5 * tf.square(input_con - q_con)
        return tf.reduce_mean(lcat) + tf.reduce_mean(lcon)
    
    
    def create_inputs(self, batch_size):
        #incompressible noise vector
        z_dim = self.latent_dim - self.n_categories
        #continuous codes are the same as z
        z = tf.random.normal(shape=(batch_size, z_dim))

        #categorical latent codes
        #list of integers range 0 -> n_categories (batch size number of them)
        #idxs = np.random.randint(self.n_categories, size=batch_size)
        #create array of zeros with batch size rows and n_categories columns
        #cat = np.zeros((batch_size, self.n_categories))
        #set elements of c to 1. the np.arange bit is looking at each row, and then sets row[idx] to 1 (with the random index) 
        #cat[np.arange(batch_size), idxs] = 1
        #c = tf.convert_to_tensor(cat, dtype=tf.float32)
        
        #fiw codes
        c = tf.random.normal(shape=(batch_size, self.n_categories))
        #thresholding values to 0 and 1
        c = tf.cast(c < 0, dtype=tf.float32)



        inputs = tf.concat([z, c], axis=1)
        return inputs
    
        
    def train_step(self, real_data):
        batch_size = tf.shape(real_data)[0]

        #update discriminator a few times
        for i in range(self.discriminator_steps):
            inputs = self.create_inputs(BATCH_SIZE)

            with tf.GradientTape() as tape:
                generated_data = self.generator(inputs, training = True)
                generated_predictions = self.discriminator(generated_data, training = True)
                real_predictions = self.discriminator(real_data, training = True)

                d_wass_loss = tf.reduce_mean(generated_predictions) - tf.reduce_mean(real_predictions)
                d_gp = self.gradient_penalty(batch_size, real_data, generated_data)
                d_loss = d_wass_loss + d_gp*self.gp_weight
            
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_gradient, self.discriminator.trainable_variables))
        
        #update generator
        inputs = self.create_inputs(BATCH_SIZE)
        z_dim = self.latent_dim - self.n_categories
        c = inputs[:, z_dim:]
        
        with tf.GradientTape(persistent= True) as tape:
            generated_data = self.generator(inputs, training = True)
            generated_predictions = self.discriminator(generated_data, training = True)
            code_predictions = self.auxiliary(generated_data, training = True)
            g_loss = -tf.reduce_mean(generated_predictions)
            #q_loss = self.q_cost_tf(inputs, code_predictions)
            q_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = c, logits = code_predictions))
        

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        
        
        #update g and q together with q loss
        q_gen_gradient, q_aux_gradient = tape.gradient(q_loss, [self.generator.trainable_variables,self.auxiliary.trainable_variables])
        self.q_optimizer.apply_gradients(zip(q_gen_gradient, self.generator.trainable_variables))
        self.q_optimizer.apply_gradients(zip(q_aux_gradient, self.auxiliary.trainable_variables))

        #update generator based off g loss
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        # discriminator accuracy metric
        real_predictions = self.discriminator(real_data, training = True)
        #set predictions to -1 is <0 or 1 if >=0
        real_predictions = tf.where(tf.greater_equal(real_predictions, 0), tf.ones_like(real_predictions), tf.ones_like(real_predictions) * -1)
        generated_predictions = tf.where(tf.greater_equal(generated_predictions, 0), tf.ones_like(generated_predictions), tf.ones_like(generated_predictions) * -1)

        #creating tensors for true labels
        real_true_labels = tf.ones_like(real_predictions)
        gen_true_labels = tf.ones_like(generated_predictions)

        #metric.update_state(true, pred)
        self.d_acc_gen_metric.update_state(real_true_labels, real_predictions)
        self.d_acc_real_metric.update_state(gen_true_labels, generated_predictions)
        self.d_loss_metric.update_state(d_loss)
        self.d_wass_loss_metric.update_state(d_wass_loss)
        self.d_gp_metric.update_state(d_gp)
        self.g_loss_metric.update_state(g_loss)
        self.q_loss_metric.update_state(q_loss)
        
        
        return {m.name: m.result() for m in self.metrics}
