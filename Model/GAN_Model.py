import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import scipy
import csv

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy, Mean
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.utils  import img_to_array, load_img, plot_model, Progbar

class SPADE(Layer):

    def __init__(self, filters):
        super(SPADE, self).__init__()

        self.epsilon = 1e-08

        self.conv1 = tf.keras.layers.Conv2D(128, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),  
                                            use_bias=False)
        self.conv2 = tf.keras.layers.Conv2D(filters, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),  
                                            use_bias=False)
        self.conv3 = tf.keras.layers.Conv2D(filters, 
                                            kernel_size=3, 
                                            strides=1, 
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),  
                                            use_bias=False)
        
    def build(self, input_shape):
    
        self.batch, self.width, self.height, self.channels = input_shape[0]

    def call(self, inputs):

        x, mask = inputs

        mask = tf.image.resize(mask, [self.width, self.height], method='nearest')
        mask = self.conv1(mask)
        mask = tf.nn.relu(mask)

        gamma = self.conv2(mask)
        beta = self.conv3(mask)

        #First and second moments of the normalized tensors along the [0, 1, 2] axes.
        mean, variance = tf.nn.moments(x, [0, 1, 2], keepdims=True)

        standard_deviation = tf.sqrt(tf.math.add(variance, self.epsilon))
        #Normalization.
        x = tf.math.divide(tf.math.subtract(x, mean), standard_deviation)

        outputs = tf.math.add(tf.math.multiply(x, gamma), beta)

        return outputs

class GeneratorBlockUp(Layer):

    def __init__(self, filters, apply_dropout=None):
        super(GeneratorBlockUp, self).__init__()

        self.apply_dropout = apply_dropout

        self.transconv = tf.keras.layers.Conv2DTranspose(filters, 
                                                         kernel_size=4, 
                                                         strides=2, 
                                                         padding='same', 
                                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),
                                                         use_bias=False)

        self.norm = SPADE(filters)

    def call(self, inputs):

        x, mask = inputs

        x = self.transconv(x)
        x = self.norm([x, mask]) 
        if self.apply_dropout:
            x = tf.nn.dropout(x, 0.5)
        outputs = tf.nn.relu(x)

        return outputs

class GeneratorBlockDown(Layer):

    def __init__(self, filters, apply_norm=None):
        super(GeneratorBlockDown, self).__init__()

        self.apply_norm = apply_norm

        self.conv = tf.keras.layers.Conv2D(filters, 
                                           kernel_size=4, 
                                           strides=2, 
                                           padding='same', 
                                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),  
                                           use_bias=False)

        self.norm = SPADE(filters)

    def call(self, inputs):

        x, mask = inputs

        x = self.conv(x)
        if self.apply_norm:
            x = self.norm([x, mask]) 
        outputs = tf.nn.leaky_relu(x)

        return outputs

class DiscriminatorBlockDown(Layer):

    def __init__(self, filters, apply_norm=None):
        super(DiscriminatorBlockDown, self).__init__()

        self.apply_norm = apply_norm

        self.conv = tf.keras.layers.Conv2D(filters, 
                                           kernel_size=4, 
                                           strides=2, 
                                           padding='same', 
                                           kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),  
                                           use_bias=False)

        self.norm = tfa.layers.InstanceNormalization()

    def call(self, inputs):

        x = inputs

        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x) 
        outputs = tf.nn.leaky_relu(x)

        return outputs

class Generator(Model):

    def __init__(self, filter_multiplier):
        super(Generator, self).__init__()

        self.filter_multiplier = filter_multiplier
 
        self.downblocks = [GeneratorBlockDown(1*filter_multiplier),                   #256
                           GeneratorBlockDown(2*filter_multiplier, apply_norm=True),  #128
                           GeneratorBlockDown(4*filter_multiplier, apply_norm=True),  #64
                           GeneratorBlockDown(8*filter_multiplier, apply_norm=True),  #32
                           GeneratorBlockDown(8*filter_multiplier, apply_norm=True),  #16
                           GeneratorBlockDown(8*filter_multiplier, apply_norm=True),  #8
                           GeneratorBlockDown(8*filter_multiplier, apply_norm=True),  #4
                           GeneratorBlockDown(8*filter_multiplier, apply_norm=True),  #2
                           GeneratorBlockDown(8*filter_multiplier, apply_norm=True)]  #1   #Additional layer.

        self.upblocks = [GeneratorBlockUp(8*filter_multiplier, apply_dropout=True),    #2
                         GeneratorBlockUp(8*filter_multiplier, apply_dropout=True),    #4
                         GeneratorBlockUp(8*filter_multiplier, apply_dropout=True),    #8
                         GeneratorBlockUp(8*filter_multiplier),                        #16 
                         GeneratorBlockUp(8*filter_multiplier),                        #32 #Additional layer.
                         GeneratorBlockUp(4*filter_multiplier),                        #64
                         GeneratorBlockUp(2*filter_multiplier),                        #128
                         GeneratorBlockUp(1*filter_multiplier)]                        #256
                         
        self.transconv = tf.keras.layers.Conv2DTranspose(1,                            #512
                                                         kernel_size=4, 
                                                         strides=2, 
                                                         padding='same', 
                                                         kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02),  
                                                         use_bias=False)

    @tf.function
    def call(self, inputs):
        #Unpacks the input tensors.
        x, mask = inputs
        skips = []

        for down in self.downblocks:
            x = down([x, mask])
            skips.append(x)
        
        skips = reversed(skips[:-1])

        for up, skip in zip(self.upblocks, skips):
            x = up([x, mask])
            x = tf.concat([x, skip], -1)

        x = self.transconv(x)
        outputs = tf.math.tanh(x)

        return outputs

    def build_graph(self):
        x = Input(shape=(512, 512, 1))
        m = Input(shape=(512, 512, 1))
        return Model(inputs=[x, m], outputs=self.call([x, m]), name='Generator')

class Discriminator(Model):

    def __init__(self, filter_multiplier):
        super(Discriminator, self).__init__()

        self.filter_multiplier = filter_multiplier
        
        self.blockdown1 = DiscriminatorBlockDown(1*filter_multiplier)                   #256
        self.blockdown2 = DiscriminatorBlockDown(2*filter_multiplier, apply_norm=True)  #128
        self.blockdown3 = DiscriminatorBlockDown(4*filter_multiplier, apply_norm=True)  #64

        self.conv1 = tf.keras.layers.Conv2D(8*filter_multiplier,                        #32
                                            kernel_size=4, 
                                            strides=1,                              
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), 
                                            use_bias=False)
        
        self.norm = tfa.layers.InstanceNormalization()

        self.conv2 = tf.keras.layers.Conv2D(1,                                          #32
                                            kernel_size=4, 
                                            strides=1, 
                                            padding='same', 
                                            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02), 
                                            use_bias=False)

    @tf.function
    def call(self, inputs):

        x1, x2 = inputs

        x = tf.concat([x1, x2], -1)

        x = self.blockdown1(x)
        x = self.blockdown2(x)
        x = self.blockdown3(x)

        x = self.conv1(x)
        x = self.norm(x)
        x = tf.nn.leaky_relu(x)

        outputs = self.conv2(x)
        
        return outputs

    def build_graph(self):
        x1 = Input(shape=(512, 512, 1))
        x2 = Input(shape=(512, 512, 1))
        return Model(inputs=[x1, x2], outputs=self.call([x1, x2]), name='Discriminator')

class GAN_Data():

    def __init__(self, dataframe, batch_size):

        self.df = pd.read_csv(dataframe)
        self.batch_size = batch_size

        self.input_images = self.df['InputImages'].to_numpy()
        self.mask_images = self.df['MaskImages'].to_numpy()
        self.target_images = self.df['TargetImages'].to_numpy()

        self.indices = np.arange(0, len(self.df), 1, dtype=np.int32)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def load_images(self, path):
        loaded = img_to_array(load_img(path, color_mode='grayscale'))
        normal = ((loaded / 255.) * 2.) -1. #Scales the images from [0, 255] to [-1, 1]  
        return normal

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))
        
    def get_batch_indicies(self, idx):
        return self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]

    def get_batch_images(self, df, batch_indices):
        return np.array([self.load_images(i) for i in df[batch_indices]])

    def __getitem__(self, idx):
        batch_indices = self.get_batch_indicies(idx)
        batch_masks = self.get_batch_images(self.mask_images, batch_indices)
        batch_targets = self.get_batch_images(self.target_images, batch_indices)
        batch_inputs = self.get_batch_images(self.input_images, batch_indices)
        return batch_inputs, batch_masks, batch_targets

class Progress_Visualiser():

    def __init__(self, save, val_data, frequency):

        self.save = save
        self.val_data = val_data
        self.frequency = frequency

    def visualise_progress(self, epoch, generator):

        if epoch % self.frequency == 0:

            progress = []
            for step in range(len(self.val_data)):

                input, mask, target = self.val_data[step]        
                output = generator([input, mask])

                record = [input, target, output]
                for r in record: 
                    progress.append(r)    
                                   
            width = len(record)
            height = len(progress)//width

            fig = plt.figure(figsize=(5.12*width, 5.12*height), dpi=300) 
            for i in range(len(progress)):
                sub = fig.add_subplot(height, width, i + 1)
                sub.imshow(progress[i][0,:,:,0], cmap='gray')
                plt.axis('off')
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig('{}Epoch-{}.png'.format(self.save, epoch), dpi=300)
            fig.set_size_inches(1*width, 1*height, forward=True)
            plt.show()

        return

class GAN(Model):

    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def g_loss_function(self, dg):
        #If Discriminator predicts -1 (is not fooled by G), then G loss is equal to 1.
        #If Discriminator predicts 1 (is dominated by G), then G loss is equal to -1.
        #If Discriminator cannot decide and outputs 0, then G loss is equal to 0.
        L_G = -1 * tf.math.reduce_mean(dg)                                           
        return L_G

    #def g_loss_function(self, dg):
    #    L_G = tf.math.reduce_mean(tf.nn.relu(1.0 - dg))                        #Target predictions  1.   
    #    return L_G                                           

    def d_loss_function(self, dg, dd):
        L_D  = tf.math.reduce_mean(tf.nn.relu(1.0 + dg))                        #Target predictions -1.   
        L_D += tf.math.reduce_mean(tf.nn.relu(1.0 - dd))                        #Target predictions  1. 
        return L_D                                   

    def optimizers(self, g_optimizer, d_optimizer):
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
    
    def metrics(self, g_losses, d_losses, d_acc, g_acc):
        #Loss metrics (average of loss per epoch).
        self.g_losses = g_losses
        self.d_losses = d_losses
        #Accuracy metrics (average of acc per epoch).
        self.g_acc = g_acc
        self.d_acc = d_acc

    def gan_log(self, epoch):

        with open(self.log_df_path, 'a+') as df:

            writer = csv.writer(df)
            if df.tell() == 0: 
              
                writer.writerow(['Epoch', 'G Loss', 'D Loss', 'G Acc', 'D Acc'])
            
            writer.writerow([epoch, 
                             float(self.g_losses.result()), 
                             float(self.d_losses.result()),
                             float(self.g_acc.result()),
                             float(self.d_acc.result())])
                             
        return

    def gan_train(self, epoch):

        gan_epoch = Progbar(target=len(self.gan_data), verbose=1, stateful_metrics=['g_loss', 'd_loss', 'g_acc', 'd_acc'])
        for step in range(0, len(self.gan_data), 1):

            input, mask, target = self.gan_data[step]

            with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

                gg = self.generator([input, mask])                        

                dg = self.discriminator([input, gg])                             
                dd = self.discriminator([input, target])                        
                
                g_loss = self.g_loss_function(dg)
                d_loss = self.d_loss_function(dg, dd)

            if epoch % self.g_update_frequency == 0:
                g_gradients = g_tape.gradient(g_loss, self.generator.trainable_weights)
                self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_weights))

            if epoch % self.d_update_frequency == 0:
                d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
                self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_weights))

            self.g_losses.update_state(g_loss)
            self.d_losses.update_state(d_loss)

            self.g_acc.update_state(tf.ones_like(dg), dg)
            self.d_acc.update_state(tf.ones_like(dd), dd)
            
            gan_epoch.update(current=step+1, values=[('g_loss', self.g_losses.result()),
                                                     ('d_loss', self.d_losses.result()),
                                                     ('g_acc', self.g_acc.result()),
                                                     ('d_acc', self.d_acc.result())])     

    def gan_fit(self, 
                epoch_first, epoch_last, batch_size,
                g_update_frequency, d_update_frequency, gan_data, 
                g_progress, log_df_path, save_directory, save_maxnumber, save_frequency, save_restoring=None):

        self.epoch_first = epoch_first
        self.epoch_last = epoch_last
        self.batch_size = batch_size
        self.g_update_frequency = g_update_frequency
        self.d_update_frequency = d_update_frequency
        self.gan_data = gan_data
        self.g_progress = g_progress
        self.log_df_path = log_df_path
        self.save_directory = save_directory
        self.save_maxnumber = save_maxnumber
        self.save_frequency = save_frequency
        self.save_restoring = save_restoring

        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              discriminator=self.discriminator,
                                              g_optimizer=self.g_optimizer,
                                              d_optimizer=self.d_optimizer)
        
        self.manager = tf.train.CheckpointManager(checkpoint=self.checkpoint,
                                                  directory=self.save_directory,
                                                  max_to_keep=self.save_maxnumber,
                                                  checkpoint_name='ckpt')

        #Models are restored if desired checkpoint path is supplied.
        if save_restoring:
            self.gan_train(0) #Dummy epoch to initialize the models and optimizers.
            self.checkpoint.restore(save_path=self.save_restoring).assert_consumed()
            print('GAN restored.')

        for epoch in range(epoch_first, epoch_last+1, 1):

            print('\nEpoch {}/{}:'.format(epoch, epoch_last))

            #Data is shuffled before batching.
            self.gan_data.shuffle()

            #GAN is trained.
            self.gan_train(epoch)

            #Results are logged.
            self.gan_log(epoch) 

            #Models are checkpointed and saved every n epochs.
            if epoch % save_frequency == 0:
                self.manager.save(checkpoint_number=epoch)
                print('GAN saved.')
            
            #Gan progress is visualised.
            self.g_progress.visualise_progress(epoch, self.generator)

            #Reset losses at the end of the epoch.
            self.g_losses.reset_states()
            self.d_losses.reset_states()
            #Reset accuracies at the end of the epoch.
            self.g_acc.reset_states()
            self.d_acc.reset_states()

#Sub-models.
generator = Generator(filter_multiplier=64)
discriminator = Discriminator(filter_multiplier=64)

model = GAN(generator, discriminator)

model.optimizers(g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, amsgrad=False), 
                 d_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999, epsilon=1e-08, amsgrad=False)) 
                 
model.metrics(g_losses=Mean(), 
              d_losses=Mean(), 
              g_acc=BinaryAccuracy(threshold=0.0), #G' acc against D.
              d_acc=BinaryAccuracy(threshold=0.0)) #D' acc against True data.

batch_size = 1

gan_data = GAN_Data(dataframe='/path/to/the/training/dataframe.csv', 
                    batch_size=batch_size)

val_data = GAN_Data(dataframe='/path/to/the/validation/dataframe.csv', 
                    batch_size=1)

g_progress = Progress_Visualiser(save='/path/to/the/progress/images/directory/',
                                 val_data=val_data,
                                 frequency=1)

model.gan_fit(epoch_first=1,
              epoch_last=4096,
              batch_size=batch_size,
              g_update_frequency=1,
              d_update_frequency=1,
              gan_data=gan_data,
              g_progress=g_progress,
              log_df_path='/path/to/the/log.csv',
              save_directory='/path/to/the/checkpoint/directory/',
              save_maxnumber=None,
              save_frequency=64,
              save_restoring=None)
              #save_restoring='/path/ckpt-1') #Path to the desired checkpoint.

#To plot and summarize, comment out the @tf.function decorator.
#generator = Generator(filter_multiplier=64)
#discriminator = Discriminator(filter_multiplier=64)
#generator.build_graph().summary()
#discriminator.build_graph().summary()
#plot_model(generator.build_graph(), show_shapes=True, show_dtype=True, expand_nested=False)
#plot_model(discriminator.build_graph(), show_shapes=True, show_dtype=True, expand_nested=False)        
