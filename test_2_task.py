import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from collections import OrderedDict
from PIL import Image
from skimage.restoration import denoise_nl_means, estimate_sigma
import numpy as np
import random
import datetime
import time
import json
import csv
import sys
import tensorflow as tf
import load_data_2_task as load_data
import imageio
from PIL import Image
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.optimizers as KO
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import SimpleITK as sitk
from scipy import ndimage, misc
print(tf.config.list_physical_devices('GPU'))
 
np.random.seed(seed=12345)


class CycleGAN():
    def __init__(self, lr_D=2e-4, lr_G=3e-4, image_shape=(320, 320, 2),
                 date_time_string_addition='', image_folder='', mode='train'):
        self.img_shape = image_shape
        self.channels = self.img_shape[-1]
        self.normalization = tfa.layers.InstanceNormalization
        # Hyper parameters
        self.lambda_1 = 10.0  # Cyclic loss weight A_2_B
        self.lambda_2 = 8.0  # Cyclic loss weight B_2_A
        self.lambda_D = 1.0  # Weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1  # Number of generator training iterations in each training loop
        self.discriminator_iterations = 1  # Number of generator training iterations in each training loop
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 5
        self.epochs = 60  # choose multiples of 20 since the models are save each 20th epoch
        self.save_interval = 1
        self.synthetic_pool_size = 25

        # Linear decay of learning rate, for both discriminators and generators
        self.use_linear_decay = True
        self.decay_epoch = 101  # The epoch where the linear decay of the learning rates start

        # Identity loss - sometimes send images from B to G_A2B (and the opposite) to teach identity mappings
        self.use_identity_learning = True
        self.identity_mapping_modulus = 10  # Identity mapping will be done each time the iteration number is divisable with this number

        # PatchGAN - if false the discriminator learning rate should be decreased
        self.use_patchgan = True

        # Multi scale discriminator - if True the generator have an extra encoding/decoding step to match discriminator information access
        self.use_multiscale_discriminator = False

        # Resize convolution - instead of transpose convolution in deconvolution layers (uk) - can reduce checkerboard artifacts but the blurring might affect the cycle-consistency
        self.use_resize_convolution = False

        # Supervised learning part - for MR images - comparison
        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        # Fetch data during training instead of pre caching all images - might be necessary for large datasets
        self.use_data_generator = False

        # Tweaks
        self.REAL_LABEL = 0.95  # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = KO.Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = KO.Adam(self.learning_rate_G, self.beta_1, self.beta_2)

        # ======= Discriminator model ==========
        if self.use_multiscale_discriminator:
            D_A = self.modelMultiScaleDiscriminator()
            D_B = self.modelMultiScaleDiscriminator()
            loss_weights_D = [0.5, 0.5] # 0.5 since we train on real and synthetic images
        else:
            D_A = self.modelDiscriminator()
            D_B = self.modelDiscriminator()
            loss_weights_D = [0.5]  # 0.5 since we train on real and synthetic images
        # D_A.summary()

        # Discriminator builds
        image_A = KL.Input(shape=self.img_shape)
        image_B = KL.Input(shape=self.img_shape)
        guess_A = D_A(image_A)
        guess_B = D_B(image_B)
        self.D_A = KM.Model(inputs=image_A, outputs=guess_A, name='D_A_model')
        self.D_B = KM.Model(inputs=image_B, outputs=guess_B, name='D_B_model')

        # self.D_A.summary()
        # self.D_B.summary()
        self.D_A.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)
        self.D_B.compile(optimizer=self.opt_D,
                         loss=self.lse,
                         loss_weights=loss_weights_D)

        # Use containers to avoid falsy keras error about weight descripancies
        self.D_A_static = KM.Model(inputs=image_A, outputs=guess_A, name='D_A_static_model')
        self.D_B_static = KM.Model(inputs=image_B, outputs=guess_B, name='D_B_static_model')

#######################################################################################################
        # ======= Generator model ==========
        # Do not update discriminator weights during generator training
        self.D_A_static.trainable = False
        self.D_B_static.trainable = False

        # Generators
        self.G_A2B = self.modelGenerator(name='G_A2B_model')
        self.G_B2A = self.modelGenerator(name='G_B2A_model')
        # self.G_A2B.summary()

        if self.use_identity_learning:
            self.G_A2B.compile(optimizer=self.opt_G, loss='MAE')
            self.G_B2A.compile(optimizer=self.opt_G, loss='MAE')

        # Generator builds
        real_A = KL.Input(shape=self.img_shape, name='real_A')
        real_B = KL.Input(shape=self.img_shape, name='real_B')
        synthetic_B = self.G_A2B(real_A)
        synthetic_A = self.G_B2A(real_B)
        dA_guess_synthetic = self.D_A_static(synthetic_A)
        dB_guess_synthetic = self.D_B_static(synthetic_B)
        reconstructed_A = self.G_B2A(synthetic_B)
        reconstructed_B = self.G_A2B(synthetic_A)

        model_outputs = [reconstructed_A, reconstructed_B]
        compile_losses = [self.cycle_loss, self.cycle_loss,
                          self.lse, self.lse]
        compile_weights = [self.lambda_1, self.lambda_2,
                           self.lambda_D, self.lambda_D]

        if self.use_multiscale_discriminator:
            for _ in range(2):
                compile_losses.append(self.lse)
                compile_weights.append(self.lambda_D)  # * 1e-3)  # Lower weight to regularize the model
            for i in range(2):
                model_outputs.append(dA_guess_synthetic[i])
                model_outputs.append(dB_guess_synthetic[i])
        else:
            model_outputs.append(dA_guess_synthetic)
            model_outputs.append(dB_guess_synthetic)

        if self.use_supervised_learning:
            model_outputs.append(synthetic_A)
            model_outputs.append(synthetic_B)
            compile_losses.append('MAE')
            compile_losses.append('MAE')
            compile_weights.append(self.supervised_weight)
            compile_weights.append(self.supervised_weight)

        self.G_model = KM.Model(inputs=[real_A, real_B],
                             outputs=model_outputs,
                             name='G_model')

        self.G_model.compile(optimizer=self.opt_G,
                             loss=compile_losses,
                             loss_weights=compile_weights)
        # self.G_A2B.summary()

        # ======= Data ==========
        # Use 'None' to fetch all available images
        nr_S1_train_imgs = None
        nr_S2_train_imgs = None
        nr_T_train_imgs = None

        nr_S1_test_imgs = None
        nr_S2_test_imgs = None
        nr_T_test_imgs = None

        if self.use_data_generator:
            print('--- Using dataloader during training ---')
        else:
            print('--- Caching data ---')
        sys.stdout.flush()

        if self.use_data_generator:
            self.data_generator = load_data.load_data(
                nr_of_channels=self.batch_size, generator=True, subfolder=image_folder)

            # Only store test images
            nr_S1_train_imgs = 0
            nr_S2_train_imgs = 0
            nr_S3_train_imgs = 0
            nr_T_train_imgs = 0


        # data = load_data.load_data(nr_of_channels=1,
        #                            batch_size=self.batch_size,
        #                            nr_S1_train_imgs=nr_S1_train_imgs,
        #                            nr_S2_train_imgs=nr_S2_train_imgs,
        #                            nr_S3_train_imgs=nr_S3_train_imgs,
        #                            nr_T_train_imgs=nr_T_train_imgs,
        #                            nr_S1_test_imgs=nr_S1_test_imgs,
        #                            nr_S2_test_imgs=nr_S2_test_imgs,
        #                            nr_S3_test_imgs=nr_S3_test_imgs,
        #                            nr_T_test_imgs=nr_T_test_imgs,
        #                            subfolder=image_folder)

        self.S1_train = 0
        self.S2_train = 0
        self.T_train = 0
        self.S1_test = 0
        self.S2_test = 0
        self.T_test = 0
        # self.S1_train = data["trainS1_images"]
        # self.S2_train = data["trainS2_images"]
        # self.S3_train = data["trainS3_images"]
        # self.T_train = data["trainT_images"]
        # self.S1_test = data["testS1_images"]
        # self.S2_test = data["testS2_images"]
        # self.S3_test = data["testS3_images"]
        # self.T_test = data["testT_images"]
        # self.testS1_image_names = data["testS1_image_names"]
        # self.testS2_image_names = data["testS2_image_names"]
        # self.testS3_image_names = data["testS3_image_names"]
        # self.testT_image_names = data["testT_image_names"]
        #
        # self.S_train = np.concatenate((self.S1_train, self.S2_train, self.S3_train), axis=-1)
        # self.S_test = np.concatenate((self.S1_test, self.S2_test,  self.S3_test), axis=-1)
        # self.T_train = np.concatenate((self.T_train, self.T_train, self.T_train), axis=-1)
        # self.T_test = np.concatenate((self.T_test, self.T_test, self.T_test), axis=-1)

        self.S_train = 0
        self.S_test = 0
        self.T_train = 0
        self.T_test = 0

        if not self.use_data_generator:
            print('Data has been loaded')

        # ======= Create designated run folder and store meta data ==========
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        #self.writeMetaDataToJSON()

        # ======= Avoid pre-allocating GPU memory ==========
        # TensorFlow wizardry
        config = tf.compat.v1.ConfigProto()

        # Don't pre-allocate memory; allocate as-needed
        config.gpu_options.allow_growth = True

        # Create a session with the above options specified.
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

        # ===== Tests ======
        # Simple Model
#         self.G_A2B = self.modelSimple('simple_T1_2_T2_model')
#         self.G_B2A = self.modelSimple('simple_T2_2_T1_model')
#         self.G_A2B.compile(optimizer=Adam(), loss='MAE')
#         self.G_B2A.compile(optimizer=Adam(), loss='MAE')
#         # self.trainSimpleModel()
#         self.load_model_and_generate_synthetic_images()

        # ======= Initialize training ==========
        sys.stdout.flush()
        #plot_model(self.G_A2B, to_file='GA2B_expanded_model_new.png', show_shapes=True)

        if mode == 'train' or mode == 'all':
            self.train(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)
        if mode == 'test' or mode == 'all':
            self.load_model_and_generate_synthetic_images()

#===============================================================================
# Architecture functions

    def s_ck(self, x, k, use_normalization):
        x = KL.SeparableConv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.LeakyReLU(alpha=0.2)(x)
        return x


    def ck(self, x, k, use_normalization):
        x = KL.Conv2D(filters=k, kernel_size=4, strides=2, padding='same')(x)
        # Normalization is not done on the first discriminator layer
        if use_normalization:
            x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.LeakyReLU(alpha=0.2)(x)
        return x

    def c7Ak(self, x, k):
        x = KL.SeparableConv2D(filters=k, kernel_size=7, strides=1, padding='valid')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        return x

    def dk(self, x, k):
        x = KL.Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        return x

    def Rk(self, x0):
        k = int(x0.shape[-1])
        # first layer
        x = KL.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x0)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        # second layer
        x = KL.Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        # merge
        x = KL.add([x, x0])
        return x

    def uk(self, x, k):
        # (up sampling followed by 1x1 convolution <=> fractional-strided 1/2)
        if self.use_resize_convolution:
            x = KL.UpSampling2D(size=(2, 2))(x)  # Nearest neighbor upsampling
            x = ReflectionPadding2D((1, 1))(x)
            x = KL.Conv2D(filters=k, kernel_size=3, strides=1, padding='valid')(x)
        else:
            x = KL.Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(x)  # this matches fractionally stided with stride 1/2
        x = self.normalization(axis=3, center=True, epsilon=1e-5)(x, training=True)
        x = KL.Activation('relu')(x)
        return x

#===============================================================================
# Models

    def modelMultiScaleDiscriminator(self, name=None):
        x1 = KL.Input(shape=self.img_shape)
        x2 = KL.AveragePooling2D(pool_size=(2, 2))(x1)
        #x4 = AveragePooling2D(pool_size=(2, 2))(x2)

        out_x1 = self.modelDiscriminator('D1')(x1)
        out_x2 = self.modelDiscriminator('D2')(x2)
        #out_x4 = self.modelDiscriminator('D4')(x4)

        return KM.Model(inputs=x1, outputs=[out_x1, out_x2], name=name)

    def modelDiscriminator(self, name=None):
        # Specify input
        input_img = KL.Input(shape=self.img_shape)
        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.s_ck(input_img, 64, False)
        # Layer 2
        x = self.ck(x, 128, True)
        # Layer 3
        x = self.ck(x, 256, True)
        # Layer 4
        x = self.ck(x, 512, True)
        # Output layer
        if self.use_patchgan:
            x = KL.Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = KL.Flatten()(x)
            x = KL.Dense(1)(x)
        x = KL.Activation('sigmoid')(x)
        return KM.Model(inputs=input_img, outputs=x, name=name)

    def modelGenerator(self, name=None):
        # Specify input
        input_img = KL.Input(shape=self.img_shape)
        # Layer 1
        x = ReflectionPadding2D((3, 3))(input_img)
        x = self.c7Ak(x, 48)
        # Layer 2
        x = self.dk(x, 72)
        # Layer 3
        x = self.dk(x, 128)

        if self.use_multiscale_discriminator:
            # Layer 3.5
            x = self.dk(x, 256)

        # Layer 4-12: Residual layer
        for _ in range(4, 13):
            x = self.Rk(x)

        if self.use_multiscale_discriminator:
            # Layer 12.5
            x = self.uk(x, 128)
        # Layer 13
        x = self.uk(x, 72)
        # Layer 14
        x = self.uk(x, 48)
        x = ReflectionPadding2D((3, 3))(x)
        x = KL.Conv2D(self.channels, kernel_size=7, strides=1)(x)
        x = KL.Activation('tanh')(x)  # They say they use Relu but really they do not
        return KM.Model(inputs=input_img, outputs=x, name=name)

#===============================================================================
# Test - simple model
    def modelSimple(self, name=None):
        inputImg = KL.Input(shape=self.img_shape)
        #x = Conv2D(1, kernel_size=5, strides=1, padding='same')(inputImg)
        #x = Dense(self.channels)(x)
        x = KL.Conv2D(256, kernel_size=1, strides=1, padding='same')(inputImg)
        x = KL.Activation('relu')(x)
        x = KL.Conv2D(self.channels, kernel_size=1, strides=1, padding='same')(x)

        return KM.Model(input=inputImg, output=x, name=name)

    def trainSimpleModel(self):
        real_A = self.S_test[0]
        real_B = self.T_test[0]
        real_A = real_A[np.newaxis, :, :, :]
        real_B = real_B[np.newaxis, :, :, :]
        epochs = 200
        for epoch in range(epochs):
            print('Epoch {} started'.format(epoch))
            self.G_A2B.fit(x=self.S_train, y=self.T_train, epochs=1, batch_size=1)
            self.G_B2A.fit(x=self.T_train, y=self.S_train, epochs=1, batch_size=1)
            #loss = self.G_A2B.train_on_batch(x=real_A, y=real_B)
            #print('loss: ', loss)
            synthetic_image_A = self.G_B2A.predict(real_B, batch_size=1)
            synthetic_image_B = self.G_A2B.predict(real_A, batch_size=1)
            self.save_tmp_images(real_A, real_B, synthetic_image_A, synthetic_image_B)

        self.saveModel(self.G_A2B, 200)
        self.saveModel(self.G_B2A, 200)

#===============================================================================
# Training
    def train(self, epochs, batch_size=1, save_interval=1):
        def run_training_iteration(loop_index, epoch_iterations):
            # ======= Discriminator training ==========
                # Generate batch of synthetic images
            synthetic_images_B = self.G_A2B.predict(real_images_A)
            synthetic_images_A = self.G_B2A.predict(real_images_B)
            synthetic_images_A = synthetic_pool_A.query(synthetic_images_A)
            synthetic_images_B = synthetic_pool_B.query(synthetic_images_B)

            for _ in range(self.discriminator_iterations):
                DA_loss_real = self.D_A.train_on_batch(x=real_images_A, y=ones)
                DB_loss_real = self.D_B.train_on_batch(x=real_images_B, y=ones)
                DA_loss_synthetic = self.D_A.train_on_batch(x=synthetic_images_A, y=zeros)
                DB_loss_synthetic = self.D_B.train_on_batch(x=synthetic_images_B, y=zeros)
                if self.use_multiscale_discriminator:
                    DA_loss = sum(DA_loss_real) + sum(DA_loss_synthetic)
                    DB_loss = sum(DB_loss_real) + sum(DB_loss_synthetic)
                    print('DA_losses: ', np.add(DA_loss_real, DA_loss_synthetic))
                    print('DB_losses: ', np.add(DB_loss_real, DB_loss_synthetic))
                else:
                    DA_loss = DA_loss_real + DA_loss_synthetic
                    DB_loss = DB_loss_real + DB_loss_synthetic
                D_loss = DA_loss + DB_loss

                if self.discriminator_iterations > 1:
                    print('D_loss:', D_loss)
                    sys.stdout.flush()

            # ======= Generator training ==========
            target_data = [real_images_A, real_images_B]  # Compare reconstructed images to real images
            if self.use_multiscale_discriminator:
                for i in range(2):
                    target_data.append(ones[i])
                    target_data.append(ones[i])
            else:
                target_data.append(ones)
                target_data.append(ones)

            if self.use_supervised_learning:
                target_data.append(real_images_A)
                target_data.append(real_images_B)

            for _ in range(self.generator_iterations):
                G_loss = self.G_model.train_on_batch(
                    x=[real_images_A, real_images_B], y=target_data)
                if self.generator_iterations > 1:
                    print('G_loss:', G_loss)
                    sys.stdout.flush()

            gA_d_loss_synthetic = G_loss[1]
            gB_d_loss_synthetic = G_loss[2]
            reconstruction_loss_A = G_loss[3]
            reconstruction_loss_B = G_loss[4]

            # Identity training
            if self.use_identity_learning and loop_index % self.identity_mapping_modulus == 0:
                G_A2B_identity_loss = self.G_A2B.train_on_batch(
                    x=real_images_B, y=real_images_B)
                G_B2A_identity_loss = self.G_B2A.train_on_batch(
                    x=real_images_A, y=real_images_A)
                print('G_A2B_identity_loss:', G_A2B_identity_loss)
                print('G_B2A_identity_loss:', G_B2A_identity_loss)

            # Update learning rates
            if self.use_linear_decay and epoch > self.decay_epoch:
                self.update_lr(self.D_A, decay_D)
                self.update_lr(self.D_B, decay_D)
                self.update_lr(self.G_model, decay_G)

            # Store training data
            DA_losses.append(DA_loss)
            DB_losses.append(DB_loss)
            gA_d_losses_synthetic.append(gA_d_loss_synthetic)
            gB_d_losses_synthetic.append(gB_d_loss_synthetic)
            gA_losses_reconstructed.append(reconstruction_loss_A)
            gB_losses_reconstructed.append(reconstruction_loss_B)

            GA_loss = gA_d_loss_synthetic + reconstruction_loss_A
            GB_loss = gB_d_loss_synthetic + reconstruction_loss_B
            D_losses.append(D_loss)
            GA_losses.append(GA_loss)
            GB_losses.append(GB_loss)
            G_losses.append(G_loss)
            reconstruction_loss = reconstruction_loss_A + reconstruction_loss_B
            reconstruction_losses.append(reconstruction_loss)

            print('\n')
            print('Epoch----------------', epoch, '/', epochs)
            print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
            print('D_loss: ', D_loss)
            print('G_loss: ', G_loss[0])
            print('reconstruction_loss: ', reconstruction_loss)
            print('dA_loss:', DA_loss)
            print('DB_loss:', DB_loss)

            if loop_index % 10 == 0:
                # Save temporary images continously
                self.save_tmp_images(real_images_A, real_images_B, synthetic_images_A, synthetic_images_B)
                self.print_ETA(start_time, epoch, epoch_iterations, loop_index)

        # ======================================================================
        # Begin training
        # ======================================================================
        training_history = OrderedDict()

        DA_losses = []
        DB_losses = []
        gA_d_losses_synthetic = []
        gB_d_losses_synthetic = []
        gA_losses_reconstructed = []
        gB_losses_reconstructed = []

        GA_losses = []
        GB_losses = []
        reconstruction_losses = []
        D_losses = []
        G_losses = []

        # Image pools used to update the discriminators
        synthetic_pool_A = ImagePool(self.synthetic_pool_size)
        synthetic_pool_B = ImagePool(self.synthetic_pool_size)

        # self.saveImages('(init)')

        # labels
        if self.use_multiscale_discriminator:
            label_shape1 = (batch_size,) + self.D_A.output_shape[0][1:]
            label_shape2 = (batch_size,) + self.D_A.output_shape[1][1:]
            #label_shape4 = (batch_size,) + self.D_A.output_shape[2][1:]
            ones1 = np.ones(shape=label_shape1) * self.REAL_LABEL
            ones2 = np.ones(shape=label_shape2) * self.REAL_LABEL
            #ones4 = np.ones(shape=label_shape4) * self.REAL_LABEL
            ones = [ones1, ones2]  # , ones4]
            zeros1 = ones1 * 0
            zeros2 = ones2 * 0
            #zeros4 = ones4 * 0
            zeros = [zeros1, zeros2]  # , zeros4]
        else:
            label_shape = (batch_size,) + self.D_A.output_shape[1:]
            ones = np.ones(shape=label_shape) * self.REAL_LABEL
            zeros = ones * 0

        # Linear decay
        if self.use_linear_decay:
            decay_D, decay_G = self.get_lr_linear_decay_rate()

        # Start stopwatch for ETAs
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            if self.use_data_generator:
                loop_index = 1
                for images in self.data_generator:
                    real_images_A = images[0]
                    real_images_B = images[1]
                    if len(real_images_A.shape) == 3:
                        real_images_A = real_images_A[:, :, :, np.newaxis]
                        real_images_B = real_images_B[:, :, :, np.newaxis]

                    if real_images_A.shape[0] == real_images_B.shape[0] and real_images_A.shape[0] == ones.shape[0]:
                        # Run all training steps
                        run_training_iteration(loop_index, self.data_generator.__len__())

                    # Store models
                    if loop_index % 20000 == 0:
                        self.saveModel(self.D_A, loop_index)
                        self.saveModel(self.D_B, loop_index)
                        self.saveModel(self.G_A2B, loop_index)
                        self.saveModel(self.G_B2A, loop_index)

                    # Break if loop has ended
                    if loop_index >= self.data_generator.__len__():
                        break

                    loop_index += 1

            else:  # Train with all data in cache
                A_train = self.S_train
                B_train = self.T_train
                random_order_A = np.random.randint(len(A_train), size=len(A_train))
                random_order_B = np.random.randint(len(B_train), size=len(B_train))
                epoch_iterations = max(len(random_order_A), len(random_order_B))
                min_nr_imgs = min(len(random_order_A), len(random_order_B))

                # If we want supervised learning the same images form
                # the two domains are needed during each training iteration
                if self.use_supervised_learning:
                    random_order_B = random_order_A
                for loop_index in range(0, epoch_iterations, batch_size):
                    if loop_index + batch_size >= min_nr_imgs:
                        # If all images soon are used for one domain,
                        # randomly pick from this domain
                        if len(A_train) <= len(B_train):
                            indexes_A = np.random.randint(len(A_train), size=batch_size)
                            indexes_B = random_order_B[loop_index:
                                                       loop_index + batch_size]
                        else:
                            indexes_B = np.random.randint(len(B_train), size=batch_size)
                            indexes_A = random_order_A[loop_index:
                                                       loop_index + batch_size]
                    else:
                        indexes_A = random_order_A[loop_index:
                                                   loop_index + batch_size]
                        indexes_B = random_order_B[loop_index:
                                                   loop_index + batch_size]

                    sys.stdout.flush()
                    real_images_A = A_train[indexes_A]
                    real_images_B = B_train[indexes_B]

                    if real_images_A.shape[0] == real_images_B.shape[0] and real_images_A.shape[0] == ones.shape[0]:
                        # Run all training steps
                        run_training_iteration(loop_index, epoch_iterations)

            #================== within epoch loop end ==========================

           # if epoch % save_interval == 0:
           #     print('\n', '\n', '-------------------------Saving images for epoch', epoch, '-------------------------', '\n', '\n')
               # self.saveImages(epoch, real_images_A, real_images_B)

            if epoch % 10 == 0:
                # self.saveModel(self.G_model)
                self.saveModel(self.D_A, epoch)
                self.saveModel(self.D_B, epoch)
                self.saveModel(self.G_A2B, epoch)
                self.saveModel(self.G_B2A, epoch)

            training_history = {
                'DA_losses': DA_losses,
                'DB_losses': DB_losses,
                'gA_d_losses_synthetic': gA_d_losses_synthetic,
                'gB_d_losses_synthetic': gB_d_losses_synthetic,
                'gA_losses_reconstructed': gA_losses_reconstructed,
                'gB_losses_reconstructed': gB_losses_reconstructed,
                'D_losses': D_losses,
                'G_losses': G_losses,
                'reconstruction_losses': reconstruction_losses}
            self.writeLossDataToFile(training_history)

            # Flush out prints each loop iteration
            sys.stdout.flush()

#===============================================================================
# Help functions

    def lse(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y_true))
        return loss

    def cycle_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.abs(y_pred - y_true))
        return loss

    def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
        if len(real.shape) > 3:
            real = real[0]
            synthetic = synthetic[0]
            reconstructed = reconstructed[0]

        synthetic = synthetic.clip(min=0)
        reconstructed = reconstructed.clip(min=0)

        # Append and save
        if real_ is not None:
            if len(real_.shape) > 4:
                real_ = real_[0]
            image = np.hstack((real_[0], real, synthetic, reconstructed))
        else:
            image = np.hstack((real, synthetic, reconstructed))

        if self.channels == 1:
            image = image[:, :, 0]

        imageio.imwrite(path_name, image)
        #scipy.misc.toimage(image, cmin=0, cmax=1).save(path_name)

    def saveImages(self, epoch, real_image_A, real_image_B, num_saved_images=1):
        directory = os.path.join('images', self.date_time)
        if not os.path.exists(os.path.join(directory, 'A')):
            os.makedirs(os.path.join(directory, 'A'))
            os.makedirs(os.path.join(directory, 'B'))
            os.makedirs(os.path.join(directory, 'Atest'))
            os.makedirs(os.path.join(directory, 'Btest'))

        testString = ''

        real_image_Ab = None
        real_image_Ba = None
        for i in range(num_saved_images + 1):
            if i == num_saved_images:
                real_image_A = self.S_test[0]
                real_image_B = self.T_test[0]
                real_image_A = np.expand_dims(real_image_A, axis=0)
                real_image_B = np.expand_dims(real_image_B, axis=0)
                testString = 'test'
                if self.channels == 1:  # Use the paired data for MR images
                    real_image_Ab = self.T_test[0]
                    real_image_Ba = self.S_test[0]
                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)
            else:
                #real_image_A = self.S_train[rand_A_idx[i]]
                #real_image_B = self.T_train[rand_B_idx[i]]
                if len(real_image_A.shape) < 4:
                    real_image_A = np.expand_dims(real_image_A, axis=0)
                    real_image_B = np.expand_dims(real_image_B, axis=0)
                if self.channels == 1:  # Use the paired data for MR images
                    real_image_Ab = real_image_B  # self.T_train[rand_A_idx[i]]
                    real_image_Ba = real_image_A  # self.S_train[rand_B_idx[i]]
                    real_image_Ab = np.expand_dims(real_image_Ab, axis=0)
                    real_image_Ba = np.expand_dims(real_image_Ba, axis=0)

            synthetic_image_B = self.G_A2B.predict(real_image_A)
            synthetic_image_A = self.G_B2A.predict(real_image_B)
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            self.truncateAndSave(real_image_Ab, real_image_A, synthetic_image_B, reconstructed_image_A,
                                 'images/{}/{}/epoch{}_sample{}.png'.format(
                                     self.date_time, 'A' + testString, epoch, i))
            self.truncateAndSave(real_image_Ba, real_image_B, synthetic_image_A, reconstructed_image_B,
                                 'images/{}/{}/epoch{}_sample{}.png'.format(
                                     self.date_time, 'B' + testString, epoch, i))

    def save_tmp_images(self, real_image_A, real_image_B, synthetic_image_A, synthetic_image_B):
        try:
            reconstructed_image_A = self.G_B2A.predict(synthetic_image_B)
            reconstructed_image_B = self.G_A2B.predict(synthetic_image_A)

            real_images = np.vstack((real_image_A[0], real_image_B[0]))
            synthetic_images = np.vstack((synthetic_image_B[0], synthetic_image_A[0]))
            reconstructed_images = np.vstack((reconstructed_image_A[0], reconstructed_image_B[0]))

            self.truncateAndSave(None, real_images, synthetic_images, reconstructed_images,
                                 'images/{}/{}.png'.format(
                                     self.date_time, 'tmp'))
        except: # Ignore if file is open
            pass

    def get_lr_linear_decay_rate(self):
        # Calculate decay rates
        if self.use_data_generator:
            max_nr_images = len(self.data_generator)
        else:
            max_nr_images = max(len(self.S_train), len(self.T_train))

        updates_per_epoch_D = 2 * max_nr_images + self.discriminator_iterations - 1
        updates_per_epoch_G = max_nr_images + self.generator_iterations - 1
        if self.use_identity_learning:
            updates_per_epoch_G *= (1 + 1 / self.identity_mapping_modulus)
        denominator_D = (self.epochs - self.decay_epoch) * updates_per_epoch_D
        denominator_G = (self.epochs - self.decay_epoch) * updates_per_epoch_G
        decay_D = self.learning_rate_D / denominator_D
        decay_G = self.learning_rate_G / denominator_G

        return decay_D, decay_G

    def update_lr(self, model, decay):
        new_lr = K.get_value(model.optimizer.lr) - decay
        if new_lr < 0:
            new_lr = 0
        # print(K.get_value(model.optimizer.lr))
        K.set_value(model.optimizer.lr, new_lr)

    def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
        passed_time = time.time() - start_time

        iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
        iterations_total = self.epochs * epoch_iterations / self.batch_size
        iterations_left = iterations_total - iterations_so_far
        eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

        passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
        eta_string = str(datetime.timedelta(seconds=eta))
        print('Time passed', passed_time_string, ': ETA in', eta_string)


#===============================================================================
# Save and load

    def saveModel(self, model, epoch):
        # Create folder to save model architecture and weights
        directory = os.path.join('saved_models', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path_w = 'saved_models/{}/{}_weights_epoch_{}.hdf5'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_w)
        model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch)
        model.save_weights(model_path_m)
        json_string = model.to_json()
        with open(model_path_m, 'w') as outfile:
            json.dump(json_string, outfile)
        print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))

    def writeLossDataToFile(self, history):
        keys = sorted(history.keys())
        with open('images/{}/loss_output.csv'.format(self.date_time), 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(keys)
            writer.writerows(zip(*[history[key] for key in keys]))

    def writeMetaDataToJSON(self):

        directory = os.path.join('images', self.date_time)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Save meta_data
        data = {}
        data['meta_data'] = []
        data['meta_data'].append({
            'img shape: height,width,channels': self.img_shape,
            'batch size': self.batch_size,
            'save interval': self.save_interval,
            'normalization function': str(self.normalization),
            'lambda_1': self.lambda_1,
            'lambda_2': self.lambda_2,
            'lambda_d': self.lambda_D,
            'learning_rate_D': self.learning_rate_D,
            'learning rate G': self.learning_rate_G,
            'epochs': self.epochs,
            'use linear decay on learning rates': self.use_linear_decay,
            'use multiscale discriminator': self.use_multiscale_discriminator,
            'epoch where learning rate linear decay is initialized (if use_linear_decay)': self.decay_epoch,
            'generator iterations': self.generator_iterations,
            'discriminator iterations': self.discriminator_iterations,
            'use patchGan in discriminator': self.use_patchgan,
            'beta 1': self.beta_1,
            'beta 2': self.beta_2,
            'REAL_LABEL': self.REAL_LABEL,
            'number of A train examples': len(self.S_train),
            'number of B train examples': len(self.T_train),
            'number of A test examples': len(self.S_test),
            'number of B test examples': len(self.T_test),
        })

        with open('images/{}/meta_data.json'.format(self.date_time), 'w') as outfile:
            json.dump(data, outfile, sort_keys=True)

    def load_model_and_weights(self, model):
        path_to_weights = os.path.join('/home/william/Jan_GAN/CycleGAN-2/generate_images', 'models', '{}.hdf5'.format(model.name))
        model.load_weights(path_to_weights)
    def denoise_norl(self, image):
        sigma_est = np.mean(estimate_sigma(image, multichannel=False))
        patch_kw = dict(patch_size=5,  # 5x5 patches
                        patch_distance=6,  # 13x13 search area
                        multichannel=True)
        image_de = denoise_nl_means(image, h=0.8 * sigma_est, fast_mode=True, **patch_kw)
        # intensity normalisation to [-1, 1]
        imo_den_nor = (2 * (image_de - np.min(image_de)) / (np.max(image_de) - np.min(image_de)))-1
        return imo_den_nor

    def padding(self, pred, orig):
        row_pred = np.shape(pred)[1]
        col_pred = np.shape(pred)[2]
        row_orig = np.shape(orig)[1]
        col_orig = np.shape(orig)[2]

        pred_orig = np.zeros(np.shape(orig), dtype='float32')
        pred_orig[:, int((row_orig-row_pred)/2): int((row_orig-row_pred)/2)+row_pred, int((col_orig-col_pred)/2): int((col_orig-col_pred)/2)+col_pred] = pred[...]

        return pred_orig


    def load_model_and_generate_synthetic_images(self):
        def extract_patches(model, slice_, intl):  # interval
            patches = []
            slice_pred = np.zeros(np.shape(slice_), dtype='float32')
            for ii in range(0, np.shape(slice_)[0] - 2 * intl, intl):
                for jj in range(0, np.shape(slice_)[1] - 2 * intl, intl):
                    patch = slice_[ii:ii + 2 * intl, jj:jj + 2 * intl, :]
                    syn_patch = model.predict(patch[np.newaxis, ...])
                    # if ii != 0:
                    #     orig_v = np.mean(slice_pred[ii:ii + 2 * intl, jj + 2 * intl-1]) - np.mean(syn_patch[:, 1])
                    #     syn_patch = syn_patch + orig_v
                    slice_pred[ii:ii + 2 * intl, jj:jj + 2 * intl, :] = syn_patch
                    if ii + 2*intl > np.shape(slice_)[0] - 2 * intl:
                        patch = slice_[-2 * intl:, jj:jj + 2 * intl, :]
                        syn_patch = model.predict(patch[np.newaxis, ...])
                        slice_pred[-2 * intl:, jj:jj + 2 * intl, :] = syn_patch
                    if jj + 2*intl > np.shape(slice_)[1] - 2 * intl:
                        patch = slice_[ii:ii + 2 * intl, -2 * intl:, :]
                        syn_patch = model.predict(patch[np.newaxis, ...])
                        slice_pred[ii:ii + 2 * intl, -2 * intl:, :] = syn_patch
            patch = slice_[-2 * intl:, -2 * intl:, :]
            syn_patch = model.predict(patch[np.newaxis, ...])
            # orig_v = np.mean(slice_pred[:, -2 * intl]) - np.mean(syn_patch[:, 1])
            # syn_patch = syn_patch + orig_v
            slice_pred[-2 * intl:, -2 * intl:, :] = syn_patch
            #    print('yes')
            #    patches.append(slice_denoised[ii:ii + 2 * intl, :-2 * intl])
            return slice_pred

        self.load_model_and_weights(self.G_A2B)
        self.load_model_and_weights(self.G_B2A)
        data_dir = '/home/william/CT_MR_Nrad'
        test_id = [158, 159, 160]
        t2_ch1_dir = 'testS1/'
        t2_ch2_dir = 'testS2/'
        t1_dir = 'testT/'

        for iid in test_id:
            count = 0
            print(iid)
            if iid < 10:
                subject_dir = os.path.join(data_dir, 'DIXCTWS00' + str(iid))
            elif 9 < iid < 100:
                subject_dir = os.path.join(data_dir, 'DIXCTWS0' + str(iid))
            else:
                subject_dir = os.path.join(data_dir, 'DIXCTWS' + str(iid))
                if not os.path.exists(str(iid)):
                    os.mkdir(str(iid))

            t2_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subject_dir, 'original/T2_b.nii.gz')))
            t1_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(subject_dir, 'original/T1_b.nii.gz')))


            #print(np.shape(t2_array))
            spacing = sitk.ReadImage(os.path.join(subject_dir, 'original/T2_b.nii.gz')).GetSpacing()
            origin = sitk.ReadImage(os.path.join(subject_dir, 'original/T2_b.nii.gz')).GetOrigin()
            direction = sitk.ReadImage(os.path.join(subject_dir, 'original/T1_b.nii.gz')).GetDirection()

            print(spacing)
            print(origin)
            print(direction)
            if np.shape(t2_array)[1] == np.shape(t1_array)[0]:
                if np.shape(t2_array)[1:4] != np.shape(t1_array):
                    t1_array_new = np.zeros(np.shape(t2_array)[1:4], dtype='float32')
                    for ss in range(np.shape(t1_array)[0]):
                        new_array = np.array(Image.fromarray(t1_array[ss]).resize(np.shape(t2_array)[2:4], Image.ANTIALIAS))
                        t1_array_new[ss, ...] = new_array
                    t1_array = t1_array_new

                mask = t2_array[0, 0, ...]
                mask[mask > 0] = 1
                # Find the bounding box of those pixels
                coords = np.array(np.nonzero(mask))
                top_left = np.min(coords, axis=1)
                bottom_right = np.max(coords, axis=1)

                out = np.float32(t2_array[:, :, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
                t2_ch2 = out[1, ...]
                t2_ch2 = self.denoise_norl(t2_ch2)
                #print(np.shape(t2_ch1))
                t1 = np.float32(t1_array[:, top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]])
                
                test_T_2ch = np.concatenate((t2_ch2[..., np.newaxis], t1[..., np.newaxis]), axis=3)
                pred_3D = []
                for ss in range(np.shape(test_T_2ch)[0]):
                    print(ss)
                    pred = extract_patches(self.G_A2B, test_T_2ch[ss], intl=160)
                    pred_ = pred[..., 0]
                    pred_ = (pred_ + 1)*127.5
                    #img_pred = ndimage.median_filter(pred_, size=3)
                    img_pred = Image.fromarray(np.uint8(pred_))
                    img_pred.save(os.path.join(str(iid), str(ss) + '.png'))
                    pred_3D.append(pred_)
                print(np.shape(np.asarray(pred_3D)))
                pred_3D_orig = self.padding(np.asarray(pred_3D), t2_array[0])
                pred_3D_orig_i = sitk.GetImageFromArray(pred_3D_orig)
                pred_3D_orig_i.SetSpacing(spacing[0:3])
                pred_3D_orig_i.SetOrigin(origin[0:3])
                pred_3D_orig_i.SetDirection(direction)
                sitk.WriteImage(pred_3D_orig_i, str(iid)+'_fluid.nii.gz')




                # print(np.shape(patches_1))
                # for pp in range(np.shape(patches_1)[0]):
                #     patch_1 = patches_1[pp]
                #     patch_2 = patches_2[pp]
                #     patch_3 = patches_3[pp]
                #     # patch_t1 = patches_t1[pp]
                #
                #     img_1 = Image.fromarray(np.uint8(patch_1))
                #     img_1.save(os.path.join(str(iid), t2_ch1_dir, str(count) + '.png'))
                #     img_2 = Image.fromarray(np.uint8(patch_2))
                #     img_2.save(os.path.join(str(iid), t2_ch2_dir, str(count) + '.png'))
                #     img_3 = Image.fromarray(np.uint8(patch_3))
                #     img_3.save(os.path.join(str(iid), t2_ch3_dir, str(count) + '.png'))
                #     count += 1


        # synthetic_images_T = self.G_A2B.predict(self.S_test)
        # synthetic_images_S = self.G_B2A.predict(self.T_test)
        # print(synthetic_images_T.shape, synthetic_images_S.shape)
        #

#
#
#         def scale_image_to_value_range(image, cmin=0, cmax=255):
#             return np.interp(image, (-1, 1), (cmin, cmax))
#
#         def save_image(image, name, domain):
#             if self.channels == 1:
#                 image = image[:, :, 0]
#             image_ = Image.fromarray(np.uint8(image))
#             image_.save(os.path.join('generate_images/', domain, name))
#
#         # Test S images
# #        for i in range(synthetic_images_S.shape[0]):
#             # Get the name from the image it was conditioned on
# #            name = self.testT1_image_names[i].strip('.npy') + '_synthetic.npy'
# #            synt_S1 = scale_image_to_value_range(synthetic_images_S[i, :, :, 0])
# #            synt_S2 = scale_image_to_value_range(synthetic_images_S[i, :, :, 1])
# #            save_image(synt_S1, name, 'S1')
# #            save_image(synt_S2, name, 'S2')
#
#         # Test T images
#         for i in range(len(synthetic_images_T)):
#             # Get the name from the image it was conditioned on
#             name = self.testS1_image_names[i].strip('.png') + '_synthetic.png'
#             synt_T1 = scale_image_to_value_range(synthetic_images_T[i, :, :, 0])
#  #           synt_T2 = scale_image_to_value_range(synthetic_images_T[i, :, :, 1])
#             save_image(synt_T1, name, 'testT')
#  #           save_image(synt_T2, name, 'T2')
#
#         print('{} synthetic images for each domain have been generated and placed in /generated_images/'
#               .format(synthetic_images_S.shape[0]))


# reflection padding taken from
# https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
class ReflectionPadding2D(KL.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [KL.InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'padding': self.padding
        })
        return config


class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            if len(image.shape) == 3:
                image = image[np.newaxis, :, :, :]

            if self.num_imgs < self.pool_size:  # fill up the image pool
                self.num_imgs = self.num_imgs + 1
                if len(self.images) == 0:
                    self.images = image
                else:
                    self.images = np.vstack((self.images, image))

                if len(return_images) == 0:
                    return_images = image
                else:
                    return_images = np.vstack((return_images, image))

            else:  # 50% chance that we replace an old synthetic image
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id, :, :, :]
                    tmp = tmp[np.newaxis, :, :, :]
                    self.images[random_id, :, :, :] = image[0, :, :, :]
                    if len(return_images) == 0:
                        return_images = tmp
                    else:
                        return_images = np.vstack((return_images, tmp))
                else:
                    if len(return_images) == 0:
                        return_images = image
                    else:
                        return_images = np.vstack((return_images, image))

        return return_images


if __name__ == '__main__':
    import argparse

    #parser = argparse.ArgumentParser()
    #parser.add_argument("-m", "--mode", help="'train' or 'test' mode or 'all' for both")
    #args = parser.parse_args()
    GAN = CycleGAN(mode='test')
