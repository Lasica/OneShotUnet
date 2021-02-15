import tensorflow as tf

def deconv_block(inputs, num_filters, kernel_size, strides, weight_init, bn=True):
    x = tf.keras.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=weight_init,
        padding="same",
        strides=strides,
        use_bias=False
        )(inputs)

    if bn:
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


def conv_block(inputs, num_filters, kernel_size, weight_init, padding="same", strides=2, activation=True):
    x = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        kernel_initializer=weight_init,
        padding=padding,
        strides=strides,
    )(inputs)

    if activation:
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    return x


def build_generator_ref(latent_dim, weight_init, **kwargs):
    f = [2**i for i in range(5)][::-1]
    filters = 32
    output_strides = 16
    IMG_H = kwargs['IMG_H']
    IMG_W = kwargs['IMG_W']
    IMG_C = kwargs['IMG_C']
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

    noise = tf.keras.layers.Input(shape=(latent_dim,), name="generator_noise_input")

    x = tf.keras.layers.Dense(f[0] * filters * h_output * w_output, use_bias=False)(noise)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Reshape((h_output, w_output, 16 * filters))(x)

    for i in range(1, 5):
        x = deconv_block(x,
            num_filters=f[i] * filters,
            kernel_size=5,
            strides=2,
            weight_init=weight_init,
            bn=True
        )

    x = conv_block(x,
        num_filters=IMG_C,
        kernel_size=5,
        weight_init=weight_init,
        strides=1,
        activation=False
    )
    fake_output = tf.keras.layers.Activation("tanh")(x)

    return tf.keras.models.Model(noise, fake_output, name="generator")


## train separately?
def build_encoder_ref(latent_dim, weight_init, **kwargs):
    f = [2**i for i in range(4)]
    IMG_H = kwargs['IMG_H']
    IMG_W = kwargs['IMG_W']
    IMG_C = kwargs['IMG_C']
    image_input = tf.keras.layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, weight_init=weight_init, strides=2)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(latent_dim, use_bias=False)(x)

    return tf.keras.models.Model(image_input, x, name="encoder")


def build_discriminator_ref(weight_init, **kwargs):
    f = [2**i for i in range(4)]
    IMG_H = kwargs['IMG_H']
    IMG_W = kwargs['IMG_W']
    IMG_C = kwargs['IMG_C']
    image_input = tf.keras.layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64

    for i in range(0, 4):
        x = conv_block(x, num_filters=f[i] * filters, kernel_size=5, weight_init=weight_init, strides=2)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)

    return tf.keras.models.Model(image_input, x, name="discriminator")


class GAN_ref(tf.keras.models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN_ref, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN_ref, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        generated_labels = tf.zeros((batch_size, 1))
        labels = tf.ones((batch_size, 1))
        misleading_labels = tf.ones((batch_size, 1))

        for _ in range(1):
            ## Train the discriminator with real images

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images, training=True)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))
                
            ## Train the generator and discriminator with generated images
            
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            with tf.GradientTape(persistent=True) as ftape:
                generated_images = self.generator(random_latent_vectors, training=True)
                predictions = self.discriminator(generated_images, training=True)
                d1_loss = self.loss_fn(generated_labels, predictions)
                g_loss = self.loss_fn(misleading_labels, predictions)
                with ftape.stop_recording():
                    grads_d = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
                    grads_g = ftape.gradient(g_loss, self.generator.trainable_weights)
                    self.d_optimizer.apply_gradients(zip(grads_d, self.discriminator.trainable_weights))
                    self.g_optimizer.apply_gradients(zip(grads_g, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss, "total":d1_loss+d2_loss+g_loss}


class GAN_old(tf.keras.models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(GAN_old, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN_old, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss, "total":d1_loss+d2_loss+g_loss}


class GAN_autoencoder(tf.keras.models.Model):
    def __init__(self, discriminator, generator, encoder, latent_dim):
        super(GAN_autoencoder, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.encoder = encoder
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, e_optimizer, loss_fn, ae_loss_fn):
        super(GAN_autoencoder, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.e_optimizer = e_optimizer
        self.loss_fn = loss_fn
        self.ae_loss_fn = ae_loss_fn

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for _ in range(2):
            ## Train the discriminator
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            generated_images = self.generator(random_latent_vectors)
            generated_labels = tf.zeros((batch_size, 1))

            with tf.GradientTape() as ftape:
                predictions = self.discriminator(generated_images)
                d1_loss = self.loss_fn(generated_labels, predictions)
            grads = ftape.gradient(d1_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

            ## Train the discriminator
            labels = tf.ones((batch_size, 1))

            with tf.GradientTape() as rtape:
                predictions = self.discriminator(real_images)
                d2_loss = self.loss_fn(labels, predictions)
            grads = rtape.gradient(d2_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        ## Train the generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as gtape:
            gen_samples = self.generator(random_latent_vectors)
            predictions = self.discriminator(gen_samples)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = gtape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        ## Train the encoder
        with tf.GradientTape() as etape:
            encoding_prediction = self.encoder(gen_samples)
            e_loss = self.ae_loss_fn(random_latent_vectors, encoding_prediction)
        grads = etape.gradient(e_loss, self.encoder.trainable_weights)
        self.e_optimizer.apply_gradients(zip(grads, self.encoder.trainable_weights))

        ## Add step for full stack autoencoding?

        return {"d1_loss": d1_loss, "d2_loss": d2_loss, "g_loss": g_loss, "e_loss":e_loss, "total":d1_loss+d2_loss+g_loss}
    


### Cycle gan defs
def downsample(filters, size, apply_batchnorm=True, initializer = tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, dropout=0.0, initializer = tf.random_normal_initializer(0., 0.02)):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if dropout:
        result.add(tf.keras.layers.Dropout(max(min(0.9,dropout), 0.1)))

    result.add(tf.keras.layers.ReLU())

    return result


def generator_layers(input, outputs_shape, first_layer_filters):
    W, H, C = outputs_shape

    powers = [2**i for i in range(4)]
    first_layer_filters = 32
    kernel_size = 3 

    initializer = tf.random_normal_initializer(0., 0.02)

    down_stack = [
        downsample(first_layer_filters*2, kernel_size, apply_batchnorm=False), # (bs, 16, 16, 64)
        downsample(first_layer_filters*4, kernel_size), # (bs, 8, 8, 128)
        downsample(first_layer_filters*8, kernel_size), # (bs, 4, 4, 256)
        downsample(first_layer_filters*16, kernel_size), # (bs, 2, 2, 512)
        downsample(first_layer_filters*16, kernel_size), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(first_layer_filters*16, kernel_size, dropout=0.5), # (bs, 2, 2, 1024)
        upsample(first_layer_filters*8, kernel_size, dropout=0.2), # (bs, 4, 4, 512)
        upsample(first_layer_filters*4, kernel_size), # (bs, 8, 8, 256)
        upsample(first_layer_filters*2, kernel_size), # (bs, 16, 16, 128)
    ]

    last = tf.keras.layers.Conv2DTranspose(C, kernel_size,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh') # (bs, 32, 32, 1)

    x = input

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return x


def cycle_generator_xy(input, outputs_shape, first_layer_filters):
    # input shape: (latent_dim)
    W, H, C = outputs_shape
    
    x = tf.keras.layers.Dense(first_layer_filters * W * H, use_bias=False)(input) 
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Reshape((W, H, first_layer_filters))(x)

    x = generator_layers(x, outputs_shape, first_layer_filters)
    return tf.keras.models.Model(inputs=input, outputs=x)


def cycle_generator_yx(input, outputs_shape, first_layer_filters):
    # input shape: (W, H, C)
    W, H, C = input.shape

    x = generator_layers(input, input.shape, first_layer_filters)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.Dense(outputs_shape[0], use_bias=False)(x) 

    return tf.keras.models.Model(inputs=input, outputs=x)


def patch_discriminator(norm_type='batchnorm'):
    """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
    Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
    Returns:
    Discriminator model
    """

    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[None, None, 3], name='input_image')
    x = inp
    kernel_size = 3

    down1 = downsample(32, kernel_size, norm_type, False)(x)  # (bs, 32, 32, 32)
    down2 = downsample(64, kernel_size, norm_type)(down1)  # (bs, 16, 16, 64)
    down3 = downsample(128, kernel_size, norm_type)(down2)  # (bs, 8, 8, 128)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 10, 10, 256)
    conv = tf.keras.layers.Conv2D(
        256, kernel_size, strides=1, kernel_initializer=initializer,
        use_bias=False)(zero_pad1)  # (bs, 8, 8, 256)

    if norm_type.lower() == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type.lower() == 'instancenorm':
        norm1 = tf.keras.layers.InstanceNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 10, 10, 256)

    last = tf.keras.layers.Conv2D(
        1, kernel_size, strides=1,
        kernel_initializer=initializer)(zero_pad2)  # (bs, 8, 8, 1) 

    return tf.keras.Model(inputs=inp, outputs=last)


def latent_discriminator():
    ...