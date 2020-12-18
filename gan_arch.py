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


def build_discriminator_ref(weight_init, **kwargs):
    f = [2**i for i in range(4)]
    IMG_H = kwargs['IMG_H']
    IMG_W = kwargs['IMG_W']
    IMG_C = kwargs['IMG_C']
    image_input = tf.keras.layers.Input(shape=(IMG_H, IMG_W, IMG_C))
    x = image_input
    filters = 64
    output_strides = 16
    h_output = IMG_H // output_strides
    w_output = IMG_W // output_strides

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