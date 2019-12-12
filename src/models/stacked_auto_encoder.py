import numpy as np
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermidiate_dim=10, name='encoder', sparsity_weight=0, sparsity_target=0.5, l2_reg=1e-3,
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(units=intermidiate_dim, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def call(self, inputs):
        h = self.dense(inputs)
        self.add_loss(self.sparsity_loss(h))
        return h

    def sparsity_loss(self, h):
        mean_activation = tf.reduce_mean(h, axis=0)
        return self.sparsity_weight * (tf.keras.losses.KLD(self.sparsity_target, mean_activation) +
                                       tf.keras.losses.KLD(1 - self.sparsity_target, 1 - mean_activation))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, original_dim, intermidiate_dim=10, name='decoder', l2_reg=1e-3, **kwargs):
        super().__init__(name=name, **kwargs)
        self.outputs = tf.keras.layers.Dense(units=original_dim, activation='sigmoid',
                                             kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))

    def call(self, inputs):
        # x = self.dense(inputs)
        return self.outputs(inputs)


class AutoEncoder(tf.keras.Model):
    def __init__(self, original_dim, intermidiate_dim, l2_reg=1e-3, sparsity_weight=1e-2, sparsity_target=0.5):
        super().__init__()
        self.encoder = Encoder(intermidiate_dim=intermidiate_dim,
                               sparsity_weight=sparsity_weight,
                               sparsity_target=sparsity_target,
                               l2_reg=l2_reg)

        self.decoder = Decoder(original_dim=original_dim,
                               intermidiate_dim=intermidiate_dim,
                               l2_reg=l2_reg)

    def call(self, inputs):
        x = self.encoder(inputs)
        return self.decoder(x)


class StackedAutoEncoder:
    def __init__(self, layers, original_dim, intermidiate_dim, l2_reg=1e-3, sparsity_weight=1e-2, sparsity_target=0.5):
        self.ae = [AutoEncoder(original_dim, intermidiate_dim, l2_reg, sparsity_weight, sparsity_target)]
        for k in range(layers - 1):
            self.ae.append(AutoEncoder(intermidiate_dim, intermidiate_dim, l2_reg, sparsity_weight, sparsity_target))

    def train_one_ae(self,
                     model,
                     inputs,
                     learning_rate=0.01,
                     n_epochs=20,
                     batch_size=128
                     ):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        num_train = inputs.shape[0]
        n_batches = int(num_train / batch_size)

        loss_lst = []
        for epoch in range(n_epochs):
            total_loss = 0
            for i in range(n_batches):
                sample_idxs = np.random.choice(num_train, batch_size)
                x_batch = inputs[sample_idxs, :]
                with tf.GradientTape() as tape:
                    x_rec = model(x_batch)
                    regularization_loss = tf.add_n(model.losses)
                    loss = 0.5 * tf.reduce_sum(tf.square(x_rec - x_batch)) + regularization_loss
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
                loss_lst.append(loss.numpy())
                total_loss += loss.numpy()
            # if (epoch % 5 == 0):
                # print('{}/{} Average loss per epoch: {}'.format(epoch + 1, n_epochs, total_loss / n_batches))
        # plt.plot(loss_lst)

    def train_stacked_ae(self,
                         inputs,
                         learning_rate=0.01,
                         n_epochs=20,
                         batch_size=128
                         ):
        # print(f"Start to train Layer 1.")
        self.train_one_ae(self.ae[0], inputs, learning_rate, n_epochs, batch_size)
        h = inputs
        for i in range(1, len(self.ae)):
            h = self.ae[i - 1].encoder(h).numpy()
            # print(f"Start to train Layer {i + 1}.")
            self.train_one_ae(self.ae[i], h, learning_rate, n_epochs, batch_size)
            # print(f">>>>Layer {i + 1} trained!<<<<")

    def encode(self, x_input):
        h = x_input
        for ae in self.ae:
            h = ae.encoder(h).numpy()
        return h
