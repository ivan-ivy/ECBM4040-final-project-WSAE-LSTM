import numpy as np
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, intermidiate_dim=10, name='encoder', sparsity_weight=0, sparsity_target=0.5, l2_reg=1e-3,
                 **kwargs):
        """
        Args:
            intermidiate_dim: number of hidden units
            name: encoder
            sparsity_weight: weight of sparsity penalty
            sparsity_target: target level of activation
            l2_reg: weight of l2 regularization
            **kwargs: pass additional arguments to keras.layers.Layer
        """
        super().__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(units=intermidiate_dim, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight

    def call(self, inputs):
        """
        Args:
            inputs: inputs data for training
        """
        h = self.dense(inputs)
        self.add_loss(self.sparsity_loss(h))
        return h

    def sparsity_loss(self, h):
        """
        Args:
            h: inputs
        """
        mean_activation = tf.reduce_mean(h, axis=0)
        return self.sparsity_weight * (tf.keras.losses.KLD(self.sparsity_target, mean_activation) +
                                       tf.keras.losses.KLD(1 - self.sparsity_target, 1 - mean_activation))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, original_dim, name='decoder', l2_reg=1e-3, **kwargs):
        """
        Args:
            original_dim: dimension of original data
            name: name
            l2_reg: weight of l2 regularization
            **kwargs:
        """
        super().__init__(name=name, **kwargs)
        self.outputs = tf.keras.layers.Dense(units=original_dim, activation='sigmoid',
                                             kernel_regularizer=tf.keras.regularizers.l2(l=l2_reg))

    def call(self, inputs):
        # x = self.dense(inputs)
        """
        Args:
            inputs: inputs
        """
        return self.outputs(inputs)


class AutoEncoder(tf.keras.Model):
    def __init__(self, original_dim, intermidiate_dim, l2_reg=1e-3, sparsity_weight=1e-2, sparsity_target=0.5):
        """
        Args:
            original_dim: dimension of inputs data
            intermidiate_dim: number of hidden units
            l2_reg: weight of l2 regularization
            sparsity_weight: weight of sparsity penalty
            sparsity_target: target level of activation
        """
        super().__init__()
        self.encoder = Encoder(intermidiate_dim=intermidiate_dim,
                               sparsity_weight=sparsity_weight,
                               sparsity_target=sparsity_target,
                               l2_reg=l2_reg)

        self.decoder = Decoder(original_dim=original_dim,
                               intermidiate_dim=intermidiate_dim,
                               l2_reg=l2_reg)

    def call(self, inputs):
        """
        Args:
            inputs: inputs data
        """
        x = self.encoder(inputs)
        return self.decoder(x)


class StackedAutoEncoder:
    def __init__(self, layers, original_dim, intermidiate_dim, l2_reg=1e-3, sparsity_weight=1e-2, sparsity_target=0.5):
        """
        Args:
            layers: number of layers
            original_dim: dimension of inputs data
            intermidiate_dim: number of hidden units
            l2_reg:  weight of l2 regularization
            sparsity_weight: weight of sparsity penalty
            sparsity_target: target level of activation
        """
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
        """
        Args:
            model: single autoencoder instance
            inputs: inputs data
            learning_rate: learning rate
            n_epochs: number of epochs
            batch_size: batch size
        """
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

    def train_stacked_ae(self,
                         inputs,
                         learning_rate=0.01,
                         n_epochs=20,
                         batch_size=128
                         ):
        """
        Args:
            inputs: inputs data
            learning_rate: learning rate
            n_epochs: number of epochs
            batch_size: batch size
        """
        self.train_one_ae(self.ae[0], inputs, learning_rate, n_epochs, batch_size)
        h = inputs
        for i in range(1, len(self.ae)):
            h = self.ae[i - 1].encoder(h).numpy()
            self.train_one_ae(self.ae[i], h, learning_rate, n_epochs, batch_size)

    def encode(self, x_input):
        """
        Args:
            x_input: inputs data
        """
        h = x_input
        for ae in self.ae:
            h = ae.encoder(h).numpy()
        return h
