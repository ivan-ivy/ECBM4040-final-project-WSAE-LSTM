import tensorflow as tf


class MeanAbsolutePercentageError(tf.keras.metrics.Metric):
    def __init__(self, name='mape'):
        super().__init__(name=name)
        self.mape = self.add_weight(name="mape", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.clip_by_value(y_true, clip_value_min=1e-3, clip_value_max=10)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=1e-6, clip_value_max=10)
        values = tf.abs((y_true - y_pred) / y_true)
        self.mape.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.shape(y_true)[0])

    def result(self):
        return self.mape / tf.cast(self.count, tf.float32)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.mape.assign(0.)
        self.count.assign(0)


class TheilU(tf.keras.metrics.Metric):
    def __init__(self, name='theil_u'):
        super().__init__(name=name)
        self.mse = self.add_weight(name="mse", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.ms_true = self.add_weight(name="ms_true", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.ms_pred = self.add_weight(name="ms_pred", dtype=tf.float32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mse.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.ms_true.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.ms_pred.assign_add(tf.reduce_sum(tf.square(y_pred)))

    def result(self):
        return tf.sqrt(self.mse) / (tf.sqrt(self.ms_true) + tf.sqrt(self.ms_pred))

    def reset_states(self):
        self.mse.assign(0.)
        self.ms_true.assign(0.)
        self.ms_pred.assign(0.)


class LinearCorrelation(tf.keras.metrics.Metric):
    def __init__(self, name='r'):
        super().__init__(name=name)
        self.sum_prod = self.add_weight(name="sum_prod", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.sum_true = self.add_weight(name="sum_true", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.sum_pred = self.add_weight(name="sum_pred", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.squre_true = self.add_weight(name="squre_true", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.squre_pred = self.add_weight(name="squre_pred", dtype=tf.float32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.sum_prod.assign_add(tf.reduce_sum(tf.multiply(y_true, y_pred)))
        self.sum_true.assign_add(tf.reduce_sum(y_true))
        self.sum_pred.assign_add(tf.reduce_sum(y_pred))
        self.squre_true.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.squre_pred.assign_add(tf.reduce_sum(tf.square(y_pred)))
        self.count.assign_add(tf.shape(y_true)[0])

    def result(self):
        return (tf.cast(self.count, tf.float32) * self.sum_prod - self.sum_true * self.sum_pred) / (
                tf.sqrt(tf.cast(self.count, tf.float32) * self.squre_true - tf.square(self.sum_true)) *
                tf.sqrt(tf.cast(self.count, tf.float32) * self.squre_pred - tf.square(self.sum_pred)))

    def reset_states(self):
        self.sum_prod.assign(0.)
        self.sum_true.assign(0.)
        self.sum_pred.assign(0.)
        self.squre_true.assign(0.)
        self.squre_pred.assign(0.)
        self.count.assign(0)
