import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

class NoisyLiner(tf.keras.layers.Layer):
    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLiner, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.w = self.add_weight(name="weight",
                                 shape=(input_shape[1], self.output_dim),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer="zeros",
                                 trainable=True)
        
    def call(self, inputs, training=False):
        if training:
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            noise = tf.random.normal(shape=(batch, dim), mean=0.0,
                                     stddev=self.noise_stddev)
            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs

        z = tf.matmul(noisy_inputs, self.w) + self.b
        return tf.keras.activations.relu(z)
    
    def get_config(self):
        config = super(NoisyLiner, self).get_config()
        config.update({"output_dim": self.output_dim,
                       "noise_stddev": self.noise_stddev})
        return config
    

# -----+-----+----- 学習用データの準備 -----+-----+-----
tf.random.set_seed(1)
np.random.seed(1)
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0
x_train, x_valid = x[:100, :], x[100:, :]
y_train, y_valid = y[:100], y[100:]

# -----+-----+----- モデルの定義 -----+-----+-----
tf.random.set_seed(1)
model = tf.keras.Sequential([
    NoisyLiner(4, noise_stddev=0.1),
    tf.keras.layers.Dense(units=4, activation="relu"),
    tf.keras.layers.Dense(units=4, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])
model.build(input_shape=(None, 2))
model.summary()

# -----+-----+----- コンパイル -----+-----+-----
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

# -----+-----+----- トレーニング -----+-----+-----
hist = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                 epochs=200, batch_size=2, verbose=0)

# -----+-----+----- プロット -----+-----+-----
history = hist.history

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history['loss'], lw=4)
plt.plot(history['val_loss'], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history['binary_accuracy'], lw=4)
plt.plot(history['val_binary_accuracy'], lw=4)
plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.savefig("output.png", format="png")
