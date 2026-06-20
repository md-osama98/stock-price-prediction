import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)

from tensorflow.keras.models import Model


# ---------------------------------------
# Custom Attention Layer
# ---------------------------------------

class CustomAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(CustomAttention, self).__init__()

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.nn.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        weights = tf.nn.softmax(
            score,
            axis=1
        )

        context = tf.reduce_sum(
            weights * inputs,
            axis=1
        )

        return context


# ---------------------------------------
# Build Model
# ---------------------------------------

def build_model(input_shape):

    inputs = Input(shape=input_shape)

    # First Bi-LSTM
    x = Bidirectional(
        LSTM(
            128,
            return_sequences=True
        )
    )(inputs)

    x = Dropout(0.3)(x)

    # Second Bi-LSTM
    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(x)

    x = Dropout(0.3)(x)

    # Attention
    x = CustomAttention()(x)

    # Dense Layers
    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dropout(0.2)(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    output = Dense(1)(x)

    model = Model(
        inputs=inputs,
        outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="mse",
        metrics=["mae"]
    )

    return model