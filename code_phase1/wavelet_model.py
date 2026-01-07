"""
MMCD Model
Using Differentiable Trend Loss (default parameters: lambda_trend=1.7, tau=3.2)
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU, Dropout, Dense, Bidirectional,
    Conv1D, ZeroPadding1D, MaxPooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_self_attention import SeqSelfAttention

from loss_functions import DifferentiableTrendLoss


class WaveletModel:
    """MMCD Model"""

    def __init__(self, look_back=3):
        self.look_back = look_back
        self.models = {}

    def build_model(self, input_shape):
        """Build model architecture"""
        model = Sequential()

        # Bidirectional GRU layer
        model.add(Bidirectional(
            GRU(64, return_sequences=True, kernel_regularizer=l2(0.1)),
            input_shape=input_shape
        ))

        # Standard GRU layer
        model.add(GRU(64, return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(Dropout(0.2))

        # Convolutional layer
        model.add(ZeroPadding1D(padding=2))
        model.add(Conv1D(32, 3, dilation_rate=2, activation='relu'))

        # Self-attention layer
        model.add(SeqSelfAttention(
            attention_activation='sigmoid',
            attention_width=5,
            kernel_regularizer=l2(0.01)
        ))

        # Max pooling layer
        model.add(MaxPooling1D(pool_size=2))

        # GRU layer
        model.add(GRU(32, return_sequences=False))

        # Fully connected layer
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))

        # Use Differentiable Trend Loss (default parameters)
        loss_fn = DifferentiableTrendLoss(lambda_trend=1.7, tau=3.2)

        # Compile model
        optimizer = Adam(learning_rate=0.0005)
        model.compile(loss=loss_fn, optimizer=optimizer)

        return model

    def train_single_component(self, component_name, trainX, trainY, epochs=100):
        """Train single component model"""
        model = self.build_model(trainX.shape[1:])

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                min_delta=0.0001
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                min_delta=0.0001
            )
        ]

        history = model.fit(
            trainX, trainY,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
            shuffle=False,
            callbacks=callbacks
        )

        self.models[component_name] = model

        return history.history

    def predict_single_component(self, component_name, testX):
        """Predict single component"""
        if component_name in self.models:
            return self.models[component_name].predict(testX, verbose=0).flatten()
        return None