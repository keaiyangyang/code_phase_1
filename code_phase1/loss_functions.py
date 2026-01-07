"""
Differentiable Trend Loss Function - Core Innovation
Default parameters: lambda_trend=1.7, tau=3.2
"""

import tensorflow as tf
from tensorflow.keras.losses import Loss


class DifferentiableTrendLoss(Loss):
    """
    Differentiable Trend Loss Function

    Loss function formula:
    L = MSE(y, ŷ) + λ * MSE(tanh(τ*Δy), tanh(τ*Δŷ))

    Default parameters (optimal values from paper):
    λ = 1.7, τ = 3.2
    """

    def __init__(self, lambda_trend=1.7, tau=3.2, name="trend_loss"):
        super().__init__(name=name)
        self.lambda_trend = lambda_trend
        self.tau = tau

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Base MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Trend loss (if sequence length is sufficient)
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true_current = y_true[:, 1:]
            y_true_prev = y_true[:, :-1]
            y_pred_current = y_pred[:, 1:]
            y_pred_prev = y_pred[:, :-1]

            true_trend = tf.tanh(self.tau * (y_true_current - y_true_prev))
            pred_trend = tf.tanh(self.tau * (y_pred_current - y_pred_prev))

            trend_loss = tf.reduce_mean(tf.square(true_trend - pred_trend))
        else:
            trend_loss = 0.0

        return mse_loss + self.lambda_trend * trend_loss

    def get_config(self):
        return {
            "lambda_trend": self.lambda_trend,
            "tau": self.tau,
            "name": self.name
        }