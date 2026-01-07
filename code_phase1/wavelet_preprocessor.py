"""
Wavelet Preprocessor
"""

import numpy as np
from sklearn.preprocessing import QuantileTransformer
from wavelet_decomposition import hybrid_wavelet_decomposition


class WaveletPreprocessor:
    """Wavelet data preprocessor"""

    def __init__(self, wavelet='db4', level=3, look_back=3):
        self.wavelet = wavelet
        self.level = level
        self.look_back = look_back
        self.component_scalers = {}
        self.component_names = []

    def create_dataset(self, series):
        """Create supervised learning dataset"""
        dataX, dataY = [], []
        for i in range(len(series) - self.look_back):
            dataX.append(series[i:(i + self.look_back)])
            dataY.append(series[i + self.look_back])

        return np.array(dataX, dtype=np.float32), np.array(dataY, dtype=np.float32)

    def preprocess_single_window(self, train_series, test_series):
        """Preprocess a single window"""
        # Train set wavelet decomposition
        components_train, self.component_names, _, _ = hybrid_wavelet_decomposition(
            train_series, self.wavelet, self.level
        )

        # Component standardization
        component_scalers = {}
        components_train_scaled = {}

        for i, (comp, name) in enumerate(zip(components_train, self.component_names)):
            scaler = QuantileTransformer(output_distribution='normal')
            comp_scaled = scaler.fit_transform(comp.reshape(-1, 1)).flatten()
            component_scalers[name] = scaler
            components_train_scaled[name] = comp_scaled

        # Test set wavelet decomposition and standardization
        components_test, _, _, _ = hybrid_wavelet_decomposition(
            test_series, self.wavelet, self.level
        )

        components_test_scaled = {}
        for i, (comp, name) in enumerate(zip(components_test, self.component_names)):
            comp_scaled = component_scalers[name].transform(comp.reshape(-1, 1)).flatten()
            components_test_scaled[name] = comp_scaled

        # Prepare training and test data
        train_data = {}
        test_data = {}

        for name in self.component_names:
            # Training data
            trainX, trainY = self.create_dataset(components_train_scaled[name])
            trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
            train_data[name] = {'X': trainX, 'Y': trainY}

            # Test data
            testX, testY = self.prepare_test_inputs(
                components_train_scaled[name],
                components_test_scaled[name]
            )
            test_data[name] = {'X': testX, 'Y': testY}

        return {
            'train_data': train_data,
            'test_data': test_data,
            'component_names': self.component_names,
            'component_scalers': component_scalers
        }

    def prepare_test_inputs(self, train_scaled, test_scaled):
        """Prepare test inputs"""
        test_inputs = []

        for j in range(len(test_scaled)):
            if j < self.look_back:
                needed = self.look_back - j
                input_seq = np.concatenate([
                    train_scaled[-needed:],
                    test_scaled[:j]
                ])
            else:
                input_seq = test_scaled[j - self.look_back:j]

            test_inputs.append(input_seq)

        testX = np.array(test_inputs, dtype=np.float32).reshape(-1, self.look_back, 1)
        return testX, test_scaled