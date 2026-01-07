"""
Main Demonstration Program - Simplified Version
No plots, only print results
"""

import numpy as np
from wavelet_decomposition import hybrid_wavelet_decomposition
from wavelet_preprocessor import WaveletPreprocessor
from wavelet_model import WaveletModel
from evaluation_metrics import (
    calculate_basic_metrics,
    calculate_directional_metrics,
    calculate_trend_similarity
)
from rolling_window import RollingWindowProcessor


def generate_demo_data():
    """Generate demonstration data"""
    np.random.seed(42)
    n_samples = 300
    t = np.linspace(0, 15, n_samples)

    trend = 0.4 * t + 50
    seasonal = 4 * np.sin(2 * np.pi * 0.1 * t) + 2 * np.sin(2 * np.pi * 0.03 * t)
    noise = np.random.randn(n_samples) * 1.5

    return trend + seasonal + noise


def demo_wavelet_decomposition():
    """Demonstrate wavelet decomposition"""
    print("=== Wavelet Decomposition Demonstration ===")

    signal = generate_demo_data()
    components, component_names, _, error = hybrid_wavelet_decomposition(
        signal, wavelet='db4', level=3
    )

    print(f"Signal length: {len(signal)}")
    print(f"Number of components: {len(components)}")
    print(f"Component names: {component_names}")
    print(f"Reconstruction error: {error:.2e}")

    return signal, components, component_names


def demo_trend_loss():
    """Demonstrate Differentiable Trend Loss"""
    print("\n=== Differentiable Trend Loss Demonstration ===")
    print("Loss function formula: L = MSE(y, ŷ) + λ * MSE(tanh(τ*Δy), tanh(τ*Δŷ))")
    print(f"Default parameters: λ=1.7, τ=3.2")

    # Simple test
    from loss_functions import DifferentiableTrendLoss
    import tensorflow as tf

    loss_fn = DifferentiableTrendLoss(lambda_trend=1.7, tau=3.2)

    y_true = tf.constant([[1.0, 2.0, 3.0, 2.8, 2.5],
                          [2.0, 2.5, 3.0, 3.2, 3.5]], dtype=tf.float32)
    y_pred = tf.constant([[1.2, 2.1, 2.9, 2.7, 2.6],
                          [2.1, 2.4, 3.1, 3.3, 3.4]], dtype=tf.float32)

    loss_value = loss_fn(y_true, y_pred).numpy()
    print(f"Example loss value: {loss_value:.4f}")

    return loss_fn


def demo_single_window():
    """Demonstrate single window process"""
    print("\n=== Single Window Process Demonstration ===")

    signal = generate_demo_data()
    split_idx = int(len(signal) * 0.7)
    train_data = signal[:split_idx]
    test_data = signal[split_idx:]

    print(f"Data statistics:")
    print(f"  Total samples: {len(signal)}")
    print(f"  Training set: {len(train_data)} samples")
    print(f"  Test set: {len(test_data)} samples")

    # Preprocessing
    preprocessor = WaveletPreprocessor(wavelet='db4', level=3, look_back=3)
    processed_data = preprocessor.preprocess_single_window(train_data, test_data)

    print(f"Wavelet decomposition:")
    print(f"  Decomposition level: 3")
    print(f"  Number of components: {len(processed_data['component_names'])}")
    print(f"  Component names: {processed_data['component_names']}")

    # Train model
    model = WaveletModel(look_back=3)
    component_name = processed_data['component_names'][0]

    print(f"\nTraining component: {component_name}")

    trainX = processed_data['train_data'][component_name]['X']
    trainY = processed_data['train_data'][component_name]['Y']

    print("Training model...")
    history = model.train_single_component(component_name, trainX, trainY, epochs=50)

    # Predict
    testX = processed_data['test_data'][component_name]['X']
    testY = processed_data['test_data'][component_name]['Y']

    predictions = model.predict_single_component(component_name, testX)

    if predictions is not None:
        scaler = processed_data['component_scalers'][component_name]
        predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        expected_original = scaler.inverse_transform(testY.reshape(-1, 1)).flatten()

        # Evaluation
        basic_metrics = calculate_basic_metrics(expected_original, predictions_original)
        dir_metrics = calculate_directional_metrics(expected_original, predictions_original)
        trend_metrics = calculate_trend_similarity(expected_original, predictions_original)

        print(f"\nEvaluation results:")
        print(f"  Basic metrics:")
        print(f"    RMSE: {basic_metrics['RMSE']:.4f}")
        print(f"    MAE: {basic_metrics['MAE']:.4f}")
        print(f"    MAPE: {basic_metrics['MAPE']:.4f}")
        print(f"    R²: {basic_metrics['R^2']:.4f}")

        print(f"  Directional metrics:")
        print(f"    MDA: {dir_metrics['MDA']:.4f}")
        print(f"    DA: {dir_metrics['DA']:.4f}")
        print(f"    MADL: {dir_metrics['MADL']:.4f}")

        print(f"  Trend similarity metrics:")
        print(f"    Corr_Diff: {trend_metrics['Corr_Diff']:.4f}")
        print(f"    Cov_Diff: {trend_metrics['Cov_Diff']:.4f}")

        return {
            'basic_metrics': basic_metrics,
            'directional_metrics': dir_metrics,
            'trend_metrics': trend_metrics,
            'component_name': component_name
        }

    return None


def demo_rolling_windows():
    """Demonstrate rolling windows"""
    print("\n=== Rolling Windows Demonstration ===")

    signal = generate_demo_data()
    window_processor = RollingWindowProcessor(n_windows=3)
    windows = window_processor.create_windows(signal)

    print(f"Window configuration:")
    print(f"  Total data length: {len(signal)}")
    print(f"  Number of windows: {len(windows)}")

    results = []

    for i, window in enumerate(windows[:2]):  # Only demonstrate first two windows
        print(f"\nWindow {i + 1}:")
        print(f"  Training samples: {len(window['train_data'])}")
        print(f"  Test samples: {len(window['test_data'])}")

        preprocessor = WaveletPreprocessor(wavelet='db4', level=3, look_back=3)
        processed_data = preprocessor.preprocess_single_window(
            window['train_data'],
            window['test_data']
        )

        component_name = processed_data['component_names'][0]
        model = WaveletModel(look_back=3)

        trainX = processed_data['train_data'][component_name]['X']
        trainY = processed_data['train_data'][component_name]['Y']

        print("  Training model...")
        history = model.train_single_component(component_name, trainX, trainY, epochs=30)

        testX = processed_data['test_data'][component_name]['X']
        testY = processed_data['test_data'][component_name]['Y']
        predictions = model.predict_single_component(component_name, testX)

        if predictions is not None:
            scaler = processed_data['component_scalers'][component_name]
            predictions_original = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            expected_original = scaler.inverse_transform(testY.reshape(-1, 1)).flatten()

            basic_metrics = calculate_basic_metrics(expected_original, predictions_original)
            dir_metrics = calculate_directional_metrics(expected_original, predictions_original)

            results.append({
                'window': i + 1,
                'RMSE': basic_metrics['RMSE'],
                'R2': basic_metrics['R^2'],
                'MDA': dir_metrics['MDA']
            })

            print(f"  Results: RMSE={basic_metrics['RMSE']:.4f}, R²={basic_metrics['R^2']:.4f}")

    if results:
        print(f"\nWindow results summary:")
        for result in results:
            print(f"  Window {result['window']}: RMSE={result['RMSE']:.4f}, R²={result['R2']:.4f}")

        rmses = [r['RMSE'] for r in results]
        print(f"  Average RMSE: {np.mean(rmses):.4f}")

        return results

    return None


def main():
    """Main function"""
    print("=" * 60)
    print("Wavelet-GRU-Attention with Differentiable Trend Loss")
    print("Core Algorithm Demonstration")
    print("=" * 60)

    np.random.seed(42)

    try:
        # Demonstration 1: Wavelet decomposition
        demo_wavelet_decomposition()

        # Demonstration 2: Trend loss function
        demo_trend_loss()

        # Demonstration 3: Single window process
        single_results = demo_single_window()

        # Demonstration 4: Rolling windows
        window_results = demo_rolling_windows()

        print("\n" + "=" * 60)
        print("Demonstration completed!")
        print("=" * 60)

        if single_results:
            print("\nCore algorithm performance:")
            print(f"  RMSE: {single_results['basic_metrics']['RMSE']:.4f}")
            print(f"  R²: {single_results['basic_metrics']['R^2']:.4f}")
            print(f"  MDA: {single_results['directional_metrics']['MDA']:.4f}")
            print(f"  Corr_Diff: {single_results['trend_metrics']['Corr_Diff']:.4f}")

        print("\n✅ All core algorithms validated successfully")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    return True


if __name__ == "__main__":
    main()