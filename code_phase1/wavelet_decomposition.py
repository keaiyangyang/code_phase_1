"""
Wavelet Decomposition Module
"""

import numpy as np
import pywt


def hybrid_wavelet_decomposition(signal, wavelet='db4', level=3, mode='symmetric'):
    """Hybrid wavelet decomposition"""
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode=mode)

    zero_coeffs = [np.zeros_like(c) for c in coeffs]
    components = []

    for i in range(len(coeffs)):
        temp_coeffs = zero_coeffs.copy()
        temp_coeffs[i] = coeffs[i]
        component = pywt.waverec(temp_coeffs, wavelet, mode=mode)

        if len(component) > len(signal):
            component = component[:len(signal)]
        elif len(component) < len(signal):
            component = np.pad(component, (0, len(signal) - len(component)))

        components.append(component)

    component_names = [f'A{level}'] + [f'D{i}' for i in range(level, 0, -1)]

    total_recon = sum(components)
    reconstruction_error = np.max(np.abs(signal - total_recon))

    return components, component_names, coeffs, reconstruction_error