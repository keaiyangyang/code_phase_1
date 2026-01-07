# MMCD

## Important Note

**This repository contains a demonstration version of the core algorithms only.**  
The complete experimental framework, including full implementation of repeated experiments, ablation studies, parallel computing, and complete experimental results, will be made publicly available **after the paper is accepted for publication**.

## Core Algorithms
1. Differentiable Trend Loss Function (λ=1.7, τ=3.2) - Novel loss function that incorporates trend information through differentiable tanh functions
2. Wavelet Decomposition Preprocessing (db4, level=3) - Hybrid wavelet decomposition and reconstruction
3. BiGRU-DC-Attention-FPN Hybrid Architecture - Combines wavelet decomposition with bidirectional GRU, attention, and dilated convolution
4. Rolling Time Window Validation - Three expanding windows to evaluate model stability

## File Structure
- `loss_functions.py` - Differentiable Trend Loss Function (core innovation)
- `wavelet_decomposition.py` - Wavelet decomposition and reconstruction
- `wavelet_preprocessor.py` - Data preprocessing with wavelet decomposition
- `wavelet_model.py` - Wavelet-GRU-Attention model architecture
- `evaluation_metrics.py` - Evaluation metrics including directional and trend similarity metrics
- `rolling_window.py` - Rolling time window mechanism for validation
- `run_demo.py` - Demonstration program
- `requirements.txt` - Dependencies

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run the demonstration
python run_demo.py