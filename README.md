# iTransformer: Rust Implementation

An **iTransformer** implementation in **Rust**, inspired by the [lucidrains iTransformer repository](https://github.com/lucidrains/iTransformer), and based on the original research and implementation from [Tsinghua University's iTransformer repository](https://github.com/thuml/iTransformer).

## 📚 **What is iTransformer?**

iTransformer introduces an **inverted Transformer architecture** designed for **multivariate time series forecasting (MTSF)**. By reversing the conventional structure of Transformers, iTransformer achieves state-of-the-art results in handling complex multivariate time series data.

### 🚀 **Key Features:**
- **Inverted Transformer Architecture:** Captures multivariate correlations efficiently.
- **Layer Normalization & Feed-Forward Networks:** Optimized for time series representation.
- **Flexible Prediction Lengths:** Supports predictions at multiple horizons (e.g., 12, 24, 36, 48 steps ahead).
- **Scalability:** Handles hundreds of variates with efficiency.
- **Zero-Shot Generalization:** Train on partial variates and generalize to unseen variates.
- **Efficient Attention Mechanisms:** Compatible with advanced techniques like FlashAttention.

### 🛠️ **Architecture Overview**

iTransformer treats each time series variate as a **token**, applying attention mechanisms across variates, followed by feed-forward networks and normalization layers.

![Architecture](https://raw.githubusercontent.com/thuml/iTransformer/main/figures/architecture.png)

### 📊 **Key Benefits:**
- **State-of-the-Art Performance:** On benchmarks such as Traffic, Weather, and Electricity datasets.
- **Improved Interpretability:** Multivariate self-attention reveals meaningful correlations.
- **Scalable and Efficient Training:** Can accommodate long time series without performance degradation.

### 📈 **Performance Highlights:**

iTransformer consistently outperforms other Transformer-based architectures in multivariate time series forecasting benchmarks.

![Results](https://raw.githubusercontent.com/thuml/iTransformer/main/figures/main_results_alipay.png)

## 📥 **Installation**

To get started, ensure you have Rust and Cargo installed. Then:

```bash
# Add iTransformer to your project dependencies
soon...
```

## 📝 **Usage**

```bash
# Add iTransformer to your project dependencies
soon...
```


## 📖 **References**

- **Paper:** [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)
- **Original Implementation:** [thuml/iTransformer](https://github.com/thuml/iTransformer)
- **Inspired by:** [lucidrains/iTransformer](https://github.com/lucidrains/iTransformer)

## 🏆 **Acknowledgments**

This work draws inspiration and insights from the following projects:
- [Reformer](https://github.com/lucidrains/reformer-pytorch)
- [Informer](https://github.com/zhouhaoyi/Informer2020)
- [FlashAttention](https://github.com/shreyansh26/FlashAttention-PyTorch)
- [Autoformer](https://github.com/thuml/Autoformer)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

Special thanks to the contributors and researchers behind iTransformer for their pioneering work.

## 📑 **Citation**

```bibtex
@article{liu2023itransformer,
  title={iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  author={Liu, Yong and Hu, Tengge and Zhang, Haoran and Wu, Haixu and Wang, Shiyu and Ma, Lintao and Long, Mingsheng},
  journal={arXiv preprint arXiv:2310.06625},
  year={2023}
}
```

## 🌟 **Contributing**

Contributions are welcome! Please follow the standard GitHub pull request process and ensure your code adheres to Rust best practices.

---

This repository is a Rust adaptation of the cutting-edge iTransformer model, aiming to bring efficient and scalable time series forecasting capabilities to the Rust ecosystem.

