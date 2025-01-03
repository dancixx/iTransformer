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
cargo add itransformer-rs
```

## 📝 **Usage**

```rust
use tch::{Device, Tensor, nn::VarStore, Kind};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let vs = VarStore::new(Device::Cpu);
    let model = ITransformer::new(
        &(vs.root() / "itransformer"),
        137, // num_variates
        96,  // lookback_len
        6,   // depth
        256, // dim
        Some(1),       // num_tokens_per_variate
        vec![12, 24, 36, 48], // pred_length
        Some(64),     // dim_head
        Some(8),      // heads
        None,         // attn_drop_p
        None,         // ff_mult
        None,         // ff_drop_p
        None,         // num_mem_tokens
        Some(true),   // use_reversible_instance_norm
        None,         // reversible_instance_norm_affine
        false,        // flash_attn
        &Device::Cpu,
    )?;
    let time_series = Tensor::randn([2, 96, 137], (Kind::Float, Device::Cpu));
    let preds = model.forward(&time_series, None, false);
    println!("{:?}", preds);
    Ok(())
}
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
