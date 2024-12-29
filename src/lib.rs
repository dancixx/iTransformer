use candle_core::{utils::cuda_is_available, DType, Device, Result, Tensor, WithDType, D};
use candle_ext::{TensorExt, F};
use candle_nn::ops::softmax_last_dim;
use candle_nn::{Dropout, Module};

pub struct ITransformer;

impl Module for ITransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

struct Attention;

impl Attention {
    fn new() -> Self {
        Self
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

struct GEGLU;

impl Module for GEGLU {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        todo!()
    }
}

struct FeedForward;

struct Attend {
    scale: Option<f64>,
    dropout: f32,
    is_training: bool,
    attn_dropout: Dropout,
    heads: usize,
    flash: bool,
    causal: bool,
    device: Device,
}

impl Attend {
    fn new(
        dropout: Option<f32>,
        heads: Option<usize>,
        scale: Option<f64>,
        flash: Option<bool>,
        causal: Option<bool>,
        is_training: Option<bool>,
    ) -> Result<Self> {
        let device = if cuda_is_available() {
            Device::cuda_if_available(0)?
        } else {
            Device::Cpu
        };

        Ok(Self {
            scale,
            dropout: dropout.unwrap_or(0.0),
            attn_dropout: Dropout::new(dropout.unwrap_or(0.)),
            heads: heads.unwrap_or(8),
            flash: flash.unwrap_or(false),
            causal: causal.unwrap_or(false),
            device,
            is_training: is_training.unwrap_or(false),
        })
    }

    fn flash_attn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        F::scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            Some(self.dropout),
            Some(self.causal),
            self.scale,
        )
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let scale = if let Some(scale) = self.scale {
            scale
        } else {
            (q.dim(D::Minus1)? as f64).sqrt()
        };

        if self.flash {
            return self.flash_attn(q, k, v);
        }

        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut sim = (q.matmul(&k_t)? * scale)?;

        if self.causal {
            let (i, j, dtype) = (sim.dim(D::Minus2)?, sim.dim(D::Minus1)?, sim.dtype());
            // select the causal mask value based on the dtype
            let mask_value = match dtype {
                DType::F32 => f32::NEG_INFINITY as f64,
                DType::F64 => f64::NEG_INFINITY,
                _ => panic!("Unsupported dtype for causal mask"),
            };
            let casual_mask = Tensor::triu2(j - i + 1, DType::U8, &self.device)?;
            sim = sim.masked_fill(&casual_mask, mask_value)?;
        }

        let attn = softmax_last_dim(&sim)?;
        let attn = attn.to_dtype(q.dtype())?;
        let attn = self.attn_dropout.forward(&attn, self.is_training)?;

        let v_t = v.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = attn.matmul(&v_t);

        attn
    }
}

struct Statistics {
    mean: Tensor,
    variance: Tensor,
    gamma: Tensor,
    beta: Tensor,
}

/// A reversible instance normalization module.
/// https://openreview.net/forum?id=cGDAkQo1C0p
struct RevIn {
    num_variants: usize,
    eps: f64,
    affine: bool,
    gamma: Tensor,
    beta: Tensor,
}

impl RevIn {
    fn new(num_variants: usize, affine: Option<bool>, eps: Option<f64>) -> Self {
        let gamma = Tensor::ones((num_variants, 1), DType::F64, &Device::Cpu).unwrap();
        let beta = Tensor::zeros((num_variants, 1), DType::F64, &Device::Cpu).unwrap();

        Self {
            num_variants,
            affine: affine.unwrap_or(true),
            eps: eps.unwrap_or(1e-5),
            gamma,
            beta,
        }
    }

    fn forward(
        self,
        xs: &Tensor,
        return_statistics: Option<bool>,
    ) -> Result<(
        Tensor,
        Box<dyn FnOnce(&Tensor) -> Result<Tensor>>,
        Option<Statistics>,
    )> {
        assert_eq!(xs.dims()[1] == self.num_variants, true);

        let mean = xs.mean_keepdim(D::Minus1)?;
        // Use biased variance instead of unbiased variance
        // let var = xs.var_keepdim(last_dim)?;
        let biased_var = xs
            .broadcast_sub(&mean)?
            .powf(2.0)?
            .mean_keepdim(D::Minus1)?;
        let var_rsqrt = biased_var.clamp(self.eps, f64::INFINITY)?.sqrt()?.recip()?;
        let instance_norm = xs.broadcast_sub(&mean)?.broadcast_mul(&var_rsqrt)?;
        let rescaled = (instance_norm
            .broadcast_mul(&self.gamma)?
            .broadcast_add(&self.beta))?;

        let reverse_fn = {
            let mean = mean.clone();
            let biased_var = biased_var.clone();
            let gamma = self.gamma.clone();
            let beta = self.beta.clone();

            move |xs: &Tensor| -> Result<Tensor> {
                let clamped_gamma = gamma.sign()? * gamma.abs()?.clamp(self.eps, f64::INFINITY)?;
                let unscaled_output = xs.broadcast_sub(&beta)?.broadcast_div(&clamped_gamma?);
                unscaled_output?
                    .broadcast_mul(&biased_var.sqrt()?)?
                    .broadcast_add(&mean)
            }
        };

        let mut statistics = None;

        if return_statistics.unwrap_or(false) {
            statistics = Some(Statistics {
                mean: mean.clone(),
                variance: biased_var.clone(),
                gamma: self.gamma.clone(),
                beta: self.beta.clone(),
            });
        }

        Ok((rescaled, Box::new(reverse_fn), statistics))
    }
}

#[cfg(test)]
mod tests {
    use crate::RevIn;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_rev_in_v1() {
        let rev_in = RevIn::new(4, None, None);
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0,
            59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
        ];

        let xs = Tensor::from_vec(data, (2, 4, 8), &Device::Cpu).unwrap();
        let (normalized, reverse_fn, _) = rev_in.forward(&xs, Some(true)).unwrap();
        let out = reverse_fn(&normalized).unwrap();

        fn round_to_decimal_places(x: f64, places: u32) -> f64 {
            let factor = 10f64.powi(places as i32);
            (x * factor).round() / factor
        }

        let xs = xs.to_vec3::<f64>().unwrap();
        let out = out.to_vec3::<f64>().unwrap();
        let out = out
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| {
                        v.iter()
                            .map(|&x| round_to_decimal_places(x, 5))
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();

        assert_eq!(out, xs);
    }

    #[test]
    fn test_rev_in_v2() {
        let rev_in = RevIn::new(4, None, None);
        let xs = Tensor::randn(0., 1., (2, 4, 8), &Device::Cpu).unwrap();
        let (normalized, reverse_fn, _) = rev_in.forward(&xs, Some(true)).unwrap();
        let out = reverse_fn(&normalized).unwrap();

        fn round_to_decimal_places(x: f64, places: u32) -> f64 {
            let factor = 10f64.powi(places as i32);
            (x * factor).round() / factor
        }

        let xs = xs.to_vec3::<f64>().unwrap();
        let out = out.to_vec3::<f64>().unwrap();
        let out = out
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| {
                        v.iter()
                            .map(|&x| round_to_decimal_places(x, 5))
                            .collect::<Vec<f64>>()
                    })
                    .collect::<Vec<Vec<f64>>>()
            })
            .collect::<Vec<Vec<Vec<f64>>>>();

        assert_eq!(out, xs);
    }
}
