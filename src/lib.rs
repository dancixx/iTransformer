use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Dropout, Module};

/// A transformer module.
pub struct ITransformer;

impl Module for ITransformer {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        unimplemented!()
    }
}

struct Attention {
    scale: f32,
    dropout: f32,
    attn_dropout: Dropout,
    heads: usize,
    flash_attn: bool,
    causal: bool,
}

impl Attention {
    fn new(
        dropout: Option<f32>,
        heads: Option<usize>,
        scale: Option<f32>,
        flash_attn: Option<bool>,
        causal: Option<bool>,
    ) -> Self {
        Self {
            scale: scale.unwrap_or(1.0),
            dropout: dropout.unwrap_or(0.0),
            attn_dropout: Dropout::new(dropout.unwrap_or(0.0)),
            heads: heads.unwrap_or(8),
            flash_attn: flash_attn.unwrap_or(false),
            causal: causal.unwrap_or(false),
        }
    }

    fn flash_attn() {
        unimplemented!()
    }

    fn forward() {
        unimplemented!()
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

        let last_dim = xs.dims().len() - 1;
        let mean = xs.mean_keepdim(last_dim)?;
        // Use biased variance instead of unbiased variance
        // let var = xs.var_keepdim(last_dim)?;
        let biased_var = xs.broadcast_sub(&mean)?.powf(2.0)?.mean_keepdim(last_dim)?;
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
