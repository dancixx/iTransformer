use candle_core::{utils::cuda_is_available, DType, Device, Result, Tensor, WithDType, D};
use candle_einops::einops;
use candle_ext::{TensorExt, F};
use candle_nn::ops::{sigmoid, softmax_last_dim};
use candle_nn::{
    layer_norm, linear, Dropout, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder,
};
use either::Either;

pub struct ITransformer {
    num_variates: usize,
    lookback_len: usize,
    pred_length: Vec<usize>,
    num_tokens_per_variate: usize,
    mem_tokens: Option<Tensor>,
    reversible_instance_norm: Option<RevIn>,
    // expand_streams: HyperStream,
    // reduce_streams: HyperStream,
    layers: Vec<(Attention, FeedForward)>,
    mlp_in: MlpIn,
    pred_heads: Vec<PredHead>,
}

impl ITransformer {
    pub fn new(
        vb: &VarBuilder,
        num_variates: usize,
        lookback_len: usize,
        depth: usize,
        dim: usize,
        num_tokens_per_variate: Option<usize>,
        pred_length: Vec<usize>,
        dim_head: Option<usize>,
        heads: Option<usize>,
        attn_dropout: Option<f32>,
        ff_mult: Option<usize>,
        ff_dropout: Option<f32>,
        num_mem_tokens: Option<usize>,
        // num_residual_streams: Option<usize>,
        use_reversible_instance_norm: Option<bool>,
        reversible_instance_norm_affine: Option<bool>,
        flash_attn: bool,
        device: &Device,
    ) -> Result<Self> {
        let num_tokens_per_variate = num_tokens_per_variate.unwrap_or(1);
        let dim_head = dim_head.unwrap_or(32);
        let heads = heads.unwrap_or(4);
        let attn_dropout = attn_dropout.unwrap_or(0.0);
        let ff_mult = ff_mult.unwrap_or(4);
        let ff_dropout = ff_dropout.unwrap_or(0.0);
        let num_mem_tokens = num_mem_tokens.unwrap_or(4);
        // let num_residual_streams = num_residual_streams.unwrap_or(4);
        let use_reversible_instance_norm = use_reversible_instance_norm.unwrap_or(false);
        let reversible_instance_norm_affine = reversible_instance_norm_affine.unwrap_or(true);

        let mem_tokens = if num_mem_tokens > 0 {
            Some(Tensor::randn(0.0, 1.0, (num_mem_tokens, dim), device)?)
        } else {
            None
        };

        let reversible_instance_norm = if use_reversible_instance_norm {
            Some(RevIn::new(
                num_variates,
                Some(reversible_instance_norm_affine),
                None,
            ))
        } else {
            None
        };

        let mut layers = Vec::with_capacity(depth);

        for idx in 0..depth {
            let attn = Attention::new(
                vb,
                Some(idx),
                dim,
                Some(dim_head),
                Some(heads),
                Some(attn_dropout),
                Some(flash_attn),
                None,
                None,
            )?;
            let ff = FeedForward::new(vb, Some(idx), dim, ff_mult, Some(ff_dropout), None)?;
            layers.push((attn, ff));
        }

        let mlp_in = MlpIn::new(
            &vb,
            lookback_len,
            dim * num_tokens_per_variate,
            num_tokens_per_variate,
        )?;
        let mut pred_heads = Vec::with_capacity(pred_length.len());
        for &pl in &pred_length {
            pred_heads.push(PredHead::new(vb, dim, pl, num_tokens_per_variate)?);
        }

        Ok(Self {
            num_variates,
            lookback_len,
            pred_length,
            num_tokens_per_variate,
            mem_tokens,
            reversible_instance_norm,
            mlp_in,
            pred_heads,
            layers,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        targets: Option<&Vec<Tensor>>,
    ) -> Result<Either<Vec<(usize, Tensor)>, f64>> {
        /// einstein notation
        /// b: batch size
        /// n: time
        /// v: variate
        /// t: num tokens per variate
        let t = self.num_tokens_per_variate;
        let has_mem = self.mem_tokens.is_some();
        assert_eq!(xs.dims()[1..], [self.lookback_len, self.num_variates]);

        // einsum 'b n v -> b v n'
        let xs = einops!("b n v -> b v n", &xs);
        let mut revin: Option<RevInResult> = None;

        if let Some(ref reversible_instance_norm) = &self.reversible_instance_norm {
            let result = reversible_instance_norm.to_owned();
            let result = result.forward(&xs, Some(false));
            revin = Some(result);
        };

        let (xs, reverse_fn, ..) = revin.unwrap()?;
        let mut xs = self.mlp_in.forward(&xs)?;
        let mut mem_ps = 0;

        if has_mem {
            let m = self.mem_tokens.as_ref().unwrap().dim(0)?;
            let d = self.mem_tokens.as_ref().unwrap().dim(1)?;
            let repeated =
                self.mem_tokens
                    .as_ref()
                    .unwrap()
                    .unsqueeze(0)?
                    .repeat(&[xs.dim(0)?, m, d])?;
            mem_ps = repeated.dim(1)?;
            xs = Tensor::cat(&[repeated, xs.clone()], 1)?;
        }

        for (attn, ff) in &self.layers {
            let (attn_out, cache_v) = attn.forward(&xs, self.mem_tokens.as_ref())?;
            let xs_ = &xs + attn_out;
            let xs_ = ff.forward(&xs_?)?;
            xs = xs_
        }

        if has_mem {
            let b = xs.dim(1)?;
            let total_tokens = xs.dim(1)?;
            let d = xs.dim(2)?;
            xs = xs.narrow(1, mem_ps, total_tokens - mem_ps)?;
        }

        if self.reversible_instance_norm.is_some() {
            // einops 'b (n t) d -> t b n d
            let xs_ = xs
                .reshape(&[t, xs.dim(0)?, xs.dim(1)? / t, xs.dim(2)?])?
                .transpose(0, 1)?;
            let xs_ = reverse_fn(&xs_)?;
            // einops 't b n d -> b (n t) d'
            let xs_ = xs_
                .transpose(0, 1)?
                .reshape(&[xs.dim(1)?, xs.dim(2)? * t, xs.dim(3)?])?;
            xs = xs_
        }

        let mut preds = Vec::with_capacity(self.pred_heads.len());
        for pred_head in self.pred_heads {
            let pred = pred_head.forward(&xs)?;
            preds.push(pred);
        }

        if let Some(targets) = targets {
            let mut mse_loss = 0.0;
            for (target, pred) in targets.iter().zip(&preds) {
                assert_eq!(target.dims(), pred.dims());
                mse_loss = mse_loss + (target - pred)?.powf(2.0).sum()?;
            }

            Ok(Either::Right(mse_loss))
        } else {
            let result = self.pred_length.clone();
            let result = result.into_iter().zip(preds).collect::<Vec<_>>();
            Ok(Either::Left(result))
        }
    }
}

pub struct PredHead {
    num_tokens_per_variate: usize,
    linear: Linear,
}

impl PredHead {
    fn new(
        vb: &VarBuilder,
        dim: usize,
        one_pred_length: usize,
        num_tokens_per_variate: usize,
    ) -> Result<Self> {
        Ok(Self {
            num_tokens_per_variate,
            linear: linear(
                dim * num_tokens_per_variate,
                one_pred_length,
                vb.pp("head_linear"),
            )?,
        })
    }
}

impl Module for PredHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let n = self.num_tokens_per_variate;
        // einsum 'b (v n) d -> b v (n d)', n = num_tokens_per_variate
        let xs = einops!("b (v {n}) d -> b v ({n} d)", &xs);
        let xs = self.linear.forward(&xs)?;
        // einsum 'b v n -> b n v'
        let xs = einops!("b v n -> b n v", &xs);
        Ok(xs)
    }
}

struct MlpIn {
    num_tokens_per_variate: usize,
    linear: Linear,
    layer_norm: LayerNorm,
}

impl MlpIn {
    fn new(
        vb: &VarBuilder,
        dim: usize,
        hidden_size: usize,
        num_tokens_per_variate: usize,
    ) -> Result<Self> {
        Ok(Self {
            num_tokens_per_variate,
            linear: linear(dim, hidden_size, vb.pp("mlp_in_linear"))?,
            layer_norm: layer_norm(dim, LayerNormConfig::default(), vb.pp("mlp_in_layernorm"))?,
        })
    }
}

impl Module for MlpIn {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(&xs)?;
        // einsum 'b v (n d) -> b (v n) d', n = num_tokens_per_variate
        let n = self.num_tokens_per_variate;
        let xs = einops!("b v ({n} d) -> b (v {n}) d", &xs);
        let xs = self.layer_norm.forward(&xs)?;
        Ok(xs)
    }
}

struct ToQKV {
    linear: Linear,
    heads: usize,
}

impl ToQKV {
    fn new(vb: &VarBuilder, dim: usize, hidden_size: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            linear: linear(dim, hidden_size * 3, vb.pp("attn_to_qkv"))?,
            heads,
        })
    }
}

impl Module for ToQKV {
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let xs = self.linear.forward(xs)?;
        // einsum: rearrange 'b n (qkv h d) -> qkv b h n d', where qkv = 3 & h = heads
        // [b, n, 3 * heads * dim_head] -> [b, n, 3, heads, dim_head]
        let (h, qkv) = (self.heads, 3);
        let xs = einops!("b n ({qkv} {h} d) -> {qkv} b {h} n d", &xs);

        // chunk qkv into q, k, v
        let chunks = xs.chunk(3, 0)?;
        let q = chunks[0].contiguous()?;
        let k = chunks[1].contiguous()?;
        let v = chunks[2].contiguous()?;

        Ok((q, k, v))
    }
}

pub struct ToValueResidualMix {
    linear: Linear,
}

impl ToValueResidualMix {
    fn new(vb: &VarBuilder, dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            linear: linear(dim, heads, vb.pp("attn_to_value_residual_mix"))?,
        })
    }
}

impl Module for ToValueResidualMix {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(&xs)?;
        let xs = sigmoid(&xs)?;

        // einsum: 'b n h -> b h n 1'
        let xs = einops!("b n h -> b h n 1", &xs);

        Ok(xs)
    }
}

struct ToVGates {
    linear: Linear,
    heads: usize,
}

impl ToVGates {
    fn new(vb: &VarBuilder, dim: usize, heads: usize) -> Result<Self> {
        Ok(Self {
            linear: linear(dim, heads, vb.pp("attn_to_v_gates"))?,
            heads,
        })
    }
}

impl Module for ToVGates {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(&xs)?;
        let xs = sigmoid(&xs)?;

        // einsum: 'b n h -> b h n 1'
        let h = self.heads;
        let xs = einops!("b n h -> b {h} n 1", &xs);

        Ok(xs)
    }
}

pub struct ToOut {
    linear: Linear,
    dropout: Dropout,
    is_train: bool,
}

impl ToOut {
    fn new(
        vb: &VarBuilder,
        dim: usize,
        heads: usize,
        dim_head: usize,
        drop_p: f32,
        is_train: Option<bool>,
    ) -> Result<Self> {
        Ok(Self {
            linear: linear(dim_head * heads, dim, vb.pp("attn_to_out"))?,
            dropout: Dropout::new(drop_p),
            is_train: is_train.unwrap_or(false),
        })
    }
}

impl Module for ToOut {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // einsum 'b n h d -> b n (h d)'
        let xs = einops!("b n h d -> b n (h d)", &xs);
        let xs = self.linear.forward(&xs)?;
        let xs = self.dropout.forward(&xs, self.is_train)?;

        Ok(xs)
    }
}

struct Attention {
    scale: f64,
    layer_norm: LayerNorm,
    to_qkv: ToQKV,
    to_value_residual_mix: Option<ToValueResidualMix>,
    to_v_gates: ToVGates,
    attend: Attend,
    to_out: ToOut,
    dropout: Dropout,
    learned_value_residual_mix: bool,
}

impl Attention {
    fn new(
        vb: &VarBuilder,
        idx: Option<usize>,
        dim: usize,
        dim_head: Option<usize>,
        heads: Option<usize>,
        drop_p: Option<f32>,
        is_flash: Option<bool>,
        is_train: Option<bool>,
        learned_value_residual_mix: Option<bool>,
    ) -> Result<Self> {
        let dim_head = dim_head.unwrap_or(32);
        let heads = heads.unwrap_or(4);
        let scale = (dim_head as f64).sqrt();

        let layer_norm = layer_norm(
            dim,
            LayerNormConfig::default(),
            vb.pp(format!("attn_{}_layernorm", idx.unwrap_or(0))),
        )?;
        let to_qkv = ToQKV::new(vb, dim, dim_head, heads)?;
        let to_value_residual_mix = if learned_value_residual_mix.unwrap_or(false) {
            Some(ToValueResidualMix::new(vb, dim, heads)?)
        } else {
            None
        };
        let to_v_gates = ToVGates::new(vb, dim, heads)?;
        let to_out = ToOut::new(vb, dim, heads, dim_head, drop_p.unwrap_or(0.0), is_train)?;

        Ok(Self {
            scale,
            layer_norm,
            to_qkv,
            to_value_residual_mix,
            to_v_gates,
            attend: Attend::new(drop_p, None, None, is_flash, None, None)?,
            to_out,
            dropout: Dropout::new(drop_p.unwrap_or(0.0)),
            learned_value_residual_mix: learned_value_residual_mix.unwrap_or(false),
        })
    }

    fn forward(&self, xs: &Tensor, value_residual: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let xs = self.layer_norm.forward(xs)?;
        let (q, k, mut v) = self.to_qkv.forward(&xs)?;
        let cache_v = v.clone();

        let lerp = |start: &Tensor, end: &Tensor, weight: &Tensor| -> Result<Tensor> {
            (start * (1.0 - weight))? + (end * weight)?
        };

        if let Some(to_value_residual_mix) = &self.to_value_residual_mix {
            if let Some(value_residual) = value_residual {
                let mix = self.to_value_residual_mix.forward(&xs)?;
                v = lerp(&v, value_residual, &mix)?;
            }
        }

        let out = self.attend.forward(&q, &k, &v)?;
        let out = self.to_out.forward(&out)?;

        Ok((out, cache_v))
    }
}

struct GEGLU;

impl Module for GEGLU {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (x, gate) = xs.chunk2(D::Minus1)?;
        let gate = gate.gelu()?;
        x * gate
    }
}

struct FeedForward {
    is_train: bool,
    layer_norm: LayerNorm,
    linear1: Linear,
    geglu: GEGLU,
    dropout: Dropout,
    linear2: Linear,
}

impl FeedForward {
    fn new(
        vb: &VarBuilder,
        idx: Option<usize>,
        dim: usize,
        mult: usize,
        drop_p: Option<f32>,
        is_train: Option<bool>,
    ) -> Result<Self> {
        let hidden_size = (dim as f64 * mult as f64 * 2.0 / 3.0).round() as usize;

        let layer_norm = layer_norm(
            dim,
            LayerNormConfig::default(),
            vb.pp(format!("ff_{}_layer_norm", idx.unwrap_or(0))),
        )?;
        let linear1 = linear(
            dim,
            hidden_size,
            vb.pp(format!("ff_{}_linear1", idx.unwrap_or(0))),
        )?;
        let linear2 = linear(
            hidden_size,
            dim,
            vb.pp(format!("ff_{}_linear2", idx.unwrap_or(0)))?,
        )?;

        Ok(Self {
            is_train: is_train.unwrap_or(false),
            layer_norm,
            linear1,
            geglu: GEGLU,
            dropout: Dropout::new(drop_p.unwrap_or(0.0)),
            linear2,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.layer_norm.forward(xs)?;
        let xs = self.linear1.forward(&xs)?;
        let xs = self.geglu.forward(&xs)?;
        let xs = self.dropout.forward(&xs, self.is_train)?;
        let xs = self.linear2.forward(&xs)?;
        Ok(xs)
    }
}

struct Attend {
    scale: Option<f64>,
    drop_p: f32,
    is_train: bool,
    attn_dropout: Dropout,
    heads: usize,
    is_flash: bool,
    is_causal: bool,
    device: Device,
}

impl Attend {
    fn new(
        drop_p: Option<f32>,
        heads: Option<usize>,
        scale: Option<f64>,
        is_flash: Option<bool>,
        is_causal: Option<bool>,
        is_train: Option<bool>,
    ) -> Result<Self> {
        let device = if cuda_is_available() {
            Device::cuda_if_available(0)?
        } else {
            Device::Cpu
        };

        Ok(Self {
            scale,
            drop_p: drop_p.unwrap_or(0.0),
            attn_dropout: Dropout::new(drop_p.unwrap_or(0.)),
            heads: heads.unwrap_or(8),
            is_flash: is_flash.unwrap_or(false),
            is_causal: is_causal.unwrap_or(false),
            device,
            is_train: is_train.unwrap_or(false),
        })
    }

    fn flash_attn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        F::scaled_dot_product_attention(
            q,
            k,
            v,
            None,
            Some(self.drop_p),
            Some(self.is_causal),
            self.scale,
        )
    }

    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let scale = if let Some(scale) = self.scale {
            scale
        } else {
            (q.dim(D::Minus1)? as f64).sqrt()
        };

        if self.is_flash {
            return self.flash_attn(q, k, v);
        }

        // einsum('b h i d, b h j d -> b h i j', q, k)
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let mut sim = (q.matmul(&k_t)? * scale)?;

        if self.is_causal {
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
        let attn = self.attn_dropout.forward(&attn, self.is_train)?;

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
#[derive(Clone)]
struct RevIn {
    num_variants: usize,
    eps: f64,
    affine: bool,
    gamma: Tensor,
    beta: Tensor,
}

type RevInResult = Result<(
    Tensor,
    Box<dyn FnOnce(&Tensor) -> Result<Tensor>>,
    Option<Statistics>,
)>;

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

    fn forward(self, xs: &Tensor, return_statistics: Option<bool>) -> RevInResult {
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
