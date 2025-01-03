use either::Either;
use tch::{
    nn::{
        layer_norm, linear, LayerNorm, LayerNormConfig, Linear, LinearConfig, Module, ModuleT, Path,
    },
    Device, Kind, Result, Tensor,
};

// pub struct ITransformer {
//     num_variates: usize,
//     lookback_len: usize,
//     pred_length: Vec<usize>,
//     num_tokens_per_variate: usize,
//     mem_tokens: Option<Tensor>,
//     reversible_instance_norm: Option<RevIn>,
//     // expand_streams: HyperStream,
//     // reduce_streams: HyperStream,
//     layers: Vec<(Attention, FeedForward)>,
//     mlp_in: MlpIn,
//     pred_heads: Vec<PredHead>,
// }
//
// impl ITransformer {
//     pub fn new(
//         vb: &VarBuilder,
//         num_variates: usize,
//         lookback_len: usize,
//         depth: usize,
//         dim: usize,
//         num_tokens_per_variate: Option<usize>,
//         pred_length: Vec<usize>,
//         dim_head: Option<usize>,
//         heads: Option<usize>,
//         attn_dropout: Option<f32>,
//         ff_mult: Option<usize>,
//         ff_dropout: Option<f32>,
//         num_mem_tokens: Option<usize>,
//         // num_residual_streams: Option<usize>,
//         use_reversible_instance_norm: Option<bool>,
//         reversible_instance_norm_affine: Option<bool>,
//         flash_attn: bool,
//         device: &Device,
//     ) -> Result<Self> {
//         let num_tokens_per_variate = num_tokens_per_variate.unwrap_or(1);
//         let dim_head = dim_head.unwrap_or(32);
//         let heads = heads.unwrap_or(4);
//         let attn_dropout = attn_dropout.unwrap_or(0.0);
//         let ff_mult = ff_mult.unwrap_or(4);
//         let ff_dropout = ff_dropout.unwrap_or(0.0);
//         let num_mem_tokens = num_mem_tokens.unwrap_or(4);
//         // let num_residual_streams = num_residual_streams.unwrap_or(4);
//         let use_reversible_instance_norm = use_reversible_instance_norm.unwrap_or(false);
//         let reversible_instance_norm_affine = reversible_instance_norm_affine.unwrap_or(true);
//
//         let mem_tokens = if num_mem_tokens > 0 {
//             Some(Tensor::randn(0.0f32, 1.0f32, (num_mem_tokens, dim), device)?)
//         } else {
//             None
//         };
//
//         let reversible_instance_norm = if use_reversible_instance_norm {
//             Some(RevIn::new(
//                 num_variates,
//                 Some(reversible_instance_norm_affine),
//                 None,
//                 device
//             )?)
//         } else {
//             None
//         };
//
//         let mut layers = Vec::with_capacity(depth);
//
//         for idx in 0..depth {
//             let attn = Attention::new(
//                 vb,
//                 Some(idx),
//                 dim,
//                 Some(dim_head),
//                 Some(heads),
//                 Some(attn_dropout),
//                 Some(flash_attn),
//                 None,
//                 None,
//                 &device
//             )?;
//             let ff = FeedForward::new(vb, Some(idx), dim, ff_mult, Some(ff_dropout), None)?;
//             layers.push((attn, ff));
//         }
//
//         let mlp_in = MlpIn::new(
//             &vb,
//             lookback_len,
//             dim * num_tokens_per_variate,
//             num_tokens_per_variate,
//         )?;
//         let mut pred_heads = Vec::with_capacity(pred_length.len());
//         for (idx, pl) in pred_length.iter().enumerate() {
//             pred_heads.push(PredHead::new(
//                 vb,
//                 Some(idx),
//                 dim,
//                 *pl,
//                 num_tokens_per_variate,
//             )?);
//         }
//
//         Ok(Self {
//             num_variates,
//             lookback_len,
//             pred_length,
//             num_tokens_per_variate,
//             mem_tokens,
//             reversible_instance_norm,
//             mlp_in,
//             pred_heads,
//             layers,
//         })
//     }
//
//     fn forward(
//         &self,
//         xs: &Tensor,
//         targets: Option<&Vec<Tensor>>,
//     ) -> Result<Either<Vec<(usize, Tensor)>, f64>> {
//         // einstein notation
//         // b: batch size
//         // n: time
//         // v: variate
//         // t: num tokens per variate
//         let t = self.num_tokens_per_variate;
//         let has_mem = self.mem_tokens.is_some();
//         assert_eq!(xs.dims()[1..], [self.lookback_len, self.num_variates]);
//
//         // einsum 'b n v -> b v n'
//         let xs = xs.transpose(1, 2)?;
//         let mut revin: Option<RevInResult> = None;
//
//         if let Some(ref reversible_instance_norm) = &self.reversible_instance_norm {
//             let result = reversible_instance_norm.to_owned();
//             let result = result.forward(&xs, Some(false));
//             revin = Some(result);
//         };
//
//         let (xs, reverse_fn, ..) = revin.unwrap()?;
//         let mut xs = self.mlp_in.forward(&xs)?;
//         let mut mem_ps = 0;
//
//         if has_mem {
//             let mem_tokens = self.mem_tokens.as_ref().unwrap();
//             let b = xs.dim(0)?; // batch-méret
//             let m = mem_tokens.dim(0)?;
//             let d = mem_tokens.dim(1)?;
//
//             let repeated = mem_tokens
//               .unsqueeze(0)?
//               .repeat(&[b, 1, 1])?.to_dtype(DType::F32)?;
//             mem_ps = repeated.dim(1)?;
//             xs = Tensor::cat(&[repeated, xs.clone()], 1)?;
//         }
//
//         for (attn, ff) in &self.layers {
//             let (attn_out, ..) = attn.forward(&xs, self.mem_tokens.as_ref())?;
//             let xs_ = &xs + attn_out;
//             let xs_ = ff.forward(&xs_?)?;
//             xs = xs_
//         }
//
//         if has_mem {
//             let b = xs.dim(1)?;
//             let total_tokens = xs.dim(1)?;
//             let d = xs.dim(2)?;
//             xs = xs.narrow(1, mem_ps, total_tokens - mem_ps)?;
//         }
//
//         if self.reversible_instance_norm.is_some() {
//             // einops 'b (n t) d -> t b n d
//             let xs_ = xs
//                 .reshape(&[t, xs.dim(0)?, xs.dim(1)? / t, xs.dim(2)?])?
//                 .transpose(0, 1)?;
//             let xs_ = reverse_fn(&xs_)?;
//             // einops 't b n d -> b (n t) d'
//             let xs_ = xs_
//                 .transpose(0, 1)?
//                 .reshape(&[xs.dim(1)?, xs.dim(2)? * t, xs.dim(3)?])?;
//             xs = xs_
//         }
//
//         let mut preds = Vec::with_capacity(self.pred_heads.len());
//         for pred_head in &self.pred_heads {
//             let pred = pred_head.forward(&xs)?;
//             preds.push(pred);
//         }
//
//         if let Some(targets) = targets {
//             let mut mse_loss = 0.0;
//             for (target, pred) in targets.iter().zip(&preds) {
//                 assert_eq!(target.dims(), pred.dims());
//                 // mse_loss = mse_loss + (target - pred)?.powf(2.0)?.into_iter().sum()?;
//             }
//
//             Ok(Either::Right(mse_loss))
//         } else {
//             let result = self.pred_length.clone();
//             let result = result.into_iter().zip(preds).collect::<Vec<_>>();
//             Ok(Either::Left(result))
//         }
//     }
// }
//
// pub struct PredHead {
//     num_tokens_per_variate: usize,
//     linear: Linear,
// }
//
// impl PredHead {
//     fn new(
//         vb: &VarBuilder,
//         idx: Option<usize>,
//         dim: usize,
//         one_pred_length: usize,
//         num_tokens_per_variate: usize,
//     ) -> Result<Self> {
//         Ok(Self {
//             num_tokens_per_variate,
//             linear: linear(
//                 dim * num_tokens_per_variate,
//                 one_pred_length,
//                 vb.pp(format!("head_{}_linear", idx.unwrap_or(0))),
//             )?,
//         })
//     }
// }
//
// impl Module for PredHead {
//     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
//         let n = self.num_tokens_per_variate;
//         let b = xs.dim(0)?;
//         let v = xs.dim(1)?;
//         let d = xs.dim(2)?;
//         // einsum 'b (v n) d -> b v (n d)', n = num_tokens_per_variate
//         let xs = xs.reshape(&[b, v, n * d])?;
//         let xs = self.linear.forward(&xs)?;
//         // einsum 'b v n -> b n v'
//         let xs = xs.transpose(1, 2)?;
//         Ok(xs)
//     }
// }
//
// struct MlpIn {
//     num_tokens_per_variate: usize,
//     linear: Linear,
//     layer_norm: LayerNorm,
// }
//
// impl MlpIn {
//     fn new(
//         vb: &VarBuilder,
//         dim: usize,
//         hidden_size: usize,
//         num_tokens_per_variate: usize,
//     ) -> Result<Self> {
//         Ok(Self {
//             num_tokens_per_variate,
//             linear: linear(dim, hidden_size, vb.pp("mlp_in_linear"))?,
//             layer_norm: layer_norm(hidden_size, LayerNormConfig::default(), vb.pp("mlp_in_layernorm"))?,
//         })
//     }
// }
//
// impl Module for MlpIn {
//     fn forward(&self, xs: &Tensor) -> Result<Tensor> {
//         let xs = self.linear.forward(&xs)?;
//         // einsum 'b v (n d) -> b (v n) d', n = num_tokens_per_variate
//         let n = self.num_tokens_per_variate;
//         let b = xs.dim(0)?;
//         let v = xs.dim(1)?;
//         let nd = xs.dim(2)?;
//         let d = nd / n;
//         let xs = xs.reshape(&[b, v * n, d])?;
//         let xs = self.layer_norm.forward(&xs)?;
//         Ok(xs)
//     }
// }
//

#[derive(Debug)]
struct ToQKV {
    linear: Linear,
    heads: i64,
}

impl ToQKV {
    fn new(vs: &Path, dim: i64, hidden_size: i64, heads: i64) -> Self {
        Self {
            linear: linear(
                vs,
                dim,
                hidden_size * 3,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            heads,
        }
    }

    fn rearrange(&self, xs: &Tensor) -> (Tensor, Tensor, Tensor) {
        // einsum: rearrange 'b n (qkv h d) -> qkv b h n d', where qkv = 3 & h = heads
        // [b, n, 3 * heads * dim_head] -> [b, n, 3, heads, dim_head]
        let xs_dims = xs.size();

        let (h, qkv) = (self.heads, 3);
        let b = xs_dims[0];
        let n = xs_dims[1];
        let total_dim = xs_dims[2];
        let dim_head = total_dim / (qkv * h);
        let xs = xs.reshape(&[b, n, qkv, h, dim_head]);
        let xs = xs.permute(&[2, 0, 3, 1, 4]);

        let chunks = xs.chunk(qkv, 0);
        let q = chunks[0].squeeze();
        let k = chunks[1].squeeze();
        let v = chunks[2].squeeze();

        (q, k, v)
    }

    fn forward(&self, xs: &Tensor) -> (Tensor, Tensor, Tensor) {
        let xs = self.linear.forward(xs);
        let qkv = self.rearrange(&xs);
        qkv
    }
}

#[derive(Debug)]
pub struct ToValueResidualMix {
    linear: Linear,
}

impl ToValueResidualMix {
    fn new(vs: &Path, dim: i64, heads: i64) -> Self {
        Self {
            linear: linear(
                vs,
                dim,
                heads,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
        }
    }

    fn rearrange(&self, xs: &Tensor) -> Tensor {
        // einsum: 'b n h -> b h n 1'
        let xs = xs.permute(&[0, 2, 1]);
        let xs = xs.unsqueeze(-1);
        xs
    }
}

impl Module for ToValueResidualMix {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.linear.forward(&xs);
        let xs = self.rearrange(&xs);
        let xs = xs.sigmoid();
        xs
    }
}

#[derive(Debug)]
struct ToVGates {
    linear: Linear,
    heads: i64,
}

impl ToVGates {
    fn new(vs: &Path, dim: i64, heads: i64) -> Self {
        Self {
            linear: linear(
                vs,
                dim,
                heads,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
            heads,
        }
    }

    fn rearrange(&self, xs: &Tensor) -> Tensor {
        // einsum: 'b n h -> b h n 1', where h = heads
        let xs = xs.permute(&[0, 2, 1]);
        let xs = xs.unsqueeze(-1);
        xs
    }
}

impl Module for ToVGates {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let xs = self.linear.forward(&xs);
        let xs = xs.sigmoid();

        // einsum: 'b n h -> b h n 1'
        let xs = self.rearrange(&xs);

        xs
    }
}

#[derive(Debug)]
pub struct ToOut {
    drop_p: f64,
    linear: Linear,
}

impl ToOut {
    fn new(vs: &Path, dim: i64, heads: i64, dim_head: i64, drop_p: Option<f64>) -> Self {
        Self {
            drop_p: drop_p.unwrap_or(0.0),
            linear: linear(
                vs,
                dim_head * heads,
                dim,
                LinearConfig {
                    bias: false,
                    ..Default::default()
                },
            ),
        }
    }

    fn rearrange(&self, xs: &Tensor) -> Tensor {
        // einsum 'b n h d -> b n (h d)'
        let xs_dims = xs.size();
        let (b, h, n, d) = (xs_dims[0], xs_dims[1], xs_dims[2], xs_dims[3]);
        let xs = xs.reshape(&[b, n, h * d]);
        xs
    }
}

impl ModuleT for ToOut {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.rearrange(xs);
        let xs = self.linear.forward(&xs);
        let xs = xs.dropout(self.drop_p, train);
        xs
    }
}

struct Attention {
    scale: f64,
    drop_p: f64,
    norm: LayerNorm,
    to_qkv: ToQKV,
    to_value_residual_mix: Option<ToValueResidualMix>,
    to_v_gates: ToVGates,
    attend: Attend,
    to_out: ToOut,
    learned_value_residual_mix: bool,
}

impl Attention {
    fn new(
        vs: &Path,
        dim: i64,
        dim_head: Option<i64>,
        heads: Option<i64>,
        drop_p: Option<f64>,
        is_flash: Option<bool>,
        learned_value_residual_mix: Option<bool>,
    ) -> Self {
        let dim_head = dim_head.unwrap_or(32);
        let heads = heads.unwrap_or(4);
        let scale = (dim_head as f64).sqrt();

        let norm = layer_norm(vs, vec![dim], LayerNormConfig::default());
        let to_qkv = ToQKV::new(vs, dim, dim_head * heads, heads);
        let to_value_residual_mix = if learned_value_residual_mix.unwrap_or(false) {
            Some(ToValueResidualMix::new(vs, dim, heads))
        } else {
            None
        };
        let to_v_gates = ToVGates::new(vs, dim, heads);
        let to_out = ToOut::new(vs, dim, heads, dim_head, drop_p);

        Self {
            scale,
            drop_p: drop_p.unwrap_or(0.0),
            norm,
            to_qkv,
            to_value_residual_mix,
            to_v_gates,
            attend: Attend::new(drop_p, None, None, is_flash, None),
            to_out,
            learned_value_residual_mix: learned_value_residual_mix.unwrap_or(false),
        }
    }

    fn forward_t(
        &self,
        xs: &Tensor,
        value_residual: Option<&Tensor>,
        train: bool,
    ) -> Result<(Tensor, Tensor)> {
        let xs = self.norm.forward(xs);
        let (q, k, mut v) = self.to_qkv.forward(&xs);
        let cache_v = v.copy();

        if self.to_value_residual_mix.is_some() {
            if let Some(value_residual) = value_residual {
                let mix = self.to_value_residual_mix.as_ref().unwrap().forward(&xs);
                v = v.lerp_tensor(&value_residual, &mix);
            }
        }

        let out = self.attend.forward_t(&q, &k, &v, train)?;
        let gates = self.to_v_gates.forward(&xs);
        let out = out * gates;
        let out = self.to_out.forward_t(&out, train);

        Ok((out, cache_v))
    }
}

#[derive(Debug)]
struct GEGLU;

impl Module for GEGLU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let tensors = self.rearrange(xs);
        let gate = tensors[1].gelu("none");
        // TODO: find a way to avoid cloning the variables
        let x = tensors[0].copy();
        x * gate
    }
}

impl GEGLU {
    fn rearrange(&self, xs: &Tensor) -> Vec<Tensor> {
        let last_dim = xs.size();
        let last_dim = last_dim[last_dim.len() - 1];
        // x, gate = rearrange(x, '... (r d) -> r ... d', r = 2)
        let reshaped = xs.view([-1, 2, last_dim / 2]);
        let tensors = reshaped.unbind(1);
        tensors
    }
}

#[derive(Debug)]
struct FeedForward {
    drop_p: f64,
    norm: LayerNorm,
    linear1: Linear,
    geglu: GEGLU,
    linear2: Linear,
}

impl FeedForward {
    fn new(vs: &Path, dim: i64, mult: i64, drop_p: Option<f64>) -> Self {
        let hidden_size = (dim as f64 * mult as f64 * 2.0 / 3.0).round() as i64;
        let norm = layer_norm(vs, vec![dim], LayerNormConfig::default());
        let linear1 = linear(vs, dim, hidden_size * 2, Default::default());
        let linear2 = linear(vs, hidden_size, dim, Default::default());

        Self {
            drop_p: drop_p.unwrap_or(0.0),
            norm,
            linear1,
            geglu: GEGLU,
            linear2,
        }
    }
}

impl ModuleT for FeedForward {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let xs = self.norm.forward(xs);
        let xs = self.linear1.forward(&xs);
        let xs = self.geglu.forward(&xs);
        let xs = xs.dropout(self.drop_p, train);
        let xs = self.linear2.forward(&xs);
        xs
    }
}

#[derive(Debug)]
struct Attend {
    scale: Option<f64>,
    drop_p: f64,
    heads: i64,
    is_flash: bool,
    is_causal: bool,
}

impl Attend {
    fn new(
        drop_p: Option<f64>,
        heads: Option<i64>,
        scale: Option<f64>,
        is_flash: Option<bool>,
        is_causal: Option<bool>,
    ) -> Self {
        Self {
            scale,
            drop_p: drop_p.unwrap_or(0.0),
            heads: heads.unwrap_or(8),
            is_flash: is_flash.unwrap_or(false),
            is_causal: is_causal.unwrap_or(false),
        }
    }

    fn flash_attn(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        // SDPA Backend is not implemented yet
        Ok(Tensor::scaled_dot_product_attention::<Tensor>(
            q,
            k,
            v,
            None,
            self.drop_p,
            self.is_causal,
            self.scale,
            false,
        ))
    }

    fn forward_t(&self, q: &Tensor, k: &Tensor, v: &Tensor, train: bool) -> Result<Tensor> {
        let q_size = q.size();
        let scale = self
            .scale
            .unwrap_or((q_size[q_size.len() - 1] as f64).sqrt());

        if self.is_flash {
            return self.flash_attn(q, k, v);
        }

        // einsum('b h i d, b h j d -> b h i j', q, k)
        // b: batch size
        // h: num heads
        // n, i, j: sequence length (base sequence length, source sequence length, target sequence length)
        // d: feature dimension
        let k_t = k.transpose(-2, -1);
        let mut sim = q.matmul(&k_t) * scale;

        if self.is_causal {
            let sim_dims = sim.size();
            let (i, j, dtype) = (
                sim_dims[sim_dims.len() - 2],
                sim_dims[sim_dims.len() - 1],
                sim.kind(),
            );
            let mask_value = match dtype {
                Kind::Float => f64::NEG_INFINITY,
                Kind::Double => f64::NEG_INFINITY,
                _ => unreachable!(),
            };
            let casual_mask = Tensor::ones(&[i, j], (Kind::Bool, q.device())).triu(j - i + 1);
            sim = sim.masked_fill(&casual_mask, mask_value);
        }

        let attn = sim.softmax(-1, sim.kind());
        let attn = attn.to_kind(sim.kind());
        let attn = attn.dropout(self.drop_p, train);

        // einsum('b h i j, b h j d -> b h i d', attn, v)
        let attn = attn.matmul(&v);

        Ok(attn)
    }
}

struct Statistics {
    mean: Tensor,
    var: Tensor,
    gamma: Tensor,
    beta: Tensor,
}

/// A reversible instance normalization module.
/// https://openreview.net/forum?id=cGDAkQo1C0p
// #[derive(Clone)]
struct RevIn {
    num_variants: i64,
    eps: f64,
    affine: bool,
    gamma: Tensor,
    beta: Tensor,
}

impl RevIn {
    fn new(
        num_variants: i64,
        affine: Option<bool>,
        eps: Option<f64>,
        device: &Device,
    ) -> Result<Self> {
        let gamma = Tensor::ones(&[num_variants, 1], (tch::Kind::Float, *device));
        let beta = Tensor::zeros(&[num_variants, 1], (tch::Kind::Float, *device));

        Ok(Self {
            num_variants,
            affine: affine.unwrap_or(true),
            eps: eps.unwrap_or(1e-5),
            gamma,
            beta,
        })
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
        assert_eq!(xs.size()[1] == self.num_variants, true);

        let var = xs.var_dim(-1, false, true);
        let mean = xs.mean_dim(-1, true, None);
        let var_rsqrt = var.clamp(self.eps, f64::INFINITY).rsqrt();
        let instance_norm = (xs - &mean) * &var_rsqrt;
        let rescaled = instance_norm * &self.gamma + &self.beta;

        // TODO: find a way to avoid cloning the variables
        let reverse_fn = {
            let mean = mean.copy();
            let var = var.copy();
            let gamma = self.gamma.copy();
            let beta = self.beta.copy();

            move |scaled_output: &Tensor| -> Result<Tensor> {
                let clamped_gamma = gamma.sign() * gamma.abs().clamp(self.eps, f64::INFINITY);
                let unscaled_output = (scaled_output - beta) / clamped_gamma;
                Ok(unscaled_output * var.sqrt() + mean)
            }
        };

        let mut statistics = None;

        if return_statistics.unwrap_or(false) {
            statistics = Some(Statistics {
                mean,
                var,
                gamma: self.gamma,
                beta: self.beta,
            });
        }

        Ok((rescaled, Box::new(reverse_fn), statistics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;
    use tch::{nn::VarStore, Device, Kind, Tensor};

    static DEVICE: LazyLock<Device> = LazyLock::new(|| Device::Cpu);

    #[test]
    fn test_rev_in_v1() {
        let xs = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0,
            59.0, 60.0, 61.0, 62.0, 63.0, 64.0,
        ];

        let xs = Tensor::from_slice(&xs)
            .reshape(&[2, 4, 8])
            .to_device(*DEVICE);
        let rev_in = RevIn::new(4, None, None, &DEVICE).unwrap();
        let (normalized, reverse_fn, _) = rev_in.forward(&xs, Some(true)).unwrap();
        let out = reverse_fn(&normalized).unwrap();

        assert_eq!(Tensor::allclose(&xs, &out, 1e-5, 1e-5, false), true);
    }

    #[test]
    fn test_rev_in_v2() {
        let xs = Tensor::randn(&[2, 512, 1024], (Kind::Float, *DEVICE));
        let rev_in = RevIn::new(512, None, None, &DEVICE).unwrap();
        let (normalized, reverse_fn, _) = rev_in.forward(&xs, Some(true)).unwrap();
        let out = reverse_fn(&normalized).unwrap();

        assert_eq!(Tensor::allclose(&xs, &out, 1e-5, 1e-5, false), true);
    }

    #[test]
    fn test_geglu_rearrange() {
        let data = (1..=8).collect::<Vec<_>>();
        let xs = Tensor::from_slice(&data)
            .reshape(&[2, 4])
            .to_device(*DEVICE);

        let geglu = GEGLU;
        let tensors = geglu.rearrange(&xs);

        let x = Tensor::from_slice(&[1, 2, 5, 6])
            .reshape(&[2, 2])
            .to_device(*DEVICE);
        assert_eq!(tensors[0], x);

        let gate = Tensor::from_slice(&[3, 4, 7, 8])
            .reshape(&[2, 2])
            .to_device(*DEVICE);
        assert_eq!(tensors[1], gate);
    }

    #[test]
    fn test_geglu_forward() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let xs = Tensor::from_slice(&data)
            .reshape(&[2, 4])
            .to_device(*DEVICE);

        let geglu = GEGLU;
        let _ = geglu.forward(&xs);
    }

    #[test]
    fn test_feed_forward() {
        let vs = VarStore::new(*DEVICE);
        let xs = Tensor::randn(&[2, 512], (Kind::Float, *DEVICE));
        let feed_forward = FeedForward::new(&(vs.root() / "ff"), 512, 4, Some(0.1));
        let _ = feed_forward.forward_t(&xs, true);
    }

    #[test]
    fn test_attend_einsum_operations() {
        let b = 2; // batch size
        let h = 4; // number of heads
        let i = 5; // query length
        let j = 5; // key length
        let d = 8; // dimension
        let scale = 0.125;

        // Initialize tensors
        let q = Tensor::randn(&[b, h, i, d], (Kind::Float, Device::Cpu));
        let k = Tensor::randn(&[b, h, j, d], (Kind::Float, Device::Cpu));
        let v = Tensor::randn(&[b, h, j, d], (Kind::Float, Device::Cpu));

        // Verify tensor shapes
        assert_eq!(q.size(), vec![b, h, i, d]);
        assert_eq!(k.size(), vec![b, h, j, d]);
        assert_eq!(v.size(), vec![b, h, j, d]);

        // 1. Attention Logit Matrix: einsum('b h i d, b h j d -> b h i j', q, k) * scale
        let k_t = k.transpose(-2, -1); // Transpose: [b, h, j, d] -> [b, h, d, j]
        let sim = q.matmul(&k_t) * scale; // [b, h, i, d] x [b, h, d, j] -> [b, h, i, j]

        // Verify result shape
        assert_eq!(sim.size(), vec![b, h, i, j]);

        // Apply softmax to the similarity matrix
        let attn = sim.softmax(-1, Kind::Float); // Softmax on the last dimension (j)

        // 2. Apply attention to the Value tensor: einsum('b h i j, b h j d -> b h i d', attn, v)
        let attn_output = attn.matmul(&v); // [b, h, i, j] x [b, h, j, d] -> [b, h, i, d]

        // Verify final result shape
        assert_eq!(attn_output.size(), vec![b, h, i, d]);

        println!("Attention logit shape: {:?}", sim.size());
        println!("Attention output shape: {:?}", attn_output.size());
    }

    #[test]
    fn test_attend_forward() {
        let b = 2; // batch size
        let h = 4; // number of heads
        let i = 5; // query length
        let j = 5; // key length
        let d = 8; // dimension

        // Initialize tensors
        let q = Tensor::randn(&[b, h, i, d], (Kind::Float, Device::Cpu));
        let k = Tensor::randn(&[b, h, j, d], (Kind::Float, Device::Cpu));
        let v = Tensor::randn(&[b, h, j, d], (Kind::Float, Device::Cpu));

        // Iterate over all boolean combinations
        let combinations = [(false, false), (false, true), (true, false), (true, true)];

        for (is_flash, is_causal) in combinations {
            let attend = Attend::new(
                Some(0.1),
                Some(4),
                Some(0.125),
                Some(is_flash),
                Some(is_causal),
            );
            println!(
                "Testing Attend with is_flash: {}, is_causal: {}",
                is_flash, is_causal
            );

            match attend.forward_t(&q, &k, &v, true) {
                Ok(result) => {
                    println!("Success! Output shape: {:?}\n", result.size());
                }
                Err(err) => {
                    eprintln!(
                        "Error with is_flash: {}, is_causal: {}: {}\n",
                        is_flash, is_causal, err
                    );
                }
            }
        }
    }

    #[test]
    fn test_rearrange() {
        // Test Parameters
        let b = 2; // Batch size
        let n = 3; // Sequence length
        let qkv = 3; // Number of Q, K, V
        let h = 2; // Number of heads
        let dim_head = 4; // Dimension per head

        let total_dim = qkv * h * dim_head; // Calculate total dimension

        // Create input tensor with known values
        let data: Vec<i64> = (1..=(b * n * total_dim)).collect();
        let xs = Tensor::from_slice(&data)
            .reshape(&[b, n, total_dim])
            .to_device(Device::Cpu);

        println!("Input Tensor Shape: {:?}", xs.size());

        let vs = VarStore::new(*DEVICE);
        let to_qkv = ToQKV::new(&(vs.root() / "to_qkv"), total_dim, dim_head, h);
        let (q, k, v) = to_qkv.rearrange(&xs);

        // Expected shapes
        let expected_q_shape = [1, b, h, n, dim_head];
        let expected_k_shape = [1, b, h, n, dim_head];
        let expected_v_shape = [1, b, h, n, dim_head];

        assert_eq!(q.size(), expected_q_shape, "Q tensor shape mismatch");
        assert_eq!(k.size(), expected_k_shape, "K tensor shape mismatch");
        assert_eq!(v.size(), expected_v_shape, "V tensor shape mismatch");

        // Verify Q, K, V contain expected slices
        let q_expected = xs
            .reshape(&[b, n, qkv, h, dim_head])
            .permute(&[2, 0, 3, 1, 4])
            .slice(0, 0, 1, 1)
            .contiguous();

        let k_expected = xs
            .reshape(&[b, n, qkv, h, dim_head])
            .permute(&[2, 0, 3, 1, 4])
            .slice(0, 1, 2, 1)
            .contiguous();

        let v_expected = xs
            .reshape(&[b, n, qkv, h, dim_head])
            .permute(&[2, 0, 3, 1, 4])
            .slice(0, 2, 3, 1)
            .contiguous();

        assert_eq!(q, q_expected, "Q tensor values mismatch");
        assert_eq!(k, k_expected, "K tensor values mismatch");
        assert_eq!(v, v_expected, "V tensor values mismatch");

        println!("Test passed: Q, K, and V tensors match the expected shapes and values!");
    }

    #[test]
    fn test_to_qkv_forward() {
        let b = 2; // batch size
        let n = 3; // sequence length
        let qkv = 3; // number of Q, K, V
        let h = 2; // number of heads
        let dim_head = 4; // dimension per head
        let total_dim = qkv * h * dim_head; // calculate total dimension

        // create input tensor with known values
        let xs = Tensor::randn(&[b, n, total_dim], (Kind::Float, Device::Cpu));

        let vs = VarStore::new(*DEVICE);
        let to_qkv = ToQKV::new(&(vs.root() / "to_qkv"), total_dim, dim_head, h);
        let _ = to_qkv.forward(&xs);
    }

    #[test]
    fn test_to_value_residual_mix_rearrange() {
        let module = ToValueResidualMix::new(&VarStore::new(Device::Cpu).root(), 4, 4);

        // Input tensor: [batch_size, sequence_length, dim]
        let xs = Tensor::randn(&[2, 3, 4], (Kind::Float, Device::Cpu));

        let rearranged = module.rearrange(&xs);

        // Expected output shape: [batch_size, dim, sequence_length, 1]
        assert_eq!(rearranged.size(), vec![2, 4, 3, 1]);
    }

    #[test]
    fn test_to_value_residual_mix_forward() {
        let vs = VarStore::new(Device::Cpu);
        let module = ToValueResidualMix::new(&vs.root(), 8, 4);

        // Input tensor: [batch_size, sequence_length, dim]
        let xs = Tensor::randn(&[2, 3, 8], (Kind::Float, Device::Cpu));

        let output = module.forward(&xs);

        // Expected output shape: [batch_size, heads, sequence_length, 1]
        assert_eq!(output.size(), vec![2, 4, 3, 1]);
    }

    #[test]
    fn test_to_vgates_rearrange() {
        let vs = VarStore::new(Device::Cpu);
        let to_vgates = ToVGates::new(&vs.root(), 4, 4);

        // Input tensor: [batch_size, sequence_length, heads]
        let xs = Tensor::randn(&[2, 3, 4], (Kind::Float, Device::Cpu));

        let output = to_vgates.rearrange(&xs);

        // Expected output shape: [batch_size, heads, sequence_length, 1]
        assert_eq!(output.size(), vec![2, 4, 3, 1]);
    }

    #[test]
    fn test_to_vgates_forward() {
        let vs = VarStore::new(Device::Cpu);
        let to_vgates = ToVGates::new(&vs.root(), 4, 4);

        // Input tensor: [batch_size, sequence_length, heads]
        let xs = Tensor::randn(&[2, 3, 4], (Kind::Float, Device::Cpu));

        let output = to_vgates.forward(&xs);

        // Expected output shape: [batch_size, heads, sequence_length, 1]
        assert_eq!(output.size(), vec![2, 4, 3, 1]);
    }

    #[test]
    fn test_to_out_rearrange() {
        let vs = VarStore::new(Device::Cpu);
        let to_out = ToOut::new(&vs.root(), 8, 4, 2, Some(0.1));

        // Input tensor: [batch_size, sequence_length, heads, dim_head]
        let xs = Tensor::randn(&[2, 3, 4, 2], (Kind::Float, Device::Cpu));

        let output = to_out.rearrange(&xs);

        // Expected output shape: [batch_size, sequence_length, heads * dim_head]
        assert_eq!(output.size(), vec![2, 3, 8]);
    }

    #[test]
    fn test_to_out_forward() {
        let vs = VarStore::new(Device::Cpu);
        let to_out = ToOut::new(&vs.root(), 8, 4, 2, Some(0.1));

        // Input tensor: [batch_size, sequence_length, heads, dim_head]
        let xs = Tensor::randn(&[2, 3, 4, 2], (Kind::Float, Device::Cpu));

        let output = to_out.forward_t(&xs, false);

        // Expected output shape: [batch_size, sequence_length, dim]
        assert_eq!(output.size(), vec![2, 3, 8]);
    }

    #[test]
    fn test_attention_forward_t_v1() {
        // 1. Prepare the VarStore and create Attention
        let vs = VarStore::new(Device::Cpu);
        let attention = Attention::new(&vs.root(), 256, None, None, None, Some(false), Some(false));

        // 2. Input Tensor: [batch_size=2, seq_len=3, dim=256]
        let xs = Tensor::randn(&[2, 3, 256], (Kind::Float, Device::Cpu));

        // 3. Forward pass
        let (out, ..) = attention
            .forward_t(&xs, None, false)
            .expect("forward_t failed");

        println!("Output shape: {:?}", out.size());
    }

    #[test]
    fn test_attention_forward_t_v2() {
        // 1. Prepare the VarStore and create Attention
        let vs = VarStore::new(Device::Cpu);
        let attention = Attention::new(&vs.root(), 256, None, None, None, Some(false), Some(false));

        // 2. Static Input Tensor: [batch_size=2, seq_len=3, dim=256]
        // Instead of random values, we'll use a static, known set of numbers.
        let data = (1..=2 * 3 * 256).map(|x| x as f32).collect::<Vec<_>>();

        let xs = Tensor::from_slice(&data)
            .reshape(&[2, 3, 256])
            .to_device(Device::Cpu);

        // Print the static tensor for verification (optional)
        println!("Input Tensor: {:?}", xs);

        // 3. Forward pass
        let (out, ..) = attention
            .forward_t(&xs, None, false)
            .expect("forward_t failed");

        // 4. Validate Output Shape
        println!("Output shape: {:?}", out.size());
        assert_eq!(out.size(), vec![2, 3, 256], "Output shape mismatch");

        // 5. Print Output Tensor (optional)
        println!("Output Tensor: {:?}", out);

        // 6. Print to string
        println!("{:?}", out.to_string(128));
    }

    // #[test]
    // fn test_itransformer() -> Result<()> {
    //     let vb = VarBuilder::from_varmap(&VarMap::new(), DType::F32, &DEVICE);
    //     let model = ITransformer::new(
    //         &vb,
    //         16,
    //         24,
    //         6,
    //         256,
    //         Some(1),
    //         vec![12, 24, 36, 48],
    //         Some(64),
    //         Some(8),
    //         None,
    //         None,
    //         None,
    //         None,
    //         Some(true),
    //         None,
    //         false,
    //         &DEVICE
    //     )?;
    //
    //     let time_series = Tensor::randn(0.0f32, 1.0f32, (2, 24, 16), &DEVICE)?.to_dtype(DType::F32)?;
    //     let preds = model.forward(&time_series, None)?;
    //
    //     println!("{:?}", preds);
    //
    //     Ok(())
    // }
}
