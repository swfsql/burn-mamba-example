//! Logits post-processing and token sampling over plain `[f32]` slices.
//!
//! Mirrors the behaviour of `candle_transformers::generation::LogitsProcessor` and
//! `candle_transformers::utils::apply_repeat_penalty`, but without a tensor library:
//! logits leave the Burn tensor as a `Vec<f32>` and are sampled here directly.

use rand::SeedableRng;
use rand::distr::Distribution;

/// How the next token is picked from the logits.
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    /// Always take the highest logit.
    ArgMax,
    /// Sample from the full (temperature-scaled) distribution.
    All { temperature: f64 },
    /// Sample among the `k` most likely tokens.
    TopK { k: usize, temperature: f64 },
    /// Nucleus sampling: smallest set of tokens whose probability exceeds `p`.
    TopP { p: f64, temperature: f64 },
    /// Top-k, then nucleus sampling within it.
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

/// Turns a logits vector into a token id.
pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    /// From an explicit [Sampling] strategy.
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    /// A missing (or ~zero) `temperature` means [Sampling::ArgMax].
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    /// Picks the next token id from `logits`.
    pub fn sample(&mut self, logits: &[f32]) -> anyhow::Result<u32> {
        let next_token = match self.sampling.clone() {
            Sampling::ArgMax => sample_argmax(logits)?,
            Sampling::All { temperature } => {
                let prs = softmax(logits, temperature);
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = softmax(logits, temperature);
                if p <= 0.0 || p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = softmax(logits, temperature);
                self.sample_topk(&mut prs, k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = softmax(logits, temperature);
                self.sample_topk_topp(&mut prs, k, p as f32)?
            }
        };
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> anyhow::Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs)?;
        Ok(distr.sample(&mut self.rng) as u32)
    }

    /// Top-p ("nucleus") sampling: zero out everything outside the smallest set of
    /// tokens whose cumulated probability exceeds `top_p`, then sample.
    fn sample_topp(&mut self, prs: &mut [f32], top_p: f32) -> anyhow::Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        self.sample_multinomial(prs)
    }

    /// Samples among the `top_k` most likely tokens.
    fn sample_topk(&mut self, prs: &mut [f32], top_k: usize) -> anyhow::Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let sub_prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&sub_prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    /// Top-k, then top-p within the retained tokens.
    fn sample_topk_topp(&mut self, prs: &mut [f32], top_k: usize, top_p: f32) -> anyhow::Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut sub_prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = sub_prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&sub_prs)?
            } else {
                self.sample_topp(&mut sub_prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }
}

fn sample_argmax(logits: &[f32]) -> anyhow::Result<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, u), (_, v)| u.total_cmp(v))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| anyhow::anyhow!("empty logits"))
}

/// Numerically-stable softmax of `logits / temperature`.
fn softmax(logits: &[f32], temperature: f64) -> Vec<f32> {
    let temperature = temperature as f32;
    let mut prs: Vec<f32> = logits.iter().map(|&v| v / temperature).collect();
    let max = prs
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &v| if v > acc { v } else { acc });
    for p in prs.iter_mut() {
        *p = (*p - max).exp();
    }
    let sum: f32 = prs.iter().sum();
    if sum > 0. {
        for p in prs.iter_mut() {
            *p /= sum;
        }
    }
    prs
}

/// Divides (or multiplies, for negative logits) the logits of every token already
/// present in `context` by `penalty`, discouraging repetitions.
pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context: &[u32]) {
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if !already_seen.insert(*token_id) {
            continue;
        }
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
}
