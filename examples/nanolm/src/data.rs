//! Batches for the kernel: `B` windows of `T + 1` consecutive tokens (`bpe.rs`), drawn at random
//! offsets, the first `T` as inputs and the last `T` (shifted by one) as targets. Handed over
//! flattened row-major, one `extern` call per batch filling the array the kernel receives (the
//! host side takes it as a trailing `out` pointer and length).
//!
//! Batch `i` of a split is a pure function of `i`: offset `b` is `splitmix64(seed ^ (i * B + b))`
//! modulo the number of windows. A run resumed from a checkpoint at step `i` sees the batches it
//! would have seen, and the PyTorch twin (`bench/nanolm-pytorch`) draws the same ones.

use std::path::Path;
use std::sync::OnceLock;

// Sequences per batch (`B`), tokens per sequence (`T`) and tokens in the vocabulary (`VOCAB`): the
// kernel's `BATCH`, `CONTEXT` and `VOCAB`, read from it by `build.rs`.
include!(concat!(env!("OUT_DIR"), "/sizes.rs"));

pub const TRAIN_SEED: u64 = 0x5a01a_7a1;
pub const VAL_SEED: u64 = 0x5a01a_7a2;

struct Corpus {
    bpe: crate::bpe::Bpe,
    train: Vec<u16>,
    val: Vec<u16>,
    /// Characters per token over the validation text: a loss per token divided by it is a loss
    /// per character, comparable across tokenizers.
    chars_per_token: f64,
}

static CORPUS: OnceLock<Corpus> = OnceLock::new();

pub fn init(cache_dir: &str) {
    let (dir, chars, val_chars) = crate::corpus::load(Path::new(cache_dir));
    let (bpe, train, val) = crate::bpe::load(&dir, VOCAB, &chars, &val_chars);
    let chars_per_token = val_chars.len() as f64 / val.len() as f64;
    CORPUS
        .set(Corpus { bpe, train, val, chars_per_token })
        .unwrap_or_else(|_| panic!("data::init called twice"));
}

fn corpus() -> &'static Corpus {
    CORPUS.get().expect("data::init not called yet")
}

pub fn train() -> &'static [u16] {
    &corpus().train
}

pub fn val() -> &'static [u16] {
    &corpus().val
}

pub fn bpe() -> &'static crate::bpe::Bpe {
    &corpus().bpe
}

pub fn chars_per_token() -> f64 {
    corpus().chars_per_token
}

pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Where sequence `b` of batch `i` starts.
pub fn offset(text_len: usize, seed: u64, i: i32, b: usize) -> usize {
    let windows = (text_len - T) as u64;
    (splitmix64(seed ^ (i as u64 * B as u64 + b as u64)) % windows) as usize
}

/// Fills `out` (`B * T` ids) with batch `i`'s inputs (`shift` 0) or targets (`shift` 1).
fn fill(text: &[u16], seed: u64, i: i32, shift: usize, out: &mut [i32]) {
    assert_eq!(out.len(), B * T, "a batch is {B} x {T} ids");
    for (b, row) in out.chunks_exact_mut(T).enumerate() {
        let start = offset(text.len(), seed, i, b) + shift;
        for (o, &c) in row.iter_mut().zip(&text[start..start + T]) {
            *o = c as i32;
        }
    }
}

/// # Safety
/// `out` must point to `len` writable `i32`s.
unsafe fn out_slice<'a>(out: *mut i32, len: i64) -> &'a mut [i32] {
    unsafe { std::slice::from_raw_parts_mut(out, len as usize) }
}

/// # Safety
/// `out` must point to `len` writable `i32`s (a cleave `[i32; B * T]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_inputs(i: i32, out: *mut i32, len: i64) {
    fill(train(), TRAIN_SEED, i, 0, unsafe { out_slice(out, len) })
}

/// # Safety
/// As `train_inputs`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn train_targets(i: i32, out: *mut i32, len: i64) {
    fill(train(), TRAIN_SEED, i, 1, unsafe { out_slice(out, len) })
}

/// # Safety
/// As `train_inputs`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn val_inputs(i: i32, out: *mut i32, len: i64) {
    fill(val(), VAL_SEED, i, 0, unsafe { out_slice(out, len) })
}

/// # Safety
/// As `train_inputs`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn val_targets(i: i32, out: *mut i32, len: i64) {
    fill(val(), VAL_SEED, i, 1, unsafe { out_slice(out, len) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference values the PyTorch twin's own sampler is checked against.
    #[test]
    fn splitmix64_matches_the_reference() {
        assert_eq!(splitmix64(0), 0xe220_a839_7b1d_cdaf);
        assert_eq!(splitmix64(1), 0x910a_2dec_8902_5cc1);
    }

    #[test]
    fn targets_are_inputs_shifted_by_one() {
        // Longer than a sequence, whatever the kernel's `CONTEXT` (`T`).
        let text: Vec<u16> = (0..4 * T).map(|k| (k % 100) as u16).collect();
        let mut x = vec![0; B * T];
        let mut y = vec![0; B * T];
        fill(&text, 7, 3, 0, &mut x);
        fill(&text, 7, 3, 1, &mut y);
        for row in 0..B {
            for t in 0..T - 1 {
                assert_eq!(x[row * T + t + 1], y[row * T + t]);
            }
        }
    }
}
