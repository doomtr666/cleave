//! Loads the UCI "Optical Recognition of Handwritten Digits" dataset
//! (`optdigits.tra`/`optdigits.tes`, plain CSV: 64 pixel columns [0-16],
//! then a trailing label column [0-9]) via ordinary Rust `std::fs` -- cleave
//! itself has no file I/O at all (`doc/backlog.md`), so reading the real
//! dataset off disk happens entirely here, in Rust, before the compiled
//! cleave training loop ever runs. Data crosses into cleave one scalar at a
//! time, through `extern fn` getters (`export fn`'s own boundary is scalar-
//! only too, but the *call direction* matters here: cleave calling *out* to
//! fetch one pixel already has a real, array-argument-capable ABI —
//! `mlir_lower.rs`'s own array-aware extern lowering — but passing a whole
//! 64-pixel image or the full dataset *in* through `export fn` doesn't;
//! `rust_bindings.rs`'s own doc comment: "only scalars cross this boundary
//! today"), not one bulk transfer -- the simplest mechanism that actually
//! fits both boundaries' real capabilities today.

use std::sync::OnceLock;

struct Dataset {
    /// Flat, row-major: sample `s`'s own pixel `p` is `pixels[s*64 + p]`.
    /// Normalized to `[0.0, 1.0]` (raw values are `0..=16`) -- ordinary
    /// input scaling, not a cleave-specific concern.
    pixels: Vec<f32>,
    labels: Vec<i32>,
}

impl Dataset {
    fn load(path: &str) -> Self {
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
        let mut pixels = Vec::new();
        let mut labels = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let nums: Vec<i32> = line
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse()
                        .unwrap_or_else(|e| panic!("{path}:{}: bad number {s:?}: {e}", lineno + 1))
                })
                .collect();
            assert_eq!(
                nums.len(),
                65,
                "{path}:{}: expected 65 columns (64 pixels + 1 label), got {}",
                lineno + 1,
                nums.len()
            );
            pixels.extend(nums[..64].iter().map(|&p| p as f32 / 16.0));
            labels.push(nums[64]);
        }
        Dataset { pixels, labels }
    }

    fn len(&self) -> usize {
        self.labels.len()
    }
}

static TRAIN: OnceLock<Dataset> = OnceLock::new();
static TEST: OnceLock<Dataset> = OnceLock::new();

/// Must be called once, before any of the `extern "C"` getters below are
/// ever invoked (i.e. before the compiled cleave training loop runs) --
/// `main.rs`'s own first step.
pub fn init(train_path: &str, test_path: &str) {
    TRAIN
        .set(Dataset::load(train_path))
        .unwrap_or_else(|_| panic!("data::init called twice"));
    TEST.set(Dataset::load(test_path))
        .unwrap_or_else(|_| panic!("data::init called twice"));
}

#[unsafe(no_mangle)]
pub extern "C" fn train_len() -> i32 {
    TRAIN.get().expect("data::init not called yet").len() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn train_pixel(sample: i32, pixel: i32) -> f32 {
    let d = TRAIN.get().expect("data::init not called yet");
    d.pixels[sample as usize * 64 + pixel as usize]
}

#[unsafe(no_mangle)]
pub extern "C" fn train_label(sample: i32) -> i32 {
    let d = TRAIN.get().expect("data::init not called yet");
    d.labels[sample as usize]
}

#[unsafe(no_mangle)]
pub extern "C" fn test_len() -> i32 {
    TEST.get().expect("data::init not called yet").len() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn test_pixel(sample: i32, pixel: i32) -> f32 {
    let d = TEST.get().expect("data::init not called yet");
    d.pixels[sample as usize * 64 + pixel as usize]
}

#[unsafe(no_mangle)]
pub extern "C" fn test_label(sample: i32) -> i32 {
    let d = TEST.get().expect("data::init not called yet");
    d.labels[sample as usize]
}

/// One-hot encoding, computed here rather than in cleave — `1.0` if `sample`
/// 's own label equals `class`, else `0.0`. Keeps the cleave side purely
/// numeric (read floats, do math) instead of needing conditional one-hot
/// construction logic of its own.
#[unsafe(no_mangle)]
pub extern "C" fn train_target(sample: i32, class: i32) -> f32 {
    let d = TRAIN.get().expect("data::init not called yet");
    if d.labels[sample as usize] == class { 1.0 } else { 0.0 }
}

#[unsafe(no_mangle)]
pub extern "C" fn test_target(sample: i32, class: i32) -> f32 {
    let d = TEST.get().expect("data::init not called yet");
    if d.labels[sample as usize] == class { 1.0 } else { 0.0 }
}
