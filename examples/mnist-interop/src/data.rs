//! Loads the real MNIST handwritten-digit dataset (28x28 grayscale, 10
//! classes, 60,000 training / 10,000 test images) -- cleave itself has no
//! file I/O or network access of its own (`doc/backlog.md`), so fetching and
//! parsing the real dataset happens entirely here, in Rust, before the
//! compiled cleave training loop ever runs. Same crossing-the-boundary shape
//! `examples/digits-interop/src/data.rs` already established (one scalar at
//! a time, through `extern fn` getters -- `rust_bindings.rs`'s own doc
//! comment: "only scalars cross this boundary today"), just scaled up (784
//! pixels instead of 64, real IDX binary files instead of a CSV already
//! checked into the repo).
//!
//! The four official IDX files (`train-images-idx3-ubyte`, `train-labels-
//! idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`) are
//! fetched once over HTTP (gzip-compressed at the source, ~11MB total) into
//! `.cache/` (`CARGO_MANIFEST_DIR`-relative, `.gitignore`d -- real binary
//! data, ~45MB decompressed, has no business in source control) and reused
//! on every subsequent run; `ureq`/`flate2` are the only two dependencies
//! this whole mechanism needs.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const MIRROR: &str = "https://raw.githubusercontent.com/fgnt/mnist/master";
const PIXELS_PER_IMAGE: usize = 28 * 28;

struct Dataset {
    /// Flat, row-major: sample `s`'s own pixel `p` is `pixels[s*784 + p]`.
    /// Normalized to `[0.0, 1.0]` (raw bytes are `0..=255`).
    pixels: Vec<f32>,
    labels: Vec<i32>,
}

impl Dataset {
    fn len(&self) -> usize {
        self.labels.len()
    }
}

/// Downloads `name.gz` from `MIRROR` into `cache_dir/name` (decompressed) if
/// it isn't already there -- checked by simple existence, not a checksum:
/// this is a fixed, well-known, versionless dataset, no update path to
/// worry about invalidating a stale cache.
fn fetch_cached(cache_dir: &Path, name: &str) -> PathBuf {
    let dest = cache_dir.join(name);
    if dest.exists() {
        return dest;
    }
    std::fs::create_dir_all(cache_dir)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", cache_dir.display()));
    let url = format!("{MIRROR}/{name}.gz");
    eprintln!("mnist-interop: downloading {url} ...");
    let mut response = ureq::get(&url)
        .call()
        .unwrap_or_else(|e| panic!("failed to download {url}: {e}"));
    let mut compressed = Vec::new();
    response
        .body_mut()
        .as_reader()
        .read_to_end(&mut compressed)
        .unwrap_or_else(|e| panic!("failed to read response body from {url}: {e}"));
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .unwrap_or_else(|e| panic!("failed to gunzip {url}: {e}"));
    std::fs::write(&dest, &raw)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", dest.display()));
    dest
}

/// Big-endian `u32` at `bytes[offset..offset+4]` -- every IDX header field's
/// own width, confirmed against the real format spec (Yann LeCun's own MNIST
/// page), not guessed.
fn read_u32_be(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

/// Parses an IDX3 image file (`magic=0x00000803`, then `n`/`rows`/`cols`,
/// then `n*rows*cols` raw pixel bytes, row-major) -- real format, checked
/// directly against the magic number rather than assumed.
fn parse_idx3_images(bytes: &[u8]) -> Vec<f32> {
    let magic = read_u32_be(bytes, 0);
    assert_eq!(magic, 0x0000_0803, "not a real IDX3 image file (bad magic)");
    let n = read_u32_be(bytes, 4) as usize;
    let rows = read_u32_be(bytes, 8) as usize;
    let cols = read_u32_be(bytes, 12) as usize;
    assert_eq!(
        rows * cols,
        PIXELS_PER_IMAGE,
        "expected 28x28 images, got {rows}x{cols}"
    );
    let data = &bytes[16..];
    assert_eq!(data.len(), n * PIXELS_PER_IMAGE, "truncated IDX3 file");
    data.iter().map(|&b| b as f32 / 255.0).collect()
}

/// Parses an IDX1 label file (`magic=0x00000801`, then `n`, then `n` raw
/// label bytes, each already `0..=9`).
fn parse_idx1_labels(bytes: &[u8]) -> Vec<i32> {
    let magic = read_u32_be(bytes, 0);
    assert_eq!(magic, 0x0000_0801, "not a real IDX1 label file (bad magic)");
    let n = read_u32_be(bytes, 4) as usize;
    let data = &bytes[8..];
    assert_eq!(data.len(), n, "truncated IDX1 file");
    data.iter().map(|&b| b as i32).collect()
}

fn load(cache_dir: &Path, images_name: &str, labels_name: &str) -> Dataset {
    let images_path = fetch_cached(cache_dir, images_name);
    let labels_path = fetch_cached(cache_dir, labels_name);
    let images_bytes = std::fs::read(&images_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", images_path.display()));
    let labels_bytes = std::fs::read(&labels_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", labels_path.display()));
    let mut pixels = parse_idx3_images(&images_bytes);
    let mut labels = parse_idx1_labels(&labels_bytes);
    assert_eq!(
        pixels.len() / PIXELS_PER_IMAGE,
        labels.len(),
        "image count doesn't match label count"
    );
    // TEMP bisection scaffold -- remove once done.
    if let Ok(cap) = std::env::var("MNIST_DEBUG_CAP") {
        let cap: usize = cap.parse().expect("MNIST_DEBUG_CAP must be an integer");
        labels.truncate(cap);
        pixels.truncate(cap * PIXELS_PER_IMAGE);
    }
    Dataset { pixels, labels }
}

static TRAIN: OnceLock<Dataset> = OnceLock::new();
static TEST: OnceLock<Dataset> = OnceLock::new();

/// Must be called once, before any of the `extern "C"` getters below are
/// ever invoked (i.e. before the compiled cleave training loop runs) --
/// `main.rs`'s own first step.
pub fn init(cache_dir: &str) {
    let cache_dir = Path::new(cache_dir);
    TRAIN
        .set(load(
            cache_dir,
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
        ))
        .unwrap_or_else(|_| panic!("data::init called twice"));
    TEST.set(load(
        cache_dir,
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ))
    .unwrap_or_else(|_| panic!("data::init called twice"));
}

#[unsafe(no_mangle)]
pub extern "C" fn train_len() -> i32 {
    TRAIN.get().expect("data::init not called yet").len() as i32
}

#[unsafe(no_mangle)]
pub extern "C" fn train_pixel(sample: i32, pixel: i32) -> f32 {
    let d = TRAIN.get().expect("data::init not called yet");
    d.pixels[sample as usize * PIXELS_PER_IMAGE + pixel as usize]
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
    d.pixels[sample as usize * PIXELS_PER_IMAGE + pixel as usize]
}

#[unsafe(no_mangle)]
pub extern "C" fn test_label(sample: i32) -> i32 {
    let d = TEST.get().expect("data::init not called yet");
    d.labels[sample as usize]
}

/// One-hot encoding, computed here rather than in cleave -- `digits-interop/
/// src/data.rs`'s own identical `train_target`/`test_target` doc comment for
/// why.
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
