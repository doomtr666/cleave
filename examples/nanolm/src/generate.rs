//! What generation needs from the host: the prompt, encoded the way the corpus is
//! (`corpus::normalize`, then `bpe.rs`), and somewhere to print each token as it is drawn.

use std::io::Write;
use std::sync::OnceLock;

/// The context the model reads (`kernel.cleave`, `generate`).
const CONTEXT: usize = crate::data::T;

static PROMPT: OnceLock<Vec<i32>> = OnceLock::new();

/// Encodes `text` as the corpus is (typography folded, characters outside the alphabet dropped,
/// then tokenized), keeping its last `CONTEXT` tokens; an empty prompt starts a paragraph. Returns the
/// prompt as the model will see it.
pub fn set_prompt(text: &str) -> String {
    let bpe = crate::data::bpe();
    let normalized = crate::corpus::normalize(text);
    let mut prompt = normalized.trim_end_matches('\n').to_string();
    if prompt.is_empty() {
        prompt.push('\n');
    }
    let tokens = bpe.encode(&crate::corpus::encode(&prompt));
    let kept = &tokens[tokens.len().saturating_sub(CONTEXT)..];
    PROMPT
        .set(kept.iter().map(|&t| i32::from(t)).collect())
        .unwrap_or_else(|_| panic!("generate::set_prompt called twice"));
    bpe.decode(kept)
}

fn prompt() -> &'static [i32] {
    PROMPT.get().expect("generate::set_prompt not called yet")
}

#[unsafe(no_mangle)]
pub extern "C" fn prompt_len() -> i32 {
    prompt().len() as i32
}

/// # Safety
/// `out` must point to `len` writable `i32`s (a cleave `[i32; CONTEXT]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prompt_ids(out: *mut i32, len: i64) {
    let out = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
    out.fill(0);
    out[..prompt().len()].copy_from_slice(prompt());
}

#[unsafe(no_mangle)]
pub extern "C" fn emit_token(t: i32) {
    let mut stdout = std::io::stdout().lock();
    let _ = write!(stdout, "{}", crate::data::bpe().decode(&[t as u16]));
    let _ = stdout.flush();
}
