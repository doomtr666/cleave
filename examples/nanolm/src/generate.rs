//! What generation needs from the host: the prompt, encoded the way the corpus is
//! (`corpus::normalize`), and somewhere to print each character as it is drawn.

use std::io::Write;
use std::sync::OnceLock;

/// The context the model reads (`kernel.cleave`, `generate`).
const CONTEXT: usize = 128;

static PROMPT: OnceLock<Vec<i32>> = OnceLock::new();

/// Encodes `text` as the corpus is (typography folded, characters outside the alphabet dropped),
/// keeping its last 128 characters; an empty prompt starts a paragraph. Returns the prompt as the
/// model will see it.
pub fn set_prompt(text: &str) -> String {
    let normalized = crate::corpus::normalize(text);
    let mut prompt = normalized.trim_end_matches('\n').to_string();
    if prompt.is_empty() {
        prompt.push('\n');
    }
    let chars: Vec<char> = prompt.chars().collect();
    let kept: String = chars[chars.len().saturating_sub(CONTEXT)..].iter().collect();
    let ids = crate::corpus::encode(&kept).into_iter().map(i32::from).collect();
    PROMPT.set(ids).unwrap_or_else(|_| panic!("generate::set_prompt called twice"));
    kept
}

fn prompt() -> &'static [i32] {
    PROMPT.get().expect("generate::set_prompt not called yet")
}

#[unsafe(no_mangle)]
pub extern "C" fn prompt_len() -> i32 {
    prompt().len() as i32
}

/// # Safety
/// `out` must point to `len` writable `i32`s (a cleave `[i32; 128]`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn prompt_ids(out: *mut i32, len: i64) {
    let out = unsafe { std::slice::from_raw_parts_mut(out, len as usize) };
    out.fill(0);
    out[..prompt().len()].copy_from_slice(prompt());
}

#[unsafe(no_mangle)]
pub extern "C" fn emit_char(c: i32) {
    let alphabet = crate::corpus::alphabet();
    let mut stdout = std::io::stdout().lock();
    let _ = write!(stdout, "{}", alphabet[c as usize]);
    let _ = stdout.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_long_prompt_keeps_its_last_128_characters() {
        let text = "a".repeat(200) + "Gervaise";
        let kept = set_prompt(&text);
        assert_eq!(kept.chars().count(), 128);
        assert!(kept.ends_with("Gervaise"));
        assert_eq!(prompt_len(), 128);
    }
}
