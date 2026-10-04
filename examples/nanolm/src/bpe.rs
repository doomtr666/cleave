//! Byte-pair encoding over the corpus alphabet (`corpus::ALPHABET`): the first tokens are the 104
//! characters themselves, each further token the merge of two earlier ones, learned from the
//! training text (most frequent adjacent pair first, ties to the smallest pair, so training is
//! deterministic). Merges never cross a piece boundary (`pieces`): a word with the space before it,
//! a run of digits, a run of punctuation, a line break. Encoding applies the learned merges in the
//! order they were learned, as GPT-2's tokenizer does.
//!
//! The tokenized corpus is written next to the character one, in `.cache/french/bpe{V}/`:
//! `merges.txt` (one `left right` pair of token ids per line, token `104 + k` being line `k`),
//! `train.bin` and `val.bin` (little-endian `u16` ids), which the PyTorch twin reads as is.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

/// FxHash (rustc's): far faster than SipHash on the hundreds of millions of short keys hashed here.
#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.0 = (self.0.rotate_left(5) ^ u64::from_le_bytes(word)).wrapping_mul(0x517c_c1b7_2722_0a95);
        }
    }
}

type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

type Pair = (u16, u16);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Newline,
    Space,
    Word,
    Digit,
    Punct,
}

fn classes(alphabet: &[char]) -> Vec<Class> {
    alphabet
        .iter()
        .map(|&c| match c {
            '\n' => Class::Newline,
            ' ' => Class::Space,
            '\'' => Class::Word,
            c if c.is_alphabetic() => Class::Word,
            c if c.is_ascii_digit() => Class::Digit,
            _ => Class::Punct,
        })
        .collect()
}

/// Calls `f` on each piece of `text` (character ids), in order: a line break alone; otherwise an
/// optional space followed by a run of one class (letters and apostrophes, digits, punctuation).
fn pieces<'a>(text: &'a [u8], classes: &[Class], mut f: impl FnMut(&'a [u8])) {
    let class = |i: usize| classes[text[i] as usize];
    let mut i = 0;
    while i < text.len() {
        let start = i;
        match class(i) {
            Class::Newline => i += 1,
            c => {
                if c == Class::Space {
                    i += 1;
                }
                if i < text.len() && !matches!(class(i), Class::Newline | Class::Space) {
                    let run = class(i);
                    while i < text.len() && class(i) == run {
                        i += 1;
                    }
                }
            }
        }
        f(&text[start..i]);
    }
}

pub struct Bpe {
    alphabet: Vec<char>,
    classes: Vec<Class>,
    /// Token `alphabet.len() + k` is `merges[k].0` followed by `merges[k].1`.
    merges: Vec<Pair>,
    /// The token a pair merges into; also its rank, merges being numbered in learning order.
    merged: FxMap<Pair, u16>,
}

impl Bpe {
    fn new(alphabet: &[char], merges: Vec<Pair>) -> Bpe {
        let base = alphabet.len();
        let merged = merges.iter().enumerate().map(|(k, &p)| (p, (base + k) as u16)).collect();
        Bpe { alphabet: alphabet.to_vec(), classes: classes(alphabet), merges, merged }
    }

    pub fn vocab_size(&self) -> usize {
        self.alphabet.len() + self.merges.len()
    }

    /// Learns `vocab - alphabet.len()` merges from `text` (character ids), or fewer if the text
    /// runs out of pairs.
    pub fn train(text: &[u8], alphabet: &[char], vocab: usize) -> Bpe {
        assert!(vocab <= 1 << 16, "token ids are u16");
        let base = alphabet.len();
        let classes = classes(alphabet);

        let mut counts: FxMap<&[u8], i64> = FxMap::default();
        pieces(text, &classes, |p| *counts.entry(p).or_default() += 1);
        // Sorted, so that word indices (and with them everything below) don't depend on hashing.
        let mut unique: Vec<(&[u8], i64)> = counts.into_iter().collect();
        unique.sort_unstable();
        let mut words: Vec<(Vec<u16>, i64)> =
            unique.into_iter().map(|(p, n)| (p.iter().map(|&c| u16::from(c)).collect(), n)).collect();

        let mut pair_count: FxMap<Pair, i64> = FxMap::default();
        let mut containing: FxMap<Pair, Vec<u32>> = FxMap::default();
        for (w, (symbols, n)) in words.iter().enumerate() {
            for p in symbols.windows(2) {
                *pair_count.entry((p[0], p[1])).or_default() += n;
                containing.entry((p[0], p[1])).or_default().push(w as u32);
            }
        }
        let mut heap: BinaryHeap<(i64, Reverse<Pair>)> = pair_count.iter().map(|(&p, &n)| (n, Reverse(p))).collect();

        let mut merges = Vec::new();
        while base + merges.len() < vocab {
            let Some((n, Reverse(pair))) = heap.pop() else { break };
            // Entries are pushed again whenever a count changes; only the current one counts.
            if n <= 0 || pair_count.get(&pair) != Some(&n) {
                continue;
            }
            let token = (base + merges.len()) as u16;
            merges.push(pair);
            let mut ws = containing.remove(&pair).unwrap_or_default();
            ws.sort_unstable();
            ws.dedup();
            let mut touched: FxMap<Pair, ()> = FxMap::default();
            for w in ws {
                let (symbols, n) = &mut words[w as usize];
                if !symbols.windows(2).any(|p| (p[0], p[1]) == pair) {
                    continue;
                }
                for p in symbols.windows(2) {
                    *pair_count.get_mut(&(p[0], p[1])).unwrap() -= *n;
                    touched.insert((p[0], p[1]), ());
                }
                *symbols = merge(symbols, pair, token);
                for p in symbols.windows(2) {
                    *pair_count.entry((p[0], p[1])).or_default() += *n;
                    containing.entry((p[0], p[1])).or_default().push(w);
                    touched.insert((p[0], p[1]), ());
                }
            }
            for (p, ()) in touched {
                let n = pair_count[&p];
                if n > 0 {
                    heap.push((n, Reverse(p)));
                }
            }
        }
        Bpe::new(alphabet, merges)
    }

    /// One piece's tokens: the learned merges applied in learning order.
    fn encode_piece(&self, piece: &[u8]) -> Vec<u16> {
        let mut symbols: Vec<u16> = piece.iter().map(|&c| u16::from(c)).collect();
        loop {
            let best = symbols
                .windows(2)
                .filter_map(|p| self.merged.get(&(p[0], p[1])).map(|&t| (t, (p[0], p[1]))))
                .min();
            let Some((token, pair)) = best else { return symbols };
            symbols = merge(&symbols, pair, token);
        }
    }

    /// The tokens of `text` (character ids).
    pub fn encode(&self, text: &[u8]) -> Vec<u16> {
        let mut cache: FxMap<&[u8], Vec<u16>> = FxMap::default();
        let mut out = Vec::with_capacity(text.len() / 3);
        pieces(text, &self.classes, |p| {
            out.extend_from_slice(cache.entry(p).or_insert_with(|| self.encode_piece(p)));
        });
        out
    }

    /// The character ids a token stands for.
    pub fn expand(&self, token: u16, out: &mut Vec<u8>) {
        match (token as usize).checked_sub(self.alphabet.len()) {
            None => out.push(token as u8),
            Some(k) => {
                let (a, b) = self.merges[k];
                self.expand(a, out);
                self.expand(b, out);
            }
        }
    }

    pub fn decode(&self, tokens: &[u16]) -> String {
        let mut ids = Vec::new();
        for &t in tokens {
            self.expand(t, &mut ids);
        }
        ids.iter().map(|&c| self.alphabet[c as usize]).collect()
    }

    fn save(&self, path: &Path) {
        let text: String = self.merges.iter().map(|(a, b)| format!("{a} {b}\n")).collect();
        std::fs::write(path, text).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
    }

    fn read(path: &Path, alphabet: &[char]) -> Bpe {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let merges = text
            .lines()
            .map(|l| {
                let (a, b) = l.split_once(' ').unwrap_or_else(|| panic!("bad merge line {l:?}"));
                (a.parse().unwrap(), b.parse().unwrap())
            })
            .collect();
        Bpe::new(alphabet, merges)
    }
}

/// `symbols` with every non-overlapping occurrence of `pair`, left to right, replaced by `token`.
fn merge(symbols: &[u16], pair: Pair, token: u16) -> Vec<u16> {
    let mut out = Vec::with_capacity(symbols.len());
    let mut i = 0;
    while i < symbols.len() {
        if i + 1 < symbols.len() && (symbols[i], symbols[i + 1]) == pair {
            out.push(token);
            i += 2;
        } else {
            out.push(symbols[i]);
            i += 1;
        }
    }
    out
}

fn write_u16(path: &Path, tokens: &[u16]) {
    let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

fn read_u16(path: &Path) -> Vec<u16> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
}

/// The tokenizer and the tokenized corpus, `(bpe, train, val)`, built on first use from the
/// character corpus in `dir` (`.cache/french`).
pub fn load(dir: &Path, vocab: usize, train: &[u8], val: &[u8]) -> (Bpe, Vec<u16>, Vec<u16>) {
    let alphabet = crate::corpus::alphabet();
    let out = dir.join(format!("bpe{vocab}"));
    let (merges, train_path, val_path) = (out.join("merges.txt"), out.join("train.bin"), out.join("val.bin"));
    if !train_path.exists() || !val_path.exists() || !merges.exists() {
        std::fs::create_dir_all(&out).unwrap_or_else(|e| panic!("cannot create {}: {e}", out.display()));
        eprintln!("nanolm: learning a {vocab}-token BPE ...");
        let bpe = Bpe::train(train, &alphabet, vocab);
        bpe.save(&merges);
        eprintln!("nanolm: tokenizing the corpus ...");
        write_u16(&val_path, &bpe.encode(val));
        // Written last: its presence means the tokenized corpus is complete.
        write_u16(&train_path, &bpe.encode(train));
    }
    (Bpe::read(&merges, &alphabet), read_u16(&train_path), read_u16(&val_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::{alphabet, encode};

    fn piece_strings(text: &str) -> Vec<String> {
        let a = alphabet();
        let ids = encode(text);
        let mut out = Vec::new();
        pieces(&ids, &classes(&a), |p| out.push(p.iter().map(|&c| a[c as usize]).collect()));
        out
    }

    #[test]
    fn text_is_cut_into_words_with_their_space() {
        assert_eq!(
            piece_strings("--Il dit: l'homme, en 1867...\nOui"),
            ["--", "Il", " dit", ":", " l'homme", ",", " en", " 1867", "...", "\n", "Oui"]
        );
    }

    const SAMPLE: &str = "Le chat dort. Le chien dort. Le chat mange, le chien aussi.\n\
                          Les chats et les chiens dorment dans la cour; le chat, le chien.\n";

    #[test]
    fn training_is_deterministic_and_stops_at_the_vocabulary_size() {
        let a = alphabet();
        let text = encode(&SAMPLE.repeat(20));
        let one = Bpe::train(&text, &a, 130);
        let two = Bpe::train(&text, &a, 130);
        assert_eq!(one.merges, two.merges);
        assert_eq!(one.vocab_size(), 130);
        // The most frequent pair in this text: " c" (` chat`, ` chien`, ...).
        assert_eq!(one.decode(&[a.len() as u16]), " c");
    }

    #[test]
    fn encoding_round_trips_and_compresses() {
        let a = alphabet();
        let text = encode(&SAMPLE.repeat(20));
        let bpe = Bpe::train(&text, &a, 200);
        let unseen = encode("Le chien dort dans la cour, le chat aussi. Zola écrit.\n");
        let tokens = bpe.encode(&unseen);
        assert_eq!(bpe.decode(&tokens), "Le chien dort dans la cour, le chat aussi. Zola écrit.\n");
        assert!(tokens.len() * 2 < unseen.len(), "{} tokens for {} characters", tokens.len(), unseen.len());
    }

    #[test]
    fn no_token_crosses_a_piece_boundary() {
        let a = alphabet();
        let bpe = Bpe::train(&encode(&SAMPLE.repeat(20)), &a, 300);
        for t in 0..bpe.vocab_size() as u16 {
            let s = bpe.decode(&[t]);
            assert_eq!(piece_strings(&s).len(), 1, "token {t} is {s:?}");
        }
    }

    #[test]
    fn merges_survive_saving_and_reading() {
        let a = alphabet();
        let bpe = Bpe::train(&encode(&SAMPLE.repeat(20)), &a, 150);
        let path = std::env::temp_dir().join(format!("nanolm-bpe-test-{}.txt", std::process::id()));
        bpe.save(&path);
        let read = Bpe::read(&path, &a);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(read.merges, bpe.merges);
        let ids = encode(SAMPLE);
        assert_eq!(read.encode(&ids), bpe.encode(&ids));
    }
}
