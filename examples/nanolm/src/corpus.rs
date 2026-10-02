//! The training text: Zola's *Rougon-Macquart* (the 18 of 20 novels Project Gutenberg has in
//! French) and *Thérèse Raquin*, one author, one style, one world. Public domain everywhere (Zola
//! died in 1902), clean text (proofread, not raw OCR), ~15 MB: far more than a model of ~1 M
//! parameters can memorize.
//!
//! Each book is downloaded once into `.cache/gutenberg/` and cleaned: the Gutenberg header and
//! license are cut, the producers' credits and the ASCII family tree of *Le Docteur Pascal* are
//! dropped, paragraphs are rejoined onto one line each, and every character is mapped onto a fixed
//! alphabet of 104 (`ALPHABET`). The alphabet doesn't depend on the corpus, so another French
//! corpus encodes the same way and a checkpoint can be fine-tuned on it. The result is written
//! as one byte per character (`train.bin`, `val.bin`, `alphabet.txt`), which the PyTorch twin
//! (`bench/nanolm-pytorch`) reads as is.

use std::io::Read;
use std::path::{Path, PathBuf};

/// Project Gutenberg ebook numbers, in the cycle's order. *La Fortune des Rougon* and *La Joie de
/// vivre* aren't on Gutenberg in French; *Thérèse Raquin* (1867) precedes the cycle.
pub const TRAIN_BOOKS: &[(u32, &str)] = &[
    (7461, "Thérèse Raquin"),
    (17553, "La Curée"),
    (6470, "Le Ventre de Paris"),
    (8712, "La Conquête de Plassans"),
    (6558, "La Faute de l'abbé Mouret"),
    (17557, "Son Excellence Eugène Rougon"),
    (6497, "L'Assommoir"),
    (5250, "Nana"),
    (8907, "Pot-Bouille"),
    (16852, "Au Bonheur des Dames"),
    (5711, "Germinal"),
    (17517, "L'Œuvre"),
    (8563, "La Terre"),
    (17533, "Le Rêve"),
    (5154, "La Bête humaine"),
    (17516, "L'Argent"),
    (17831, "La Débâcle"),
    (8560, "Le Docteur Pascal"),
];

/// Held out whole, so that validation text is never seen in training, not even as the other half
/// of a paragraph.
pub const VAL_BOOKS: &[(u32, &str)] = &[(8561, "Une page d'amour")];

/// The model's characters; a character's id is its index. 104 = a multiple of 8, the row tile of
/// the output layer's matmul. `\n` separates paragraphs.
pub const ALPHABET: &str = concat!(
    "\n !'(),-.0123456789:;?",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "abcdefghijklmnopqrstuvwxyz",
    "«»",
    "ÀÂÇÈÉÊËÎÏÔÙÛŒ",
    "àâçèéêëîïôùûüÿœ",
);

pub fn alphabet() -> Vec<char> {
    ALPHABET.chars().collect()
}

/// The text between Gutenberg's `*** START OF ...` and `*** END OF ...` lines.
pub fn gutenberg_body(raw: &str) -> &str {
    let start = raw
        .find("*** START OF")
        .and_then(|s| raw[s..].find('\n').map(|n| s + n + 1))
        .expect("no Gutenberg START marker");
    let end = raw[start..]
        .find("*** END OF")
        .map(|e| start + e)
        .expect("no Gutenberg END marker");
    &raw[start..end]
}

/// One character as the alphabet spells it: typographic variants folded onto it, the rest
/// dropped. Returns the replacement text (empty to drop).
fn fold(c: char) -> &'static str {
    match c {
        '’' | '‘' | '`' => "'",
        '—' | '–' | '‐' => "-",
        '“' => "«",
        '”' => "»",
        '\u{a0}' | '\u{202f}' | '\t' => " ",
        '…' => "...",
        'Æ' => "AE",
        'æ' => "ae",
        // Gutenberg spells `_italics_`; quotation marks other than « » are too rare to learn.
        _ => "",
    }
}

/// Paragraphs that aren't Zola's: the producers' credits, transcribers' notes, tables drawn in
/// ASCII.
fn is_apparatus(paragraph: &str) -> bool {
    const MARKERS: &[&str] = &[
        "Produced by",
        "Proofread",
        "ebooksgratuits",
        "gallica",
        "http",
        "www.",
        "Gutenberg",
        "Distributed",
    ];
    MARKERS.iter().any(|m| paragraph.contains(m)) || paragraph.contains('|') || paragraph.contains("+--")
}

/// A book's body as the model sees it: one paragraph per line, single spaces, every character in
/// the alphabet.
pub fn normalize(body: &str) -> String {
    let alphabet = alphabet();
    let mut out = String::new();
    for paragraph in body.replace("\r\n", "\n").split("\n\n") {
        if is_apparatus(paragraph) {
            continue;
        }
        let mut line = String::new();
        for c in paragraph.chars() {
            let c = if c == '\n' { ' ' } else { c };
            if alphabet.contains(&c) {
                line.push(c);
            } else {
                line.push_str(fold(c));
            }
        }
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if !line.is_empty() {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out
}

/// Character ids; every character of `text` must be in the alphabet (`normalize`'s output is).
pub fn encode(text: &str) -> Vec<u8> {
    let alphabet = alphabet();
    text.chars()
        .map(|c| {
            alphabet
                .iter()
                .position(|&a| a == c)
                .unwrap_or_else(|| panic!("{c:?} is not in the alphabet")) as u8
        })
        .collect()
}

pub fn decode(ids: &[u8]) -> String {
    let alphabet = alphabet();
    ids.iter().map(|&i| alphabet[i as usize]).collect()
}

fn fetch_book(cache_dir: &Path, id: u32) -> String {
    let dir = cache_dir.join("gutenberg");
    let dest = dir.join(format!("pg{id}.txt"));
    if !dest.exists() {
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        let url = format!("https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt");
        eprintln!("nanolm: downloading {url} ...");
        let mut body = Vec::new();
        ureq::get(&url)
            .call()
            .unwrap_or_else(|e| panic!("cannot download {url}: {e}"))
            .body_mut()
            .with_config()
            .limit(16 << 20)
            .reader()
            .read_to_end(&mut body)
            .unwrap_or_else(|e| panic!("cannot read {url}: {e}"));
        std::fs::write(&dest, &body).unwrap_or_else(|e| panic!("cannot write {}: {e}", dest.display()));
    }
    std::fs::read_to_string(&dest).unwrap_or_else(|e| panic!("cannot read {}: {e}", dest.display()))
}

fn encode_books(cache_dir: &Path, books: &[(u32, &str)]) -> Vec<u8> {
    books
        .iter()
        .flat_map(|&(id, _)| encode(&normalize(gutenberg_body(&fetch_book(cache_dir, id)))))
        .collect()
}

/// The encoded corpus, built on first use: `(train, val)` ids.
pub fn load(cache_dir: &Path) -> (Vec<u8>, Vec<u8>) {
    let dir = cache_dir.join("zola");
    let train_path: PathBuf = dir.join("train.bin");
    let val_path = dir.join("val.bin");
    if !train_path.exists() || !val_path.exists() {
        let train = encode_books(cache_dir, TRAIN_BOOKS);
        let val = encode_books(cache_dir, VAL_BOOKS);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        std::fs::write(dir.join("alphabet.txt"), ALPHABET).unwrap();
        std::fs::write(&val_path, &val).unwrap();
        // Written last: its presence means the corpus is complete.
        std::fs::write(&train_path, &train).unwrap();
    }
    (std::fs::read(&train_path).unwrap(), std::fs::read(&val_path).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_alphabet_has_104_distinct_characters() {
        let a = alphabet();
        assert_eq!(a.len(), 104);
        let mut sorted = a.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 104);
    }

    #[test]
    fn the_body_is_cut_out_of_the_gutenberg_frame() {
        let raw = "header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nLe texte.\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlicense";
        assert_eq!(gutenberg_body(raw), "Le texte.\n");
    }

    #[test]
    fn paragraphs_are_rejoined_and_typography_folded() {
        let body = "Produced by Someone and the Online\nDistributed Proofreading Team.\n\n\
                    --Tiens, dit l’évêque,\nj’y songe.  _Vraiment._\n\n\
                    +-----+\n| arbre |\n+-----+\n\n\
                    Il dit: “oui”\u{a0}!";
        assert_eq!(
            normalize(body),
            "--Tiens, dit l'évêque, j'y songe. Vraiment.\nIl dit: «oui» !\n"
        );
    }

    #[test]
    fn encoding_round_trips() {
        let text = normalize("Œuvre à Nana, ça «brûle» ? Oui… 1867.\n");
        assert_eq!(decode(&encode(&text)), text);
    }
}
