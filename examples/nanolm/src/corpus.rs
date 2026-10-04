//! The training text: French literature of the 19th century and the turn of the 20th, the 1537
//! Project Gutenberg books of `books.txt` (Sand, Dumas, Verne, Maupassant, Zola, Hugo, Balzac,
//! Flaubert...), a model that speaks French before it is fine-tuned on anything narrower.
//! Public domain everywhere, clean text (proofread, not raw OCR).
//!
//! Each book is downloaded once into `.cache/gutenberg/` and cleaned: the Gutenberg header and
//! license are cut, the producers' credits and ASCII tables are dropped, paragraphs are rejoined
//! onto one line each, and every character is mapped onto a fixed alphabet of 104 (`ALPHABET`).
//! The alphabet doesn't depend on the corpus, so another French corpus encodes the same way and a
//! checkpoint can be fine-tuned on it. About one book in fifty, picked by its number
//! (`is_validation`), is held out whole for validation. The result is written as one byte per
//! character (`.cache/french/train.bin`, `val.bin`, `alphabet.txt`), which the PyTorch twin
//! (`bench/nanolm-pytorch`) reads as is.

use std::io::Read;
use std::path::{Path, PathBuf};

/// One line of `books.txt`.
pub struct Book {
    pub id: u32,
    pub author: &'static str,
    pub title: &'static str,
}

/// The books of `books.txt`, in ebook-number order.
pub fn books() -> Vec<Book> {
    include_str!("../books.txt")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut f = l.split('\t');
            let id = f.next().unwrap().parse().unwrap_or_else(|_| panic!("bad line in books.txt: {l:?}"));
            let author = f.next().unwrap_or_else(|| panic!("no author in books.txt: {l:?}"));
            let title = f.next().unwrap_or_else(|| panic!("no title in books.txt: {l:?}"));
            Book { id, author, title }
        })
        .collect()
}

const VALIDATION_SEED: u64 = 0x5a01a_7a3;

/// Whether a book is held out for validation: never seen in training, not even as the other half
/// of a paragraph. A function of its number alone, so adding books doesn't reshuffle the split.
pub fn is_validation(id: u32) -> bool {
    crate::data::splitmix64(VALIDATION_SEED ^ u64::from(id)) % 50 == 0
}

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

/// The book's text inside Gutenberg's frame: between the `*** START OF ...` and `*** END OF ...`
/// lines, or, in the oldest files, after the `*END*THE SMALL PRINT!` license and before `End of
/// [the] Project Gutenberg ...`. `None` when the frame isn't recognized.
pub fn gutenberg_body(raw: &str) -> Option<&str> {
    let after_line = |s: usize| raw[s..].find('\n').map(|n| s + n + 1);
    let start = ["*** START OF", "***START OF", "*END*THE SMALL PRINT"]
        .iter()
        .find_map(|m| raw.find(m))
        .and_then(after_line)?;
    let end = ["*** END OF", "***END OF", "End of the Project Gutenberg", "End of Project Gutenberg"]
        .iter()
        .filter_map(|m| raw[start..].find(m))
        .min()?;
    Some(&raw[start..start + end])
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

/// Paragraphs that aren't the book's: the producers' and digitizers' credits, transcribers' notes,
/// illustration captions (`[Illustration: ...]`), tables drawn in ASCII. Markers are matched
/// ignoring case (`GUTENBERG` in the older files' license lines).
fn is_apparatus(paragraph: &str) -> bool {
    const MARKERS: &[&str] = &[
        "produced by",
        "proofread",
        "ebooksgratuits",
        "gallica",
        "http",
        "www.",
        "gutenberg",
        "distributed",
        "copyright",
        "numérisé",
        "bibliothèque nationale",
        "note du transcripteur",
        "[illustration",
    ];
    let lower = paragraph.to_lowercase();
    MARKERS.iter().any(|m| lower.contains(m)) || paragraph.contains('|') || paragraph.contains("+--")
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

#[cfg(test)]
pub fn decode(ids: &[u8]) -> String {
    let alphabet = alphabet();
    ids.iter().map(|&i| alphabet[i as usize]).collect()
}

/// The raw text of ebook `id`, downloaded on first use; `None` (with a warning) if it can't be.
fn fetch_book(cache_dir: &Path, id: u32) -> Option<String> {
    let dir = cache_dir.join("gutenberg");
    let dest = dir.join(format!("pg{id}.txt"));
    if !dest.exists() {
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        let url = format!("https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt");
        eprintln!("nanolm: downloading {url} ...");
        let mut body = Vec::new();
        let read = ureq::get(&url).call().map_err(|e| e.to_string()).and_then(|mut r| {
            r.body_mut()
                .with_config()
                .limit(64 << 20)
                .reader()
                .read_to_end(&mut body)
                .map_err(|e| e.to_string())
        });
        // Gutenberg asks bulk downloaders to pace themselves.
        std::thread::sleep(std::time::Duration::from_secs(1));
        if let Err(e) = read {
            eprintln!("nanolm: skipping ebook {id}, cannot download {url}: {e}");
            return None;
        }
        std::fs::write(&dest, &body).unwrap_or_else(|e| panic!("cannot write {}: {e}", dest.display()));
    }
    let bytes = std::fs::read(&dest).unwrap_or_else(|e| panic!("cannot read {}: {e}", dest.display()));
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// The encoded corpus, built on first use: its directory (`.cache/french`), the train and the
/// validation character ids.
pub fn load(cache_dir: &Path) -> (PathBuf, Vec<u8>, Vec<u8>) {
    let dir = cache_dir.join("french");
    let train_path: PathBuf = dir.join("train.bin");
    let val_path = dir.join("val.bin");
    if !train_path.exists() || !val_path.exists() {
        let (mut train, mut val) = (Vec::new(), Vec::new());
        let books = books();
        for (k, book) in books.iter().enumerate() {
            if let Some(raw) = fetch_book(cache_dir, book.id) {
                match gutenberg_body(&raw) {
                    Some(body) => {
                        let ids = encode(&normalize(body));
                        if is_validation(book.id) { &mut val } else { &mut train }.extend(ids);
                    }
                    None => eprintln!(
                        "nanolm: skipping ebook {} ({}, {}), no Gutenberg frame",
                        book.id, book.author, book.title
                    ),
                }
            }
            if (k + 1) % 100 == 0 {
                eprintln!("nanolm: {} / {} books", k + 1, books.len());
            }
        }
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        std::fs::write(dir.join("alphabet.txt"), ALPHABET).unwrap();
        std::fs::write(&val_path, &val).unwrap();
        // Written last: its presence means the corpus is complete.
        std::fs::write(&train_path, &train).unwrap();
    }
    (dir, std::fs::read(&train_path).unwrap(), std::fs::read(&val_path).unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_book_list_parses_and_about_one_in_fifty_is_held_out() {
        let books = books();
        assert_eq!(books.len(), 1537);
        let mut ids: Vec<u32> = books.iter().map(|b| b.id).collect();
        ids.dedup();
        assert_eq!(ids.len(), books.len(), "books.txt is sorted, without duplicates");
        assert!(books.iter().all(|b| !b.author.is_empty() && !b.title.is_empty()));
        let held_out = books.iter().filter(|b| is_validation(b.id)).count();
        assert!((15..=50).contains(&held_out), "{held_out} validation books");
    }

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
        assert_eq!(gutenberg_body(raw), Some("Le texte.\n"));
        let old = "license\n*END*THE SMALL PRINT! FOR PUBLIC DOMAIN EBOOKS*Ver.04.29.93*END*\nLe texte.\nEnd of the Project Gutenberg EBook of X\n";
        assert_eq!(gutenberg_body(old), Some("Le texte.\n"));
        assert_eq!(gutenberg_body("no frame at all"), None);
    }

    #[test]
    fn paragraphs_are_rejoined_and_typography_folded() {
        let body = "Produced by Someone and the Online\nDistributed Proofreading Team.\n\n\
                    --Tiens, dit l’évêque,\nj’y songe.  _Vraiment._\n\n\
                    +-----+\n| arbre |\n+-----+\n\n\
                    [Illustration: LE DÉPART]\n\n\
                    END OF THE PROJECT GUTENBERG ETEXT\n\n\
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
