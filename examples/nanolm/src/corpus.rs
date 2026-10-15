//! The training text: written French in the public domain, every subject and genre, from two
//! sources. Project Gutenberg's French texts (`books.txt`: 4189 of them, literature, history,
//! essays, periodicals, translations), proofread. And Gallica's books (the Bibliothèque nationale
//! de France) as PleIAs published them (`French-PD-Books`, `PLEIAS_FILES`), raw OCR, kept only
//! where it reads well (`pleias_books`): about 2 billion tokens beside Gutenberg's 0.4.
//!
//! Each source is downloaded once into `.cache/` (`gutenberg/`, `pleias/`) and cleaned: Gutenberg's
//! header and license cut, the OCR's page numbers dropped and its lines rejoined into paragraphs,
//! the producers' credits and ASCII tables dropped, each paragraph onto one line, and every
//! character mapped onto a fixed alphabet of 104 (`ALPHABET`). The alphabet doesn't depend on the
//! corpus, so another French corpus encodes the same way and a checkpoint can be fine-tuned on
//! it. About one book in fifty, picked by its identifier (`is_validation`), is held out whole for
//! validation. The result is written as one byte per character (`.cache/french/train.bin`,
//! `val.bin`, `alphabet.txt`), streamed to disk as it's built (about 8 GB), which the tokenizer
//! (`bpe.rs`) and the PyTorch twin (`bench/nanolm-pytorch`) read.

use std::io::{Read, Write};
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
/// of a paragraph. A function of its identifier alone (a Gutenberg number, or a hash of a Gallica
/// one, `gallica_key`), so adding books doesn't reshuffle the split.
pub fn is_validation(id: u64) -> bool {
    crate::data::splitmix64(VALIDATION_SEED ^ id) % 50 == 0
}

/// A Gallica identifier (`bpt6k57852716`) as a number for `is_validation`: its FNV-1a hash, high
/// bit set so that it never meets a Gutenberg number.
fn gallica_key(file_id: &str) -> u64 {
    let hash = file_id.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3));
    hash | (1 << 63)
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
        // The old `et` (`&c.`, and every `et` of the older OCR'd books).
        '&' => "et",
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

/// A Gallica book's OCR text as paragraphs separated by blank lines, for `normalize`: its HTML
/// entities decoded, the lines with fewer than three letters dropped (page numbers, printers'
/// signatures: `( 6)`, `A3`), and the others rejoined, a paragraph ending with a line that ends a
/// sentence (`.`, `!`, `?`, `:`, `»`). The OCR keeps the printed lines, not the paragraphs.
pub fn ocr_paragraphs(text: &str) -> String {
    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'");
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let line = line.trim();
        if line.chars().filter(|c| c.is_alphabetic()).count() < 3 {
            continue;
        }
        out.push_str(line);
        out.push_str(if line.ends_with(['.', '!', '?', ':', '»']) { "\n\n" } else { " " });
    }
    out
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

/// The files of PleIAs' `French-PD-Books` the corpus takes (of its 145, about 400 MB each, 2000
/// books and 100 million tokens kept each): enough tokens for a model of about a hundred million
/// parameters (some twenty per parameter), more than its runs will see.
pub const PLEIAS_FILES: usize = 20;
const PLEIAS_URL: &str = "https://huggingface.co/datasets/PleIAs/French-PD-Books/resolve/main";

/// PleIAs file `k` (`gallica_mono_{k}.parquet`), downloaded on first use.
fn fetch_pleias(cache_dir: &Path, k: usize) -> PathBuf {
    let dir = cache_dir.join("pleias");
    let dest = dir.join(format!("gallica_mono_{k}.parquet"));
    if !dest.exists() {
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
        let url = format!("{PLEIAS_URL}/gallica_mono_{k}.parquet");
        eprintln!("nanolm: downloading {url} ...");
        let partial = dest.with_extension("partial");
        let mut response = ureq::get(&url).call().unwrap_or_else(|e| panic!("cannot download {url}: {e}"));
        let mut file = std::fs::File::create(&partial).unwrap_or_else(|e| panic!("cannot write {}: {e}", partial.display()));
        std::io::copy(&mut response.body_mut().with_config().limit(u64::MAX).reader(), &mut file)
            .unwrap_or_else(|e| panic!("cannot download {url}: {e}"));
        drop(file);
        std::fs::rename(&partial, &dest).unwrap_or_else(|e| panic!("cannot write {}: {e}", dest.display()));
    }
    dest
}

/// The books of a PleIAs file that read well, `(Gallica identifier, text)`: printed from 1800 on
/// (before it, the OCR reads the long s as an f), at least 5000 words, OCR confidence of at least
/// 90 (Gallica's own), at least 92% of their words spelled as words (the confidence alone lets
/// tables and garbled pages through), at most 5% digits (tables, statistics). About half of the
/// text.
fn pleias_books(path: &Path) -> Vec<(String, String)> {
    use parquet::file::reader::{FileReader, SerializedFileReader};
    use parquet::record::Field;
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let reader = SerializedFileReader::new(file).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let rows = reader.get_row_iter(None).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut books = Vec::new();
    for row in rows {
        let row = row.unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let (mut id, mut text, mut date, mut confidence, mut words) = (None, None, None, 0.0, 0);
        for (name, field) in row.get_column_iter() {
            match (name.as_str(), field) {
                ("file_id", Field::Str(v)) => id = Some(v.clone()),
                ("complete_text", Field::Str(v)) => text = Some(v),
                ("date", Field::Str(v)) => date = v.get(..4).and_then(|y| y.parse::<u32>().ok()),
                ("ocr", Field::Str(v)) => confidence = v.trim().parse::<f64>().unwrap_or(0.0),
                ("word_count", Field::Long(v)) => words = *v,
                _ => {}
            }
        }
        let (Some(id), Some(text)) = (id, text) else { continue };
        if date.is_none_or(|y| y < 1800) || words < 5000 || confidence < 90.0 {
            continue;
        }
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let spelled = tokens.iter().filter(|w| spelled_as_a_word(w)).count();
        let digits = text.chars().filter(char::is_ascii_digit).count();
        if spelled * 100 < tokens.len() * 92 || digits * 20 > text.chars().count() {
            continue;
        }
        books.push((id, text.clone()));
    }
    books
}

/// Whether an OCR'd token reads as a word: letters (an apostrophe, a hyphen inside), then perhaps
/// punctuation.
fn spelled_as_a_word(token: &str) -> bool {
    let body = token.trim_end_matches(['.', ',', ';', ':', '!', '?', '»', ')']);
    !body.is_empty() && body.chars().all(|c| c.is_alphabetic() || matches!(c, '\'' | '’' | '-'))
}

/// The corpus, built on first use: its directory (`.cache/french`), holding `train.bin` and
/// `val.bin`, one byte per character. Built a book at a time, straight to disk: the Gutenberg
/// books, then the PleIAs files.
pub fn build(cache_dir: &Path) -> PathBuf {
    let dir = cache_dir.join("french");
    let (train_path, val_path) = (dir.join("train.bin"), dir.join("val.bin"));
    if train_path.exists() && val_path.exists() {
        return dir;
    }
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("cannot create {}: {e}", dir.display()));
    let writer = |path: &Path| {
        std::io::BufWriter::new(std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display())))
    };
    let partial = dir.join("train.partial");
    // Each book to its split, as it comes.
    struct Sink<W> {
        train: W,
        val: W,
        train_chars: u64,
        val_chars: u64,
    }
    impl<W: Write> Sink<W> {
        fn add(&mut self, id: u64, text: &str) {
            let ids = encode(&normalize(text));
            if is_validation(id) {
                self.val_chars += ids.len() as u64;
                self.val.write_all(&ids)
            } else {
                self.train_chars += ids.len() as u64;
                self.train.write_all(&ids)
            }
            .unwrap_or_else(|e| panic!("cannot write the corpus: {e}"));
        }
    }
    let mut sink = Sink { train: writer(&partial), val: writer(&val_path), train_chars: 0, val_chars: 0 };
    let books = books();
    for (k, book) in books.iter().enumerate() {
        if let Some(raw) = fetch_book(cache_dir, book.id) {
            match gutenberg_body(&raw) {
                Some(body) => sink.add(u64::from(book.id), body),
                None => eprintln!("nanolm: skipping ebook {} ({}, {}), no Gutenberg frame", book.id, book.author, book.title),
            }
        }
        if (k + 1) % 500 == 0 {
            eprintln!("nanolm: {} / {} Gutenberg books", k + 1, books.len());
        }
    }
    for k in 1..=PLEIAS_FILES {
        let path = fetch_pleias(cache_dir, k);
        let kept = pleias_books(&path);
        let count = kept.len();
        for (id, text) in kept {
            sink.add(gallica_key(&id), &ocr_paragraphs(&text));
        }
        eprintln!(
            "nanolm: PleIAs file {k} / {PLEIAS_FILES}: {count} books kept, {:.1} G characters so far",
            (sink.train_chars + sink.val_chars) as f64 / 1e9
        );
    }
    std::fs::write(dir.join("alphabet.txt"), ALPHABET).unwrap();
    sink.train.flush().unwrap();
    sink.val.flush().unwrap();
    let (train_chars, val_chars) = (sink.train_chars, sink.val_chars);
    drop(sink);
    // Renamed last: its presence means the corpus is complete.
    std::fs::rename(&partial, &train_path).unwrap_or_else(|e| panic!("cannot write {}: {e}", train_path.display()));
    eprintln!("nanolm: corpus of {train_chars} training and {val_chars} validation characters");
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_book_list_parses_and_about_one_in_fifty_is_held_out() {
        let books = books();
        assert_eq!(books.len(), 4189);
        let mut ids: Vec<u32> = books.iter().map(|b| b.id).collect();
        ids.dedup();
        assert_eq!(ids.len(), books.len(), "books.txt is sorted, without duplicates");
        assert!(books.iter().all(|b| !b.author.is_empty() && !b.title.is_empty()));
        let held_out = books.iter().filter(|b| is_validation(u64::from(b.id))).count();
        assert!((50..=120).contains(&held_out), "{held_out} validation books");
        let gallica = (0..10000).filter(|k| is_validation(gallica_key(&format!("bpt6k{k}")))).count();
        assert!((150..=250).contains(&gallica), "{gallica} of 10000 Gallica books held out");
    }

    #[test]
    fn ocr_lines_are_rejoined_into_paragraphs_without_page_numbers() {
        let ocr = "Il ne resta que huit jours \na Paris &amp; ailleurs. \n( 6) \nA3 \nLe roi vint.\nFin";
        assert_eq!(ocr_paragraphs(ocr), "Il ne resta que huit jours a Paris & ailleurs.\n\nLe roi vint.\n\nFin ");
        assert_eq!(normalize(&ocr_paragraphs(ocr)), "Il ne resta que huit jours a Paris et ailleurs.\nLe roi vint.\nFin\n");
        assert!(spelled_as_a_word("l'homme,") && spelled_as_a_word("Paris.") && !spelled_as_a_word("34,2"));
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
