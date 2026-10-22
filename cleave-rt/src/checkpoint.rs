//! Checkpoint files: what `stdlib/checkpoint` writes a value to and reads it
//! back from (`doc/plan-nanolm.md`, step 0).
//!
//! A file is a magic and a version, then one record per leaf of the saved
//! value, in the order the stdlib visits them: an element tag, a rank, the
//! dims, the data. Reading checks every record against what the program
//! expects at that position (the tag and the shape of the value it restores
//! into), so a model of another shape is a clear error naming both, never
//! weights read out of alignment.
//!
//! A checkpoint is written to `<path>.tmp` and renamed over `<path>` only
//! once complete: a process stopped mid-save leaves the previous checkpoint
//! intact.
//!
//! Written in the background: each record is copied and handed to a thread
//! of the checkpoint's own, which writes it, then flushes the file to disk and
//! renames it once the checkpoint is closed. The program carries on meanwhile
//! (a training run's next steps on the device: nanoLM's GPU waited seconds per
//! save for the disk). One save at a time: a checkpoint created or opened, or
//! the process ending, first waits for the saves under way
//! (`cleave_ckpt_wait`), so a read always finds the last save complete. A
//! host recording that a save is in place (nanoLM's `gpt.step`) waits for it
//! (`cleave_ckpt_saves_begun`, `cleave_ckpt_wait_for`).
//!
//! Errors end the process with a message: there is no unwinding across the C
//! boundary, and cleave has no exceptions to turn one into.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Condvar, Mutex};

const MAGIC: &[u8; 8] = b"CLVCKPT\0";
const VERSION: u32 = 1;

/// The element tags of a record.
const TAG_F32: u8 = 1;
const TAG_F64: u8 = 2;
const TAG_I32: u8 = 3;
const TAG_I64: u8 = 4;

fn tag_name(tag: u8) -> &'static str {
    match tag {
        TAG_F32 => "f32",
        TAG_F64 => "f64",
        TAG_I32 => "i32",
        TAG_I64 => "i64",
        _ => "?",
    }
}

enum Open {
    /// The records go to the checkpoint's writer thread (`writer`).
    Writing {
        records: Sender<Vec<u8>>,
    },
    Reading {
        input: BufReader<File>,
        path: PathBuf,
        leaf: u64,
    },
}

static FILES: Mutex<Vec<Option<Open>>> = Mutex::new(Vec::new());

fn fail(message: String) -> ! {
    eprintln!("cleave checkpoint: {message}");
    std::process::exit(1)
}

fn path_from(ptr: *const u8, len: i64) -> PathBuf {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
    PathBuf::from(String::from_utf8_lossy(bytes).trim_end_matches('\0').to_string())
}

fn with_file<R>(handle: i64, f: impl FnOnce(&mut Open) -> R) -> R {
    let mut files = FILES.lock().unwrap_or_else(|e| e.into_inner());
    match files.get_mut(handle as usize).and_then(Option::as_mut) {
        Some(open) => f(open),
        None => fail(format!("no checkpoint open with handle {handle}")),
    }
}

/// Saves begun and saves complete (their file in place), in order: one at a
/// time, so the `n`th begun is in place once `n` are complete.
static SAVES: Mutex<(u64, u64)> = Mutex::new((0, 0));
static SAVED: Condvar = Condvar::new();

/// How many checkpoints were created so far: the last one is in place once
/// `cleave_ckpt_wait_for` that many returns.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_ckpt_saves_begun() -> i64 {
    SAVES.lock().unwrap_or_else(|e| e.into_inner()).0 as i64
}

/// Waits until `n` checkpoints are in place (written, flushed, renamed);
/// returns how many are.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_ckpt_wait_for(n: i64) -> i64 {
    let mut saves = SAVES.lock().unwrap_or_else(|e| e.into_inner());
    while (saves.1 as i64) < n {
        saves = SAVED.wait(saves).unwrap_or_else(|e| e.into_inner());
    }
    saves.1 as i64
}

/// Waits until every checkpoint created so far is in place.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_ckpt_wait() {
    cleave_ckpt_wait_for(cleave_ckpt_saves_begun());
}

/// The process's end waits for the saves under way (`atexit`): returning
/// from `main` would otherwise end the writer thread mid-file.
fn wait_at_exit() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    unsafe extern "C" {
        fn atexit(f: extern "C" fn()) -> i32;
    }
    extern "C" fn at_exit() {
        cleave_ckpt_wait();
    }
    ONCE.call_once(|| unsafe {
        atexit(at_exit);
    });
}

/// The thread writing one checkpoint: its records in order, then, once
/// they're all sent (the checkpoint closed), the file flushed to disk and
/// renamed into place.
fn writer(mut out: BufWriter<File>, records: std::sync::mpsc::Receiver<Vec<u8>>, tmp: PathBuf, path: PathBuf) {
    // A save that fails ends the process, its waits released first: the end
    // of the process waits for every save (`wait_at_exit`).
    let failed = |message: String| -> ! {
        SAVES.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
        SAVED.notify_all();
        fail(message)
    };
    for record in records {
        out.write_all(&record).unwrap_or_else(|e| failed(format!("cannot write {}: {e}", tmp.display())));
    }
    let file = out.into_inner().unwrap_or_else(|e| failed(format!("cannot write {}: {e}", tmp.display())));
    file.sync_all().unwrap_or_else(|e| failed(format!("cannot write {}: {e}", tmp.display())));
    drop(file);
    std::fs::rename(&tmp, &path).unwrap_or_else(|e| failed(format!("cannot replace {}: {e}", path.display())));
    SAVES.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
    SAVED.notify_all();
}

fn register(open: Open) -> i64 {
    let mut files = FILES.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = files.iter().position(Option::is_none) {
        files[i] = Some(open);
        i as i64
    } else {
        files.push(Some(open));
        (files.len() - 1) as i64
    }
}

/// Starts a checkpoint at `path` (written to `path.tmp` until closed).
///
/// # Safety
/// `path` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_ckpt_create(path: *const u8, len: i64) -> i64 {
    cleave_ckpt_wait();
    wait_at_exit();
    let path = path_from(path, len);
    let mut tmp = path.clone().into_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let file = File::create(&tmp).unwrap_or_else(|e| fail(format!("cannot create {}: {e}", tmp.display())));
    let mut out = BufWriter::new(file);
    let header = out.write_all(MAGIC).and_then(|()| out.write_all(&VERSION.to_le_bytes()));
    header.unwrap_or_else(|e| fail(format!("cannot write {}: {e}", tmp.display())));
    SAVES.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
    let (records, received) = channel();
    std::thread::Builder::new()
        .name("cleave checkpoint".to_string())
        .spawn(move || writer(out, received, tmp, path))
        .unwrap_or_else(|e| fail(format!("cannot start the checkpoint's writer: {e}")));
    register(Open::Writing { records })
}

/// Opens the checkpoint at `path` for reading.
///
/// # Safety
/// `path` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_ckpt_open(path: *const u8, len: i64) -> i64 {
    cleave_ckpt_wait();
    let path = path_from(path, len);
    let file = File::open(&path).unwrap_or_else(|e| fail(format!("cannot open {}: {e}", path.display())));
    let mut input = BufReader::new(file);
    let mut magic = [0u8; 8];
    let mut version = [0u8; 4];
    let header = input.read_exact(&mut magic).and_then(|()| input.read_exact(&mut version));
    if header.is_err() || &magic != MAGIC {
        fail(format!("{} is not a cleave checkpoint", path.display()));
    }
    let version = u32::from_le_bytes(version);
    if version != VERSION {
        fail(format!(
            "{} is a version {version} checkpoint, this runtime reads version {VERSION}",
            path.display()
        ));
    }
    register(Open::Reading { input, path, leaf: 0 })
}

/// Ends a checkpoint: a written one replaces `path` atomically, in the
/// background (`writer`); a read one must have been read to its end.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_ckpt_close(handle: i64) {
    let open = {
        let mut files = FILES.lock().unwrap_or_else(|e| e.into_inner());
        files.get_mut(handle as usize).and_then(Option::take)
    };
    match open {
        // The last record sent: the writer finishes the file.
        Some(Open::Writing { records }) => drop(records),
        Some(Open::Reading { mut input, path, leaf }) => {
            let mut rest = [0u8; 1];
            if input.read(&mut rest).unwrap_or(0) != 0 {
                fail(format!(
                    "{} holds more than the {leaf} values restored from it: it was saved from a value of another shape",
                    path.display()
                ));
            }
        }
        None => fail(format!("no checkpoint open with handle {handle}")),
    }
}

fn shape_text(tag: u8, dims: &[i64]) -> String {
    if dims.is_empty() {
        tag_name(tag).to_string()
    } else {
        let dims: Vec<String> = dims.iter().map(i64::to_string).collect();
        format!("{}[{}]", tag_name(tag), dims.join(", "))
    }
}

fn write_record(handle: i64, tag: u8, dims: &[i64], payload: &[u8]) {
    with_file(handle, |open| {
        let Open::Writing { records } = open else {
            fail(format!("checkpoint {handle} is open for reading, not writing"))
        };
        // A copy: the program may change or free its value as soon as this
        // returns.
        let mut record = Vec::with_capacity(2 + 8 * dims.len() + payload.len());
        record.extend_from_slice(&[tag, dims.len() as u8]);
        for d in dims {
            record.extend_from_slice(&d.to_le_bytes());
        }
        record.extend_from_slice(payload);
        // A writer gone has ended the process (`fail`).
        let _ = records.send(record);
    });
}

fn read_record(handle: i64, tag: u8, dims: &[i64], payload: &mut [u8]) {
    with_file(handle, |open| {
        let Open::Reading { input, path, leaf } = open else {
            fail(format!("checkpoint {handle} is open for writing, not reading"))
        };
        let expected = shape_text(tag, dims);
        let mut head = [0u8; 2];
        if input.read_exact(&mut head).is_err() {
            fail(format!(
                "{} ends after {leaf} values, the value restored expects more (next: {expected})",
                path.display()
            ));
        }
        let mut found_dims = vec![0i64; head[1] as usize];
        for d in &mut found_dims {
            let mut b = [0u8; 8];
            input
                .read_exact(&mut b)
                .unwrap_or_else(|_| fail(format!("{} is truncated", path.display())));
            *d = i64::from_le_bytes(b);
        }
        if head[0] != tag || found_dims != dims {
            fail(format!(
                "{}: value #{leaf} is {}, the value restored into expects {expected}",
                path.display(),
                shape_text(head[0], &found_dims)
            ));
        }
        input
            .read_exact(payload)
            .unwrap_or_else(|_| fail(format!("{} is truncated", path.display())));
        *leaf += 1;
    });
}

fn dims_of(rank: i32, d0: i32, d1: i32, d2: i32) -> Vec<i64> {
    [d0, d1, d2][..rank as usize].iter().map(|&d| d as i64).collect()
}

/// Writes `len` `f32`s of shape `dims` (rank `rank`, unused dims ignored).
///
/// # Safety
/// `data` must point to `len` readable `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_ckpt_write_f32s(
    handle: i64,
    rank: i32,
    d0: i32,
    d1: i32,
    d2: i32,
    data: *const f32,
    len: i64,
) {
    let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len as usize * 4) };
    write_record(handle, TAG_F32, &dims_of(rank, d0, d1, d2), bytes);
}

/// Reads `len` `f32`s of shape `dims` into `out`.
///
/// # Safety
/// `out` must point to `len` writable `f32`s.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_ckpt_read_f32s(
    handle: i64,
    rank: i32,
    d0: i32,
    d1: i32,
    d2: i32,
    out: *mut f32,
    len: i64,
) {
    let bytes = unsafe { std::slice::from_raw_parts_mut(out as *mut u8, len as usize * 4) };
    read_record(handle, TAG_F32, &dims_of(rank, d0, d1, d2), bytes);
}

macro_rules! scalar {
    ($write:ident, $read:ident, $ty:ty, $tag:expr) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn $write(handle: i64, x: $ty) {
            write_record(handle, $tag, &[], &x.to_le_bytes());
        }
        #[unsafe(no_mangle)]
        pub extern "C" fn $read(handle: i64) -> $ty {
            let mut b = [0u8; std::mem::size_of::<$ty>()];
            read_record(handle, $tag, &[], &mut b);
            <$ty>::from_le_bytes(b)
        }
    };
}

scalar!(cleave_ckpt_write_f32, cleave_ckpt_read_f32, f32, TAG_F32);
scalar!(cleave_ckpt_write_f64, cleave_ckpt_read_f64, f64, TAG_F64);
scalar!(cleave_ckpt_write_i32, cleave_ckpt_read_i32, i32, TAG_I32);
scalar!(cleave_ckpt_write_i64, cleave_ckpt_read_i64, i64, TAG_I64);

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint closed is written in the background, and in place for
    /// whatever comes next: opened right after its close, with no wait of the
    /// caller's own, it reads back whole; the saves begun and in place counted.
    #[test]
    fn a_checkpoint_saved_in_the_background_reads_back_at_once() {
        let path = std::env::temp_dir().join(format!("cleave-ckpt-{}.ckpt", std::process::id()));
        let name = path.to_string_lossy().into_owned();
        let begun = cleave_ckpt_saves_begun();
        let data: Vec<f32> = (0..1 << 20).map(|i| i as f32 * 0.5).collect();
        let f = unsafe { cleave_ckpt_create(name.as_ptr(), name.len() as i64) };
        unsafe { cleave_ckpt_write_f32s(f, 1, data.len() as i32, 0, 0, data.as_ptr(), data.len() as i64) };
        cleave_ckpt_write_i64(f, 42);
        cleave_ckpt_close(f);
        assert_eq!(cleave_ckpt_saves_begun(), begun + 1);

        let f = unsafe { cleave_ckpt_open(name.as_ptr(), name.len() as i64) };
        assert!(cleave_ckpt_wait_for(begun + 1) >= begun + 1, "the save is in place once a read opens it");
        let mut back = vec![0f32; data.len()];
        unsafe { cleave_ckpt_read_f32s(f, 1, data.len() as i32, 0, 0, back.as_mut_ptr(), back.len() as i64) };
        assert_eq!(cleave_ckpt_read_i64(f), 42);
        cleave_ckpt_close(f);
        assert_eq!(back, data);
        let _ = std::fs::remove_file(&path);
    }
}
