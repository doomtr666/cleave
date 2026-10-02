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
//! Errors end the process with a message: there is no unwinding across the C
//! boundary, and cleave has no exceptions to turn one into.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;

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
    Writing {
        out: BufWriter<File>,
        tmp: PathBuf,
        path: PathBuf,
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
    let path = path_from(path, len);
    let mut tmp = path.clone().into_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let file = File::create(&tmp).unwrap_or_else(|e| fail(format!("cannot create {}: {e}", tmp.display())));
    let mut out = BufWriter::new(file);
    let header = out.write_all(MAGIC).and_then(|()| out.write_all(&VERSION.to_le_bytes()));
    header.unwrap_or_else(|e| fail(format!("cannot write {}: {e}", tmp.display())));
    register(Open::Writing { out, tmp, path })
}

/// Opens the checkpoint at `path` for reading.
///
/// # Safety
/// `path` must point to `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleave_ckpt_open(path: *const u8, len: i64) -> i64 {
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

/// Ends a checkpoint: a written one replaces `path` atomically; a read one
/// must have been read to its end.
#[unsafe(no_mangle)]
pub extern "C" fn cleave_ckpt_close(handle: i64) {
    let open = {
        let mut files = FILES.lock().unwrap_or_else(|e| e.into_inner());
        files.get_mut(handle as usize).and_then(Option::take)
    };
    match open {
        Some(Open::Writing { out, tmp, path }) => {
            let file = out
                .into_inner()
                .unwrap_or_else(|e| fail(format!("cannot write {}: {e}", tmp.display())));
            file.sync_all().unwrap_or_else(|e| fail(format!("cannot write {}: {e}", tmp.display())));
            drop(file);
            std::fs::rename(&tmp, &path)
                .unwrap_or_else(|e| fail(format!("cannot replace {}: {e}", path.display())));
        }
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
        let Open::Writing { out, tmp, .. } = open else {
            fail(format!("checkpoint {handle} is open for reading, not writing"))
        };
        let mut record = vec![tag, dims.len() as u8];
        for d in dims {
            record.extend_from_slice(&d.to_le_bytes());
        }
        let written = out.write_all(&record).and_then(|()| out.write_all(payload));
        written.unwrap_or_else(|e| fail(format!("cannot write {}: {e}", tmp.display())));
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
