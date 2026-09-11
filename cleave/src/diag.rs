//! Unified diagnostic output. Every error-producing pass (parsing,
//! inference, ...) converts into the same `Diagnostic` shape, rendered
//! through the same `SourceMap`, so every error looks the same on screen —
//! and in the same `file:line:col: error: message` form that editor
//! terminals (VS Code's included) recognize and turn into a clickable link.

use crate::ast::{FileId, Span};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
}

impl Diagnostic {
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Diagnostic {
            severity: Severity::Error,
            message: message.into(),
            span: Some(span),
        }
    }

    /// Converts a pest parse error, whose own byte offset (`err.location`)
    /// is reused directly as an ordinary `ast::Span` — one offset-to-line:col
    /// path (`SourceMap::render`) serves both parsing and inference errors,
    /// rather than trusting pest's own separate `line_col` computation.
    pub fn from_pest(err: &pest::error::Error<crate::parser::Rule>, file: FileId) -> Self {
        let offset = match err.location {
            pest::error::InputLocation::Pos(p) => p,
            pest::error::InputLocation::Span((s, _)) => s,
        };
        Diagnostic::error(
            err.variant.message().into_owned(),
            Span {
                file,
                start: offset,
                end: offset,
            },
        )
    }
}

/// Maps `FileId`s to their display name and source text — the "compiler
/// driver's file table" `ast.rs` already assumes exists, needed here to turn
/// a byte offset back into a 1-based (line, column) pair.
#[derive(Debug, Default)]
pub struct SourceMap {
    files: HashMap<FileId, (String, String)>,
}

impl SourceMap {
    pub fn add(&mut self, id: FileId, name: impl Into<String>, text: impl Into<String>) {
        self.files.insert(id, (name.into(), text.into()));
    }

    /// The name (path) of the lowest-`FileId` source -- the user's own
    /// entry file, since stdlib files are loaded afterwards and get higher
    /// ids. Used as the single debug-info source-file path.
    pub fn primary_path(&self) -> Option<&str> {
        self.files
            .iter()
            .min_by_key(|(id, _)| id.0)
            .map(|(_, (name, _))| name.as_str())
    }

    /// The path for one specific file -- unlike `primary_path` (always the
    /// user's own entry file), this is per-`FileId`: a `Span`'s own file is
    /// generally *not* the entry file once it points into an inlined
    /// stdlib body.
    pub fn path(&self, file: FileId) -> Option<&str> {
        self.files.get(&file).map(|(name, _)| name.as_str())
    }

    /// Every registered file's own path, keyed by its raw `FileId.0` -- the
    /// table `mlir_lower.rs::set_gen_file_table` needs to resolve a
    /// per-function/per-statement `FileId` back to a real path at debug-info
    /// emission time, without `mlir_lower.rs` itself depending on
    /// `SourceMap`/`FileId`.
    pub fn path_table(&self) -> std::collections::HashMap<u32, String> {
        self.files
            .iter()
            .map(|(id, (name, _))| (id.0, name.clone()))
            .collect()
    }

    pub fn line_col(&self, file: FileId, byte_offset: usize) -> Option<(usize, usize)> {
        let (_, text) = self.files.get(&file)?;
        let offset = byte_offset.min(text.len());
        let mut line = 1;
        let mut col = 1;
        for ch in text[..offset].chars() {
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        Some((line, col))
    }

    pub fn render(&self, diag: &Diagnostic) -> String {
        let severity = match diag.severity {
            Severity::Error => "error",
        };
        let located = diag.span.and_then(|s| {
            let name = self.files.get(&s.file).map(|(name, _)| name.clone())?;
            let (line, col) = self.line_col(s.file, s.start)?;
            Some((name, line, col))
        });
        match located {
            Some((name, line, col)) => format!("{name}:{line}:{col}: {severity}: {}", diag.message),
            None => format!("<unknown>: {severity}: {}", diag.message),
        }
    }
}
