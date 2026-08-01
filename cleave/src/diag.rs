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
        Diagnostic { severity: Severity::Error, message: message.into(), span: Some(span) }
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
        Diagnostic::error(err.variant.message().into_owned(), Span { file, start: offset, end: offset })
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

    fn line_col(&self, file: FileId, byte_offset: usize) -> Option<(usize, usize)> {
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
