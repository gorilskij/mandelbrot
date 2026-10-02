//! The documentation cannot silently rot: every relative link (and `#anchor`)
//! resolves, every page in `docs/` is linked from another page, and
//! `CLAUDE.md` points into `docs/` (docs/README.md, "Keeping the docs true").
//!
//! `cargo test --test docs`

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Kept verbatim on purpose: historical, not maintained.
const VERBATIM: &[&str] = &["docs/history/working-notes-2026-10-03.md"];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn pages(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            pages(&path, out);
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
}

fn all_pages() -> Vec<PathBuf> {
    let mut out = Vec::new();
    pages(&root().join("docs"), &mut out);
    out.sort();
    out
}

/// The text without fenced code blocks (links and headings there don't count).
fn prose(path: &Path) -> String {
    let mut out = String::new();
    let mut fenced = false;
    for line in fs::read_to_string(path).unwrap().lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        } else if !fenced {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Targets of `[text](target)` links, images excluded.
fn links(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let b = text.as_bytes();
    let mut i = 0;
    while let Some(off) = text[i..].find("](") {
        let close = i + off;
        // the `[` that opens this link text
        let open = text[..close].rfind('[');
        let start = close + 2;
        let end = text[start..].find(')').map(|e| start + e);
        if let (Some(open), Some(end)) = (open, end) {
            let is_image = open > 0 && b[open - 1] == b'!';
            let target = &text[start..end];
            if !is_image && !target.contains(char::is_whitespace) {
                out.push(target.to_string());
            }
        }
        i = close + 2;
    }
    out
}

/// GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens.
fn slug(heading: &str) -> String {
    heading
        .trim()
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-' || *c == ' ' || *c == '_')
        .map(|c| if c == ' ' { '-' } else { c })
        .collect()
}

fn anchors(path: &Path) -> HashSet<String> {
    prose(path)
        .lines()
        .filter(|l| l.starts_with('#'))
        .map(|l| slug(l.trim_start_matches('#')))
        .collect()
}

fn checked() -> Vec<PathBuf> {
    let verbatim: Vec<PathBuf> = VERBATIM.iter().map(|p| root().join(p)).collect();
    let mut out: Vec<PathBuf> = all_pages().into_iter().filter(|p| !verbatim.contains(p)).collect();
    out.push(root().join("CLAUDE.md"));
    out
}

fn is_external(target: &str) -> bool {
    target.split_once(':').is_some_and(|(scheme, _)| scheme.chars().all(|c| c.is_ascii_lowercase()))
}

#[test]
fn every_relative_link_resolves() {
    let mut broken = Vec::new();
    for page in checked() {
        for target in links(&prose(&page)) {
            if is_external(&target) {
                continue;
            }
            let (file, anchor) = target.split_once('#').unwrap_or((&target, ""));
            let resolved = if file.is_empty() { page.clone() } else { page.parent().unwrap().join(file) };
            let rel = page.strip_prefix(root()).unwrap().display().to_string();
            if !resolved.exists() {
                broken.push(format!("{rel}: {target}"));
            } else if !anchor.is_empty()
                && resolved.extension().is_some_and(|e| e == "md")
                && !anchors(&resolved).contains(anchor)
            {
                broken.push(format!("{rel}: {target} (no such heading)"));
            }
        }
    }
    assert!(broken.is_empty(), "broken links:\n{}", broken.join("\n"));
}

#[test]
fn every_page_is_linked_from_another() {
    let mut linked = HashSet::new();
    for page in checked() {
        for target in links(&prose(&page)) {
            let file = target.split('#').next().unwrap();
            if !file.is_empty() && !is_external(file) {
                if let Ok(p) = page.parent().unwrap().join(file).canonicalize() {
                    linked.insert(p);
                }
            }
        }
    }
    let index = root().join("docs/README.md").canonicalize().unwrap();
    let orphans: Vec<String> = all_pages()
        .into_iter()
        .filter(|p| {
            let p = p.canonicalize().unwrap();
            p != index && !linked.contains(&p)
        })
        .map(|p| p.strip_prefix(root()).unwrap().display().to_string())
        .collect();
    assert!(orphans.is_empty(), "pages no other page links to: {orphans:?}");
}

#[test]
fn claude_md_points_into_the_docs() {
    let text = fs::read_to_string(root().join("CLAUDE.md")).unwrap();
    assert!(text.contains("docs/README.md") && text.contains("docs/status.md"));
}
