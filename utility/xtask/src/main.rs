use std::{fs, path::Path};

fn visit_dir(dir: &Path) {
  let entries = match fs::read_dir(dir) {
    Ok(e) => e,
    Err(_) => return,
  };

  for entry in entries.flatten() {
    let path = entry.path();

    if path.to_string_lossy().contains("target")
    || path.to_string_lossy().contains(".git")
    {
      continue;
    }

    if path.is_dir() {
      visit_dir(&path);
      continue;
    }

    if path.extension().and_then(|s| s.to_str()) != Some("rs") {
      continue;
    }

    let content = match fs::read_to_string(&path) {
      Ok(c) => c,
      Err(_) => continue,
    };

    if content.contains("SPDX-License-Identifier") {
      continue;
    }

    let new_content = format!(
      "// Copyright 2026 William Probyn\n// SPDX-License-Identifier: Apache-2.0\n\n{}",
      content
    );

    let _ = fs::write(&path, new_content);
  }
}

fn main() {
  visit_dir(Path::new("."));
}
