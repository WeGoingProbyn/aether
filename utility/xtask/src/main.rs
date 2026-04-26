use std::fs;

fn main() {
  for entry in walkdir::WalkDir::new("src") {
    let entry = entry.unwrap();
    let path = entry.path();

    if path.extension().and_then(|s| s.to_str()) != Some("rs") {
      continue;
    }

    let content = fs::read_to_string(path).unwrap();

    if content.contains("SPDX-License-Identifier") {
      continue;
    }

    let new_content = format!(
      "// Copyright 2026 William Probyn\n// SPDX-License-Identifier: Apache-2.0\n\n{}",
      content
    );

    fs::write(path, new_content).unwrap();
  }
}
