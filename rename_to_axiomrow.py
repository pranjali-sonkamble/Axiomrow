"""Rename DataGPT -> Axiomrow across a project, preserving capitalisation.

    python rename_to_axiomrow.py <project_dir>            # DRY RUN: shows what would change
    python rename_to_axiomrow.py <project_dir> --apply    # makes the changes (writes .bak-free; use git!)

Case mapping (matches 'data gpt', 'data_gpt' and 'datagpt' in any case):
    DataGPT / dataGPT / Datagpt -> Axiomrow      datagpt / data_gpt -> axiomrow      DATAGPT -> AXIOMROW
So logger names ("datagpt.llm_agent"), report footers ("DataGPT | Data Analysis Report"),
file names inside strings ("datagpt_report.pdf") and env vars (DATAGPT_CSV) are all updated together.

Only text files are touched (.py .md .txt .toml .cfg .ini .yaml .yml .json .html .css .bat .sh .env.example).
Skipped: .git, virtualenvs, __pycache__, node_modules, binary files, and real .env files (secrets).
Folder and file NAMES are not renamed - rename the project folder yourself afterwards.
Commit to git first so you can review with `git diff`.
"""
import os
import re
import sys

TEXT_EXT = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".html", ".css", ".bat", ".sh", ".example"}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".pytest_cache", ".idea", ".vscode", "site-packages"}
PATTERN = re.compile(r"data[ _]?gpt", re.IGNORECASE)


def replace_case_aware(match):
    s = match.group(0)
    if s.isupper():
        return "AXIOMROW"
    if s.islower() or "_" in s:
        return "axiomrow"
    return "Axiomrow"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    apply = "--apply" in sys.argv
    total_files = total_hits = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if name in {".env", "rename_to_axiomrow.py"} or ext not in TEXT_EXT:
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8", newline="") as f:   # newline="" keeps CRLF/LF exactly
                    text = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            new_text, n = PATTERN.subn(replace_case_aware, text)
            if n:
                total_files += 1
                total_hits += n
                print(f"{'UPDATED' if apply else 'would update'}  {os.path.relpath(path, root)}  ({n})")
                if apply:
                    with open(path, "w", encoding="utf-8", newline="") as f:
                        f.write(new_text)
    print(f"\n{total_hits} occurrence(s) in {total_files} file(s). " + ("Applied." if apply else "Dry run only - add --apply to write changes."))


if __name__ == "__main__":
    main()
