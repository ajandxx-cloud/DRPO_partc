#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


VENUE = "Transportation Research Part C: Emerging Technologies (TRC)"

ROUND1_TEMPLATE = """You are reviewing a [VENUE] paper. Please provide a detailed, structured review.
## Full Paper Text:
[paste concatenated sections]

## Review Instructions
Please act as a senior ML reviewer ([VENUE] level). Provide:
1. **Overall Score** (1-10, where 6 = weak accept, 7 = accept)
2. **Summary** (2-3 sentences)
3. **Strengths** (bullet list, ranked)
4. **Weaknesses** (bullet list, ranked: CRITICAL > MAJOR > MINOR)
5. **For each CRITICAL/MAJOR weakness**: A specific, actionable fix
6. **Missing References** (if any)
7. **Verdict**: Ready for submission? Yes / Almost / No

Focus on: theoretical rigor, claims vs evidence alignment, writing clarity,
self-containedness, notation consistency.
"""

ROUND2_TEMPLATE = """[Round 2 update]

Since your last review, we have implemented:
[fix list]

## Full Updated Paper Text:
[paste concatenated sections]

Please re-score and re-assess. Same format:
Score, Summary, Strengths, Weaknesses, Actionable fixes, Verdict.
"""

NOISE_LINE_RE = re.compile(
    r"^\\(?:begin\{document\}|end\{document\}|begin\{frontmatter\}|end\{frontmatter\}|"
    r"centering|small|footnotesize|scriptsize|tiny|normalsize|"
    r"bibliographystyle\{.*|bibliography\{.*|label\{.*)\s*$"
)

TEXT_WRAPPER_PATTERNS = [
    re.compile(r"\\href\{[^{}]*\}\{([^{}]*)\}"),
    re.compile(r"\\url\{([^{}]*)\}"),
    re.compile(r"\\path\{([^{}]*)\}"),
    re.compile(
        r"\\(?:texttt|emph|textbf|textrm|textit|mathrm|mathbf|mathit|operatorname|underline)\{([^{}]*)\}"
    ),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        pieces: list[str] = []
        escaped = False
        for char in line:
            if char == "%" and not escaped:
                break
            pieces.append(char)
            escaped = char == "\\"
        cleaned_lines.append("".join(pieces).rstrip())
    return "\n".join(cleaned_lines)


def extract_braced(text: str, start: int) -> tuple[str, int]:
    if start >= len(text) or text[start] != "{":
        raise ValueError("expected opening brace")
    depth = 0
    out: list[str] = []
    index = start
    while index < len(text):
        char = text[index]
        escaped = index > 0 and text[index - 1] == "\\"
        if char == "{" and not escaped:
            depth += 1
            if depth > 1:
                out.append(char)
        elif char == "}" and not escaped:
            depth -= 1
            if depth == 0:
                return "".join(out), index + 1
            out.append(char)
        else:
            out.append(char)
        index += 1
    raise ValueError("unbalanced braces")


def skip_optional_arg(text: str, index: int, opener: str, closer: str) -> int:
    if index >= len(text) or text[index] != opener:
        return index
    depth = 0
    while index < len(text):
        char = text[index]
        escaped = index > 0 and text[index - 1] == "\\"
        if char == opener and not escaped:
            depth += 1
        elif char == closer and not escaped:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return index


def find_command_argument(text: str, command: str) -> tuple[str, int, int] | None:
    index = text.find(command)
    if index == -1:
        return None
    cursor = index + len(command)
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    cursor = skip_optional_arg(text, cursor, "[", "]")
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None
    content, end = extract_braced(text, cursor)
    return content, end, index


def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def unwrap_text_commands(text: str) -> str:
    updated = text
    changed = True
    while changed:
        changed = False
        for pattern in TEXT_WRAPPER_PATTERNS:
            updated, count = pattern.subn(r"\1", updated)
            if count:
                changed = True
    updated = updated.replace("~", " ")
    updated = updated.replace(r"\%", "%")
    updated = updated.replace(r"\_", "_")
    updated = updated.replace(r"\&", "&")
    updated = updated.replace(r"\#", "#")
    updated = updated.replace(r"\$", "$")
    updated = updated.replace(r"\sep", "; ")
    updated = updated.replace("\\\\", "\n")
    return updated


def clean_inline_latex(text: str) -> str:
    cleaned = unwrap_text_commands(text)
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def parse_bbl_entries(bbl_text: str) -> list[tuple[str, str, str]]:
    chunks = re.split(r"(?=\\bibitem)", bbl_text)
    entries: list[tuple[str, str, str]] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk.startswith(r"\bibitem"):
            continue
        match = re.match(r"\\bibitem\[\{?(.*?)\}?\]\{([^}]+)\}", chunk, re.S)
        if not match:
            continue
        entries.append((match.group(2).strip(), match.group(1).strip(), chunk[match.end() :].strip()))
    return entries


def build_citation_map(bbl_text: str) -> dict[str, str]:
    citation_map: dict[str, str] = {}
    for key, header, _ in parse_bbl_entries(bbl_text):
        header = header.replace("~", " ")
        label_match = re.match(r"(.+?\(\d{4}[a-z]?\))", header)
        label = label_match.group(1) if label_match else header
        label = re.sub(r"([A-Za-z.])\(", r"\1 (", label, count=1)
        citation_map[key] = label
    return citation_map


def replace_citations(text: str, citation_map: dict[str, str]) -> str:
    citation_re = re.compile(r"\\(cite[a-zA-Z*]*?)(?:\[[^\]]*\]){0,2}\{([^{}]+)\}")

    def repl(match: re.Match[str]) -> str:
        command = match.group(1)
        keys = [item.strip() for item in match.group(2).split(",") if item.strip()]
        labels = [citation_map.get(key, key) for key in keys]
        if command.startswith("citet") or command.startswith("citeauthor"):
            return "; ".join(labels)
        return "(" + "; ".join(labels) + ")"

    return citation_re.sub(repl, text)


def replace_environment(text: str, env_names: list[str], handler) -> str:
    updated = text
    for env_name in env_names:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(env_name)}\}}(.*?)\\end\{{{re.escape(env_name)}\}}",
            re.S,
        )
        updated = pattern.sub(lambda match: handler(env_name, match.group(1)), updated)
    return updated


def render_highlights(_: str, body: str) -> str:
    items = [clean_inline_latex(item) for item in re.findall(r"\\item\s+(.*)", body)]
    lines = ["## Highlights"]
    lines.extend(f"- {item}" for item in items if item)
    return "\n" + "\n".join(lines) + "\n"


def render_abstract(_: str, body: str) -> str:
    return "\n## Abstract\n" + clean_inline_latex(body) + "\n"


def render_keywords(_: str, body: str) -> str:
    return "\n## Keywords\n" + clean_inline_latex(body) + "\n"


def strip_command(text: str, command: str) -> str:
    found = find_command_argument(text, command)
    if not found:
        return text
    _, end, start = found
    return text[:start] + text[end:]


def render_figure(_: str, body: str) -> str:
    caption = find_command_argument(body, r"\caption")
    if not caption:
        return "\n"
    caption_text = clean_inline_latex(caption[0])
    return f"\n[Figure] {caption_text}\n"


def clean_table_body(table_body: str) -> str:
    body = table_body
    caption = find_command_argument(body, r"\caption")
    if caption:
        body = body[: caption[2]] + body[caption[1] :]
    body = re.sub(r"\\label\{[^{}]*\}", "", body)
    body = re.sub(r"\\begin\{threeparttable\}|\\end\{threeparttable\}", "", body)
    body = re.sub(r"\\begin\{tablenotes\}|\\end\{tablenotes\}", "", body)
    cleaned_lines: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if NOISE_LINE_RE.match(stripped):
            continue
        cleaned_lines.append(line.rstrip())
    return normalize_spaces("\n".join(cleaned_lines)).strip()


def render_table(_: str, body: str) -> str:
    caption = find_command_argument(body, r"\caption")
    caption_text = clean_inline_latex(caption[0]) if caption else "Table"
    table_block = clean_table_body(body)
    if table_block:
        return f"\n[Table] {caption_text}\n```latex\n{table_block}\n```\n"
    return f"\n[Table] {caption_text}\n"


def convert_structural_commands(text: str) -> str:
    converted = text

    title = find_command_argument(converted, r"\title")
    authors = re.findall(r"\\author(?:\[[^\]]*\])?\{([^{}]*)\}", converted)
    addresses = re.findall(r"\\address(?:\[[^\]]*\])?\{([^{}]*)\}", converted)

    header_parts: list[str] = []
    if title:
        header_parts.append("# " + clean_inline_latex(title[0]))
        converted = converted[: title[2]] + converted[title[1] :]
    if authors:
        header_parts.append("Authors: " + ", ".join(clean_inline_latex(item) for item in authors))
        converted = re.sub(r"\\author(?:\[[^\]]*\])?\{[^{}]*\}", "", converted)
    if addresses:
        header_parts.append(
            "Affiliations: " + " | ".join(clean_inline_latex(item) for item in addresses)
        )
        converted = re.sub(r"\\address(?:\[[^\]]*\])?\{[^{}]*\}", "", converted)

    converted = re.sub(r"\\appendix\b", "\n## Appendix\n", converted)
    converted = re.sub(
        r"\\section\*?\{([^{}]*)\}",
        lambda match: "\n## " + clean_inline_latex(match.group(1)) + "\n",
        converted,
    )
    converted = re.sub(
        r"\\subsection\*?\{([^{}]*)\}",
        lambda match: "\n### " + clean_inline_latex(match.group(1)) + "\n",
        converted,
    )
    converted = re.sub(
        r"\\subsubsection\*?\{([^{}]*)\}",
        lambda match: "\n#### " + clean_inline_latex(match.group(1)) + "\n",
        converted,
    )
    converted = re.sub(r"\\item\b", "\n- ", converted)
    converted = re.sub(r"\\begin\{(?:itemize|enumerate)\}|\\end\{(?:itemize|enumerate)\}", "", converted)
    converted = re.sub(r"\\label\{[^{}]*\}", "", converted)
    converted = re.sub(r"\\journal\{[^{}]*\}", "", converted)
    converted = re.sub(r"\\begin\{frontmatter\}|\\end\{frontmatter\}", "", converted)
    converted = re.sub(r"\\begin\{document\}|\\end\{document\}", "", converted)

    cleaned_lines: list[str] = []
    for line in converted.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if NOISE_LINE_RE.match(stripped):
            continue
        cleaned_lines.append(line.rstrip())

    body = "\n".join(cleaned_lines)
    body = unwrap_text_commands(body)
    body = re.sub(r"\\(left|right)\b", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)

    if header_parts:
        body = "\n".join(header_parts) + "\n\n" + body.lstrip()
    return normalize_spaces(body)


def extract_document(tex_text: str, bbl_text: str) -> str:
    tex = strip_comments(tex_text)
    start = tex.find(r"\begin{document}")
    end = tex.rfind(r"\end{document}")
    if start == -1 or end == -1:
        raise ValueError("could not locate LaTeX document body")
    body = tex[start:end]
    citation_map = build_citation_map(bbl_text)
    body = replace_citations(body, citation_map)
    body = replace_environment(body, ["highlights"], render_highlights)
    body = replace_environment(body, ["abstract"], render_abstract)
    body = replace_environment(body, ["keyword"], render_keywords)
    body = replace_environment(body, ["figure", "figure*"], render_figure)
    body = replace_environment(body, ["table", "table*"], render_table)
    body = convert_structural_commands(body)
    references = format_references(bbl_text)
    return normalize_spaces(body + "\n## References\n" + references)


def format_references(bbl_text: str) -> str:
    entries: list[str] = []
    for _, _, raw_entry in parse_bbl_entries(bbl_text):
        cleaned = strip_comments(raw_entry).replace("\n", " ")
        cleaned = re.sub(r"\\newblock\b", " ", cleaned)
        cleaned = unwrap_bbl_commands(cleaned)
        cleaned = cleaned.replace("~", " ")
        cleaned = cleaned.replace("{", "").replace("}", "")
        cleaned = cleaned.replace(r"\DOIprefix", "")
        cleaned = cleaned.replace(r"\URLprefix", "")
        cleaned = cleaned.replace(r"\ArXivprefix", "arXiv: ")
        cleaned = cleaned.replace(r"\Pubmedprefix", "pmid: ")
        cleaned = cleaned.replace(r"\end{thebibliography}", "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        entries.append(f"- {cleaned}")
    return "\n".join(entries).strip() + "\n"


def unwrap_bbl_commands(text: str) -> str:
    updated = text

    while True:
        start = updated.find(r"\bibinfo")
        if start == -1:
            break
        first = find_command_argument(updated[start:], r"\bibinfo")
        if not first:
            break
        field, first_end, _ = first
        field_start = start + updated[start:].find(r"\bibinfo")
        field_arg_start = field_start + len(r"\bibinfo")
        while field_arg_start < len(updated) and updated[field_arg_start].isspace():
            field_arg_start += 1
        _, first_arg_end = extract_braced(updated, field_arg_start)
        while first_arg_end < len(updated) and updated[first_arg_end].isspace():
            first_arg_end += 1
        if first_arg_end >= len(updated) or updated[first_arg_end] != "{":
            break
        second_content, second_end = extract_braced(updated, first_arg_end)
        replacement = second_content
        updated = updated[:field_start] + replacement + updated[second_end:]

    simple_patterns = [
        (re.compile(r"\\doi\{([^{}]*)\}"), r"doi: \1"),
        (re.compile(r"\\href\{[^{}]*\}\{([^{}]*)\}"), r"\1"),
        (re.compile(r"\\path\{([^{}]*)\}"), r"\1"),
        (re.compile(r"\\url\{([^{}]*)\}"), r"\1"),
        (re.compile(r"\\texttt\{([^{}]*)\}"), r"\1"),
    ]
    for pattern, replacement in simple_patterns:
        updated = pattern.sub(replacement, updated)
    return updated


def build_round1_prompt(paper_text: str) -> str:
    return (
        ROUND1_TEMPLATE.replace("[VENUE]", VENUE).replace("[paste concatenated sections]", paper_text)
    )


def build_round2_prompt(paper_text: str, updates_text: str) -> str:
    return (
        ROUND2_TEMPLATE.replace("[fix list]", updates_text.strip()).replace(
            "[paste concatenated sections]", paper_text
        )
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_output(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def command_round1(args: argparse.Namespace) -> None:
    tex_text = read_text(args.tex)
    bbl_text = read_text(args.bbl)
    paper_text = extract_document(tex_text, bbl_text)
    prompt_text = build_round1_prompt(paper_text)

    ensure_dir(args.out_dir)
    paper_path = args.out_dir / "round1_paper_text.md"
    prompt_path = args.out_dir / "round1_prompt.md"
    meta_path = args.out_dir / "round1_metadata.json"

    write_output(paper_path, paper_text)
    write_output(prompt_path, prompt_text)
    write_output(
        meta_path,
        json.dumps(
            {
                "venue": VENUE,
                "tex_path": str(args.tex),
                "bbl_path": str(args.bbl),
                "paper_chars": len(paper_text),
                "prompt_chars": len(prompt_text),
                "paper_lines": paper_text.count("\n") + 1,
                "prompt_lines": prompt_text.count("\n") + 1,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"paper_text={paper_path}")
    print(f"prompt={prompt_path}")
    print(f"metadata={meta_path}")


def command_round2(args: argparse.Namespace) -> None:
    tex_text = read_text(args.tex)
    bbl_text = read_text(args.bbl)
    updates_text = read_text(args.updates)
    paper_text = extract_document(tex_text, bbl_text)
    prompt_text = build_round2_prompt(paper_text, updates_text)

    ensure_dir(args.out_dir)
    paper_path = args.out_dir / "round2_paper_text.md"
    prompt_path = args.out_dir / "round2_prompt.md"
    meta_path = args.out_dir / "round2_metadata.json"

    write_output(paper_path, paper_text)
    write_output(prompt_path, prompt_text)
    write_output(
        meta_path,
        json.dumps(
            {
                "venue": VENUE,
                "tex_path": str(args.tex),
                "bbl_path": str(args.bbl),
                "updates_path": str(args.updates),
                "paper_chars": len(paper_text),
                "prompt_chars": len(prompt_text),
                "paper_lines": paper_text.count("\n") + 1,
                "prompt_lines": prompt_text.count("\n") + 1,
            },
            indent=2,
        )
        + "\n",
    )
    print(f"paper_text={paper_path}")
    print(f"prompt={prompt_path}")
    print(f"metadata={meta_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare TRC manuscript review packets and prompts."
    )
    parser.add_argument(
        "--tex",
        type=Path,
        default=Path("paper/trc_latex/manuscript.tex"),
        help="Path to the main manuscript tex file.",
    )
    parser.add_argument(
        "--bbl",
        type=Path,
        default=Path("paper/trc_latex/manuscript.bbl"),
        help="Path to the compiled bibliography file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/trc_latex/review_artifacts"),
        help="Directory for generated prompt artifacts.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    round1 = subparsers.add_parser("round1", help="Generate the Round 1 review packet and prompt.")
    round1.set_defaults(func=command_round1)

    round2 = subparsers.add_parser("round2", help="Generate the Round 2 review packet and prompt.")
    round2.add_argument(
        "--updates",
        type=Path,
        required=True,
        help="Path to a text or markdown file listing implemented fixes.",
    )
    round2.set_defaults(func=command_round2)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
