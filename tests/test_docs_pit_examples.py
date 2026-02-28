from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "guides" / "pit.md"


def _extract_python_doc_examples(markdown: str) -> list[str]:
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    blocks = [match.group(1).strip() for match in pattern.finditer(markdown)]
    return [block for block in blocks if block.startswith("# docs-example:")]


def test_pit_docs_examples_execute(tmp_path) -> None:
    markdown = DOC_PATH.read_text(encoding="utf-8")
    examples = _extract_python_doc_examples(markdown)
    assert examples, "No docs examples found in docs/guides/pit.md"

    for idx, code in enumerate(examples):
        namespace = {"TMP_DIR": tmp_path / f"docs_example_{idx}"}
        exec(compile(code, f"{DOC_PATH}:{idx + 1}", "exec"), namespace, namespace)
