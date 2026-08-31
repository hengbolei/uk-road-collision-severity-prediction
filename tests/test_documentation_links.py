'''Check that local Markdown and Notebook links resolve inside the repository.'''

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')


def _local_targets(text: str, parent: Path):
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.split('#', 1)[0]
        if not target or '://' in target or target.startswith('mailto:'):
            continue
        yield (parent / target).resolve()


def test_local_markdown_links_resolve():
    files = [
        ROOT / 'README.md',
        ROOT / 'README.zh-CN.md',
        ROOT / 'data/raw/README.md',
        ROOT / 'reports/figure_story.md',
        ROOT / 'reports/figure_story_zh-CN.md',
    ]
    missing = []
    for path in files:
        for target in _local_targets(path.read_text(encoding='utf-8'), path.parent):
            if not target.exists():
                missing.append(f'{path.relative_to(ROOT)} -> {target}')
    assert not missing, '\n'.join(missing)


def test_local_notebook_links_resolve():
    missing = []
    for path in (ROOT / 'notebooks').glob('*.ipynb'):
        notebook = json.loads(path.read_text(encoding='utf-8'))
        markdown = '\n'.join(
            ''.join(cell['source'])
            for cell in notebook['cells']
            if cell['cell_type'] == 'markdown'
        )
        for target in _local_targets(markdown, path.parent):
            if not target.exists():
                missing.append(f'{path.relative_to(ROOT)} -> {target}')
    assert not missing, '\n'.join(missing)
