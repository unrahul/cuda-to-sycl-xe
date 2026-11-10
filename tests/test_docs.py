"""Structural checks for the skill's documentation.

Contract:
- every ```cpp block in SKILL.md and references/ is a verbatim snippet region from examples/,
  so documented code is code that compiles and runs in tests/snippets_test.cpp;
- every relative link resolves, and every reference is reachable from SKILL.md;
- frontmatter and size stay within skill-loader limits;
- no maintainer paths, harness wording, or provenance (where a pattern was learned) in shipped text;
- no fact is written twice: each lives in one place and other pages link to it, so an edit
  cannot leave a stale copy behind (no shared 10-word run, no near-identical sentences).
"""
import re
import unittest
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / 'SKILL.md'
REFERENCES = sorted((ROOT / 'references').glob('*.md'))
SHIPPED_MD = [SKILL, *REFERENCES, *(ROOT / 'examples').rglob('*.md'), *(ROOT / 'templates').glob('*.md')]

SNIPPET = re.compile(r'^[ \t]*// snippet: [\w-]+\n(.*?)^[ \t]*// end snippet', re.M | re.S)
CPP_BLOCK = re.compile(r'^```cpp\n(.*?)^```', re.M | re.S)
LINK = re.compile(r'\]\(([^)\s]+)\)')
FORBIDDEN = ('/home/', 'internal_references', 'internal_tools', 'managed assessment', 'mailbox',
             'candidate policy', 'assessment budget',
             # the skill states patterns; where they were learned lives in maintainer files
             'llama.cpp', 'vllm', 'sglang', 'sgl-kernel', 'llm-scaler', 'xe-forge', 'strata', 'torch-xpu-ops',
             'crk-hacc', 'tensorfold', 'pilot', 'measured on', 'verified on', 'hot chips', 'servethehome')

MAX_SKILL_LINES, MAX_SKILL_WORDS, MAX_REFERENCE_WORDS = 200, 1500, 2500
RUN, SIMILAR = 10, 0.6  # repeated word run length; Jaccard threshold for near-duplicate sentences
STOP = set('the a an and or of to in on for with is are be by as at it its this that from not no '
           'use only than then when if into per each more can does do but you your which'.split())
PROSE = [SKILL, *(p for p in REFERENCES if p.name != 'sources.md')]  # sources.md is generated


def snippet_regions():
    sources = [*ROOT.glob('examples/**/*.hpp'), *ROOT.glob('examples/**/*.cpp')]
    return {m.group(1).strip() for path in sources for m in SNIPPET.finditer(path.read_text())}


def relative_links(path):
    for target in LINK.findall(path.read_text()):
        if '://' not in target and not target.startswith(('#', 'mailto:')):
            yield (path.parent / target.split('#')[0]).resolve()


def prose_units():
    """((file, line), words) for every sentence or table cell outside code blocks."""
    for path in PROSE:
        text = re.sub(r'```.*?```', lambda m: '\n' * m.group(0).count('\n'), path.read_text(), flags=re.S)
        text = re.sub(r'\]\([^)]*\)', ']', text)
        for number, line in enumerate(text.splitlines(), 1):
            for piece in re.split(r'(?<=[.!?])\s+|\|', line):
                words = re.findall(r"[a-z0-9_+.<>-]+", piece.lower())
                if words:
                    yield (path.name, number), words


def frontmatter(text):
    match = re.match(r'---\n(.*?)\n---\n', text, re.S)
    return dict(line.split(': ', 1) for line in match.group(1).splitlines()) if match else {}


class Docs(unittest.TestCase):
    def test_code_blocks_are_compiled_snippets(self):
        regions = snippet_regions()
        for path in (SKILL, *REFERENCES):
            for block in CPP_BLOCK.findall(path.read_text()):
                with self.subTest(file=path.name, block=block.splitlines()[0]):
                    self.assertIn(block.strip(), regions, 'copy the block from an examples/ snippet region')

    def test_links_resolve(self):
        for path in SHIPPED_MD:
            for target in relative_links(path):
                with self.subTest(file=str(path.relative_to(ROOT)), target=str(target)):
                    self.assertTrue(target.exists())

    def test_references_reachable_from_skill(self):
        seen, todo = set(), [SKILL.resolve()]
        while todo:
            page = todo.pop()
            if page in seen or page.suffix != '.md' or not page.exists():
                continue
            seen.add(page)
            todo.extend(relative_links(page))
        orphans = [p.name for p in REFERENCES if p.resolve() not in seen]
        self.assertEqual(orphans, [])

    def test_frontmatter_and_size(self):
        text = SKILL.read_text()
        meta = frontmatter(text)
        self.assertEqual(meta.get('name'), 'cuda-to-sycl-xe')
        description = meta.get('description', '')
        self.assertTrue(0 < len(description) <= 1024)
        self.assertNotRegex(description, '[<>]')
        self.assertLessEqual(len(text.splitlines()), MAX_SKILL_LINES)
        self.assertLessEqual(len(text.split()), MAX_SKILL_WORDS)
        for path in REFERENCES:
            with self.subTest(file=path.name):
                self.assertLessEqual(len(path.read_text().split()), MAX_REFERENCE_WORDS)

    def test_each_fact_written_once(self):
        units = list(prose_units())
        runs = Counter(' '.join(w[i:i + RUN]) for _, w in units for i in range(len(w) - RUN + 1))
        repeated = sorted(run for run, n in runs.items() if n > 1)
        bags = [(name, frozenset(w) - STOP) for name, w in units]
        bags = [(name, bag) for name, bag in bags if len(bag) >= 7]
        near = [(a, b, ' '.join(sorted(x & y))) for (a, x), (b, y) in combinations(bags, 2)
                if a != b and len(x & y) / len(x | y) >= SIMILAR]  # cells of one table row differ by design
        self.assertEqual(repeated, [], 'the same 10-word run appears twice; keep one and link to it')
        self.assertEqual(near, [], 'near-identical sentences; keep one and link to it')

    def test_no_private_or_harness_text(self):
        for path in SHIPPED_MD:
            text = path.read_text().lower()
            for term in FORBIDDEN:
                with self.subTest(file=str(path.relative_to(ROOT)), term=term):
                    self.assertNotIn(term, text)


if __name__ == '__main__':
    unittest.main()
