from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import html
from pathlib import Path
import re

from latex2mathml.converter import convert

ALGORITHMS = [
    'reinforce',
    'actor-critic',
    'dqn',
    'a2c',
    # 'trpo',
    'ppo',
    'ddpg',
    'c51'
]
MATH_BLOCK = re.compile(r'\\\((.*?)\\\)|\\\[(.*?)\\\]')


@dataclass
class Element:
    tag: str
    attributes: dict[str, str]
    children: list[str | Element]

    def __init__(self,
                 tag: str,
                 attributes: Mapping[str, str] | None = None,
                 contents: Iterable[str, Element] | Element = '') -> None:
        self.tag = tag
        self.attributes = {**attributes} if attributes else {}

        if not contents:
            self.children = []
        elif isinstance(contents, (str, Element)):
            self.children = [contents]
        else:
            self.children = contents

    def __str__(self) -> str:
        attributes = ''.join(f' {k}="{v}"' for k, v in self.attributes.items())
        return (f'<{self.tag}{attributes}>'
                f'{"".join(map(str, self.children))}'
                f'</{self.tag}>')

    def append(self, child: str | Element) -> None:
        self.children.append(child)


@dataclass
class Line:
    text: str
    parity_: bool  # False = even, True = odd
    span: int = 1
    indent: int = 0

    def __post_init__(self) -> None:
        original_length = len(self.text)
        self.text = self.text.lstrip('\t')
        self.indent = original_length - len(self.text)

    @property
    def parity(self) -> str:
        return 'odd' if self.parity_ else 'even'


def wrap(lines: list[Line], class_name: str, tag: str) -> Element:
    article = Element('ul', {'class': class_name})
    last_parity = None
    for line in lines:
        if line.parity != last_parity:
            wrapper = Element('li')
            article.append(wrapper)

        if line.text:
            el = Element(tag, contents=line.text)
            if line.indent:
                el.attributes['class'] = f'indent-{line.indent}'
            wrapper.append(el)

        last_parity = line.parity
    return article


def parse_alg_line(line: str) -> str:
    def convert_match(match):
        if match[1]:
            math = convert(match[1])
        else:
            math = convert(match[2], display='block')
        return (
            math.replace('<mo>&#x0003D;</mo><mo>&#x0003D;</mo>',
                         '<mo>&#x0003D;&#x0003D;</mo>')  # fix == spacing
                .replace('<mo>argmax</mo>',
                         '<mo>arg&thinsp;max</mo>')  # add a space in argmax
        )
    return MATH_BLOCK.sub(convert_match, line)


def parse(
    alg_file: str,
    code_file: str,
    heading_tag: str
) -> tuple[list[Line], list[Line]]:
    with open(alg_file) as f:
        alg_line_texts = [parse_alg_line(line.rstrip()) for line in f]
    assert alg_line_texts[0]
    assert alg_line_texts.pop(1) == ''

    parity = True
    alg_line_no = 1
    alg_lines = [Line(alg_line_texts.pop(0), not parity)]
    code_lines = [Line('{}\n\n...\n\n', not parity)]
    class_line = ''
    with open(code_file) as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line == '# LINE 0':
                break

            if stripped_line.startswith('class '):
                class_line = stripped_line

        assert code_lines
        code_lines[0].text = code_lines[0].text.format(class_line)

        for line in f:
            stripped_line = line.strip()
            if stripped_line == '# END':
                break

            if stripped_line.startswith('# LINE'):
                numbers_string = stripped_line[len('# LINE '):]

                code_lines.append(Line('', parity))
                if not numbers_string:
                    alg_lines.append(Line('', parity))
                    try:
                        if not alg_line_texts[0]:
                            del alg_line_texts[0]
                    except IndexError:
                        pass
                else:
                    numbers = [*map(int, numbers_string.split('-'))]
                    line_start = numbers[0]
                    if line_start != alg_line_no:
                        raise ValueError('Line mismatch: expected '
                                         f'{alg_line_no}, got {line_start}')
                    try:
                        line_end = numbers[1]
                    except IndexError:
                        line_end = line_start

                    while not alg_line_texts[0]:
                        alg_lines.append(Line(alg_line_texts.pop(0), parity))
                        parity = not parity
                        code_lines.append(Line('', parity))
                    code_lines[-1].span = 1 + line_end - line_start
                    while alg_line_no <= line_end:
                        if alg_line_texts[0]:
                            alg_line_no += 1
                        alg_lines.append(Line(alg_line_texts.pop(0), parity))
                parity = not parity
            elif not stripped_line:
                code_lines[-1].text += '\n'
            else:
                # reindent line with 2 spaces instead of 4
                unindented_line = line.lstrip()
                indent_level = (len(line) - len(unindented_line)) // 4 - 1
                escaped_line = html.escape(unindented_line)
                code_lines[-1].text += '  ' * indent_level + escaped_line

    if alg_line_texts:
        code_lines.append(Line('', parity))
        for line in alg_line_texts:
            alg_lines.append(Line(line, parity))

    # remove blank lines at end of code lines
    for line in code_lines:
        line.text = line.text.rstrip()

    return alg_lines, code_lines


def main() -> None:
    toc = Element('menu')  # table of contents

    articles = []
    for algorithm in ALGORITHMS:
        directory = Path(algorithm)
        intro_file = directory / 'intro.html'
        alg_file = directory / 'algorithm.tex'
        code_file = directory / f'{algorithm}.py'

        with open(intro_file) as f:
            title = f.readline().strip()
            intro = f.read()

        alg_lines, code_lines = parse(alg_file, code_file, 'h3')

        # construct article
        articles.append(
            Element(
                'article',
                contents=[
                    Element(
                        'h2',
                        {'id': algorithm},
                        Element(
                            'a',
                            {'href': f'#{algorithm}'},
                            title
                        )
                    ),
                    intro,
                    Element(
                        'section',
                        contents=[
                            Element('h3', contents=alg_lines[0].text),
                            wrap(alg_lines[1:], 'alg', 'p'),
                            wrap(code_lines, 'code', 'pre')
                        ]
                    )
                ]
            )
        )

        # link in table of contents
        toc.append(
            Element(
                'li',
                contents=Element('a', {'href': f'#{algorithm}'}, title)
            )
        )

    with open('index-template.html') as f:
        doc = f.read()
    doc = (doc.replace('{% TABLE OF CONTENTS %}', str(toc))
              .replace('{% ALGORITHMS %}', ''.join(map(str, articles))))
    print(doc)


if __name__ == '__main__':
    main()
