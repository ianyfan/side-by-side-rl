import html
import re
import sys

from bs4 import BeautifulSoup as bs
from bs4.formatter import HTMLFormatter
from latex2mathml.converter import convert

math_block = re.compile(r'\\\((.*?)\\\)')


def parse_line(line: str) -> str:
    def convert_match(match):
        return convert(match[1])
    return math_block.sub(convert_match, line).replace('\t', '&emsp;')


FILE = sys.argv[1]
with open(FILE) as f:
    lines = [parse_line(line[:-1]) for line in f]
assert lines[0]
assert lines.pop(1) == ''

tagged_lines = [f'<h2 class="line-0">{lines[0]}</h2>']
line_no = 1
for line in lines[1:]:
    if line:
        tagged_lines.append(
            f'<div class="line-{line_no}"><p>{line}</p></div>'
        )
        line_no += 1
    else:
        tagged_lines.append('<div></div>')

output = (tagged_lines[0]
          + '<article>'
          + ''.join(tagged_lines[1:])
          + '</article>')

soup = bs(output, features='html.parser')


def replace_em(s: str):
    return html.escape(s).replace('\u2003', '&emsp;')


formatter = HTMLFormatter(replace_em)

print(soup.prettify(formatter=formatter))
