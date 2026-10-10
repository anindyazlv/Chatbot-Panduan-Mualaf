import re

# Arabic script (including presentation forms) and leftover ligature characters.
NOISE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
    r'\uFB50-\uFDFF\uFE70-\uFEFF\u0C80-\u0CFF]+'
)
DIGIT_LINE = re.compile(r'^\s*\d{1,3}\s*$')


def clean_page(text):
    text = NOISE.sub(' ', text)
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
    joined = ' '.join(
        line for line in lines if line and not DIGIT_LINE.match(line)
    )

    if joined.count('....') > 3:  # table of contents page
        return ''

    return joined if len(joined) >= 200 else ''
