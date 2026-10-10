import re
import unicodedata

# Arabic script (including presentation forms) and leftover ligature characters.
NOISE = re.compile(
    r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF'
    r'\uFB50-\uFDFF\uFE70-\uFEFF\u0C80-\u0CFF]+'
)
DIGIT_LINE = re.compile(r'^\s*\d{1,3}\s*$')

CHAR_FIXES = {
    'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
    '’': "'", '‘': "'", '“': '"', '”': '"',
}

WORD_FIXES = {
    'Qur√®n': 'Quran', 'Qur√¥n': 'Quran', 'qur√®n': 'quran',
    'if~¥r': 'iftar',
    'tar¥wÏ^': 'tarawih',
    'Rama\\¥n': 'Ramadan',
    'mu¤ammad': 'Muhammad',
    'I^s¥n': 'Ihsan',
}


def fix_text(text):
    for bad, good in WORD_FIXES.items():
        text = text.replace(bad, good)
    for bad, good in CHAR_FIXES.items():
        text = text.replace(bad, good)
    text = re.sub(r'(\w)- (\w)', r'\1\2', text)  # rejoin hyphenated line breaks
    return unicodedata.normalize('NFKC', text)


def clean_page(text):
    text = NOISE.sub(' ', text)
    lines = [re.sub(r'\s+', ' ', line).strip() for line in text.splitlines()]
    joined = ' '.join(
        line for line in lines if line and not DIGIT_LINE.match(line)
    )
    joined = fix_text(joined)

    if joined.count('....') > 3:  # table of contents page
        return ''

    return joined if len(joined) >= 200 else ''
