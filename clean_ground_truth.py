from pathlib import Path
import re
from typing import Generator, Union

import pdftotext

from preprocessing import MEDIA_ROOT


def fix_lithuanian_letters(text: str):
    replacements = {
        "Á": "Į",
        "è": "č",
        "é": "č",
        "ë": "ė",
        "ø": "ų",
        "Ø": "Ų",
        "¥": "Ą",
        "û": "ū",
    }
    for key, replacement in replacements.items():
        text = text.replace(key, replacement)
    return text


def replace_consecutive_whitespace(text: str) -> str:
    lines = text.split("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in lines]
    return "\n".join(lines)


def process(filepath: Union[str, Path]) -> Generator[str, None, None]:
    with open(str(MEDIA_ROOT / filepath), "rb") as f:
        pdf = pdftotext.PDF(f)
    ground_truth = "\n".join(pdf)
    ground_truth = fix_lithuanian_letters(ground_truth)
    ground_truth = replace_consecutive_whitespace(ground_truth)
    yield ground_truth
