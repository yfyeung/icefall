#!/usr/bin/env python3


import argparse

import regex as re


def asr_text_post_processing(text: str) -> str:
    """
    Modified from: https://github.com/natural-language-processing-elm/open_universal_arabic_asr_leaderboard/blob/main/eval.py
    """
    # remove punctuation and symbols
    text = re.sub(r"[\p{p}\p{s}]", "", text)

    # remove diacritics
    diacritics = r"[\u064b-\u0652]"  # arabic diacritical marks (fatha, damma, etc.)
    text = re.sub(diacritics, "", text)

    # Normalize Hamzas and Maddas
    text = re.sub("پ", "ب", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub(r"[آ]", "ا", text)
    text = re.sub(r"[أإ]", "ا", text)
    text = re.sub(r"[ؤ]", "و", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"[ء]", "", text)

    # Transliterate Eastern Arabic numerals to Western Arabic numerals
    eastern_to_western_numerals = {
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
    for eastern, western in eastern_to_western_numerals.items():
        text = text.replace(eastern, western)

    # remove tatweel (kashida, u+0640)
    text = re.sub(r"\u0640", "", text)

    # normalize multiple whitespace characters into a single space
    text = re.sub(r"\s\s+", " ", text)

    return text.upper().strip()
