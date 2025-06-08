#!/usr/bin/env python3

import sys

from preprocess_dataoceanai_arabic import normalize_text

with open(sys.argv[1]) as f:
    for line in f:
        print(normalize_text(line.strip()))
