#!/usr/bin/env python3

import sys

with open(sys.argv[1]) as f:
    data = f.read().splitlines()

charset = []
for i in data:
    charset.extend(list(i))

charset = sorted(set(charset))

for i in charset:
    print(i)
