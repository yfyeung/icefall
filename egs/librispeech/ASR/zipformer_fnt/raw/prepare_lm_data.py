import gzip
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str)
parser.add_argument("--output-file", type=str)
args = parser.parse_args()

with gzip.open(args.input_file, 'rb') as f:
    lines = f.readlines()
    lines = [line.decode() for line in lines]
    lines = [json.loads(item)['supervisions'][0]['text'] for item in lines]
    with open(args.output_file, "w") as g:
        g.write("\n".join(lines))
    # file_content = f.read()