import argparse
import os
import sys
import warnings

from keywords import extract_keywords

DEFAULT_WINDOW_SIZE = 4
DEFAULT_TOTAL_WORDS = 0
DEFAULT_DAMPING_FACTOR = 0.85

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("{} not in range [0.0, 1.0]".format(x))
    return x

def parse_args(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, prog="textrank", description="Extract most relevant keywords of a given text using the TextRank algorithm.")

    parser.add_argument('textfile', metavar="path/to/file", type=argparse.FileType('r'), nargs="+")
    parser.add_argument('--stop_words', '-s', metavar="list,of,words",
                            help="Either a string of comma separated stopwords or a path to a file which has comma separated stopwords in every line")
    parser.add_argument('--window', '-w', metavar="#window", type=int, default=DEFAULT_WINDOW_SIZE,
                            help="Window size for token pairs")
    parser.add_argument('--total', '-t', metavar="#total", type=int, default=DEFAULT_TOTAL_WORDS,
                            help="Total number of words to be ranked"),
    parser.add_argument('--damp', '-d', metavar="d", type=restricted_float, default=DEFAULT_DAMPING_FACTOR,
                            help="Float number (0,1] that defines the length of the summary. It's a proportion of the original text")
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])

    stop_words = args.stop_words
    if stop_words:
        if os.path.exists(stop_words):
            with open(stop_words, "r") as f:
                stop_words = {word for line in f.readlines() for word in line.strip().split(",")}
        else:
            stop_words = stop_words.split(",")
    else:
        warnings.warn("Stop words were not provided or recognized, hence empty list is passed.")
        stop_words = []
    
    for f in args.textfile:
        print("-"*10, f"{f.name}", "-"*10)
        keywords = extract_keywords(f.read(), stop_words=stop_words, window=args.window, d=args.damp, total=args.total,)
        print("-"*35, "\nKeywords: ", end="")
        for word in keywords:
            print(f"{word[0].title()}, ", end="")
        print("\n\n")

if __name__ == "__main__":
    main()