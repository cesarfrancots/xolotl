import argparse
import csv


def positive_int(text):
    n = int(text)  # a ValueError here is reported by argparse as an invalid value
    if n <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return n


def main():
    parser = argparse.ArgumentParser(description="Print scores, highest first.")
    parser.add_argument("path", help="CSV file with name,score columns")
    parser.add_argument("--limit", type=positive_int, metavar="N", help="print only the top N rows")
    args = parser.parse_args()
    with open(args.path, newline="", encoding="utf-8") as f:
        rows = [(r["name"], int(r["score"])) for r in csv.DictReader(f)]
    for name, score in sorted(rows, key=lambda r: r[1], reverse=True)[:args.limit]:
        print(name, score)


if __name__ == "__main__":
    main()
