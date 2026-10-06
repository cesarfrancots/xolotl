import argparse
import csv


def main():
    parser = argparse.ArgumentParser(description="Print scores, highest first.")
    parser.add_argument("path", help="CSV file with name,score columns")
    args = parser.parse_args()
    with open(args.path, newline="", encoding="utf-8") as f:
        rows = [(r["name"], int(r["score"])) for r in csv.DictReader(f)]
    for name, score in sorted(rows, key=lambda r: r[1], reverse=True):
        print(name, score)


if __name__ == "__main__":
    main()
