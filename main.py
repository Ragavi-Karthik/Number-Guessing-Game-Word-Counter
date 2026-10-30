#!/usr/bin/env python3
"""
Number Guessing Game & Word Counter  (intermediate / advanced version)

Features
--------
Guessing game
  * 3 difficulty levels with attempt limits
  * "hot / cold" proximity hints, duplicate-guess detection
  * Score based on attempts used and time taken
  * Persistent per-difficulty leaderboard (JSON, atomic writes)
  * Reproducible games with --seed

Word counter
  * Streams files line by line (works on large files)
  * Regex tokenizer (handles don't, well-known, numbers)
  * Lines, words, characters, unique words, average / longest word
  * Top-N frequent words, optional stop-word filter, min word length
  * Search for one word, combined stats for multiple files
  * Export report to JSON or CSV

Usage
-----
  python game_toolkit.py                       # interactive menu
  python game_toolkit.py play -d hard --seed 7
  python game_toolkit.py leaderboard -d medium
  python game_toolkit.py wordcount a.txt b.txt -n 15 --export report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

LEADERBOARD_FILE = Path(__file__).with_name("leaderboard.json")

WORD_RE = re.compile(r"[a-z0-9]+(?:['’\-][a-z0-9]+)*")

STOP_WORDS = frozenset(
    """a an and are as at be but by for from has have he her his i if in is it its
    me my no not of on or our she so that the their them they this to us was we
    were what when which who will with you your""".split()
)


# ======================================================================
# Shared input helpers
# ======================================================================
def read_int(prompt: str, low: int, high: int) -> int | None:
    """Ask until a valid integer in [low, high] is entered. 'q' returns None."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"q", "quit"}:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("  Enter a whole number (or 'q' to quit).")
            continue
        if not low <= value <= high:
            print(f"  Pick a number between {low} and {high}.")
            continue
        return value


def ask_yes_no(prompt: str) -> bool:
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("  Please type 'y' or 'n'.")


# ======================================================================
# Part 1: Number guessing game
# ======================================================================
@dataclass(frozen=True)
class Difficulty:
    name: str
    low: int
    high: int
    max_attempts: int
    multiplier: float


DIFFICULTIES: dict[str, Difficulty] = {
    "easy": Difficulty("easy", 1, 50, 8, 1.0),
    "medium": Difficulty("medium", 1, 100, 7, 1.5),
    "hard": Difficulty("hard", 1, 500, 9, 2.5),  # 9 = optimal binary search
}


@dataclass
class GameResult:
    won: bool
    secret: int
    attempts: int
    seconds: float
    score: int
    difficulty: str


class GuessingGame:
    def __init__(self, difficulty: Difficulty, rng: random.Random | None = None):
        self.difficulty = difficulty
        self.rng = rng or random.Random()

    @staticmethod
    def _temperature(distance: int, span: int) -> str:
        ratio = distance / span
        if ratio <= 0.02:
            return "Burning hot!"
        if ratio <= 0.08:
            return "Hot."
        if ratio <= 0.20:
            return "Warm."
        if ratio <= 0.40:
            return "Cool."
        return "Cold."

    def _score(self, attempts: int, elapsed: float) -> int:
        d = self.difficulty
        efficiency = (d.max_attempts - attempts + 1) / d.max_attempts
        return max(10, int(1000 * d.multiplier * efficiency) - int(elapsed))

    def play(self) -> GameResult | None:
        d = self.difficulty
        secret = self.rng.randint(d.low, d.high)
        guesses: list[int] = []
        span = d.high - d.low
        start = time.perf_counter()

        print(f"\nI'm thinking of a number between {d.low} and {d.high}.")
        print(f"You have {d.max_attempts} attempts. Type 'q' to give up.\n")

        while len(guesses) < d.max_attempts:
            remaining = d.max_attempts - len(guesses)
            guess = read_int(f"[{remaining} left] Your guess: ", d.low, d.high)
            if guess is None:
                print(f"Giving up? The number was {secret}.")
                return None
            if guess in guesses:
                print("  You already tried that one.")
                continue

            guesses.append(guess)

            if guess == secret:
                elapsed = time.perf_counter() - start
                score = self._score(len(guesses), elapsed)
                print(f"\n Correct! {secret} in {len(guesses)} attempt(s), "
                      f"{elapsed:.1f}s. Score: {score}")
                return GameResult(True, secret, len(guesses), round(elapsed, 1),
                                  score, d.name)

            direction = "high" if guess > secret else "low"
            print(f"  Too {direction}. {self._temperature(abs(guess - secret), span)}")

        elapsed = time.perf_counter() - start
        print(f"\nOut of attempts! The number was {secret}.")
        return GameResult(False, secret, len(guesses), round(elapsed, 1), 0, d.name)


class Leaderboard:
    def __init__(self, path: Path = LEADERBOARD_FILE, size: int = 10):
        self.path = path
        self.size = size

    def load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save(self, entries: list[dict]) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            tmp.replace(self.path)  # atomic swap: no half-written file
        except OSError as exc:
            print(f"  Warning: could not save leaderboard ({exc}).")

    @staticmethod
    def _sort_key(entry: dict) -> tuple:
        return (-entry["score"], entry["seconds"])

    def add(self, name: str, result: GameResult) -> int | None:
        """Add a winning result. Returns the rank (1-based) or None if not in top N."""
        entry = {
            "name": name,
            "score": result.score,
            "difficulty": result.difficulty,
            "attempts": result.attempts,
            "seconds": result.seconds,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        entries = self.load()
        entries.append(entry)

        grouped: dict[str, list[dict]] = {}
        for e in entries:
            grouped.setdefault(e["difficulty"], []).append(e)

        kept: list[dict] = []
        rank: int | None = None
        for diff, items in grouped.items():
            items.sort(key=self._sort_key)
            if diff == result.difficulty and entry in items[: self.size]:
                rank = items.index(entry) + 1
            kept.extend(items[: self.size])

        self._save(kept)
        return rank

    def show(self, difficulty: str | None = None) -> None:
        entries = [e for e in self.load()
                   if difficulty is None or e["difficulty"] == difficulty]
        if not entries:
            print("\nNo scores yet. Play a game first!")
            return
        entries.sort(key=lambda e: (e["difficulty"], *self._sort_key(e)))
        print(f"\n{'#':<3}{'Name':<14}{'Level':<8}{'Score':>6}{'Tries':>7}{'Time':>8}  Date")
        print("-" * 62)
        current, n = None, 0
        for e in entries:
            if e["difficulty"] != current:
                current, n = e["difficulty"], 0
            n += 1
            print(f"{n:<3}{e['name'][:13]:<14}{e['difficulty']:<8}{e['score']:>6}"
                  f"{e['attempts']:>7}{e['seconds']:>7.1f}s  {e['date']}")


def choose_difficulty() -> Difficulty | None:
    names = list(DIFFICULTIES)
    print("\nChoose difficulty:")
    for i, key in enumerate(names, 1):
        d = DIFFICULTIES[key]
        print(f"  {i}. {d.name.title():<7} {d.low}-{d.high}, {d.max_attempts} attempts")
    idx = read_int("Option: ", 1, len(names))
    return DIFFICULTIES[names[idx - 1]] if idx else None


def run_game(difficulty_name: str | None, seed: int | None, name: str | None) -> None:
    board = Leaderboard()
    name = name or input("Your name: ").strip() or "Player"

    while True:
        difficulty = DIFFICULTIES[difficulty_name] if difficulty_name else choose_difficulty()
        if difficulty is None:
            return
        rng = random.Random(seed)
        result = GuessingGame(difficulty, rng).play()

        if result and result.won:
            rank = board.add(name, result)
            if rank:
                print(f"  You're #{rank} on the {difficulty.name} leaderboard!")

        if not ask_yes_no("\nPlay again? (y/n): "):
            return
        seed = None  # a fixed seed would repeat the same number


# ======================================================================
# Part 2: Word counter
# ======================================================================
@dataclass
class FileStats:
    path: str
    lines: int = 0
    words: int = 0
    characters: int = 0
    unique_words: int = 0
    average_word_length: float = 0.0
    longest_word: str = ""
    top_words: list = field(default_factory=list)


def top_words(counts: Counter, n: int, ignore_stop: bool, min_length: int) -> list[tuple[str, int]]:
    filtered = Counter({
        w: c for w, c in counts.items()
        if len(w) >= min_length and not (ignore_stop and w in STOP_WORDS)
    })
    # Sort by count desc, then alphabetically so output is deterministic
    return sorted(filtered.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def analyze_file(path: Path, n: int, ignore_stop: bool, min_length: int) -> tuple[FileStats, Counter]:
    """Stream the file line by line so memory use stays small."""
    counts: Counter = Counter()
    lines = chars = total_letters = 0
    longest = ""

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            chars += len(line)
            for token in WORD_RE.findall(line.lower()):
                counts[token] += 1
                total_letters += len(token)
                if len(token) > len(longest):
                    longest = token

    total_words = sum(counts.values())
    stats = FileStats(
        path=str(path),
        lines=lines,
        words=total_words,
        characters=chars,
        unique_words=len(counts),
        average_word_length=round(total_letters / total_words, 2) if total_words else 0.0,
        longest_word=longest,
        top_words=top_words(counts, n, ignore_stop, min_length),
    )
    return stats, counts


def print_report(stats: FileStats) -> None:
    print(f"\n=== {stats.path} ===")
    print(f"  Lines:           {stats.lines}")
    print(f"  Words:           {stats.words}")
    print(f"  Characters:      {stats.characters}")
    print(f"  Unique words:    {stats.unique_words}")
    print(f"  Avg word length: {stats.average_word_length}")
    print(f"  Longest word:    {stats.longest_word or '-'}")
    if stats.top_words:
        print(f"  Top {len(stats.top_words)} words:")
        width = max(len(w) for w, _ in stats.top_words)
        biggest = stats.top_words[0][1]
        for rank, (word, count) in enumerate(stats.top_words, 1):
            bar = "█" * max(1, round(20 * count / biggest))
            print(f"    {rank:>2}. {word:<{width}} {count:>5}  {bar}")


def export_report(path: Path, reports: list[FileStats]) -> None:
    suffix = path.suffix.lower()
    if suffix == ".json":
        path.write_text(json.dumps([asdict(r) for r in reports], indent=2, ensure_ascii=False),
                        encoding="utf-8")
    elif suffix == ".csv":
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file", "rank", "word", "count"])
            for r in reports:
                for rank, (word, count) in enumerate(r.top_words, 1):
                    writer.writerow([r.path, rank, word, count])
    else:
        raise ValueError("Export file must end with .json or .csv")
    print(f"\nReport exported to {path}")


def run_wordcount(files: list[Path], n: int, keep_stop: bool, min_length: int,
                  find: str | None, export: Path | None) -> int:
    reports: list[FileStats] = []
    combined: Counter = Counter()
    exit_code = 0

    for path in files:
        try:
            stats, counts = analyze_file(path, n, not keep_stop, min_length)
        except FileNotFoundError:
            print(f"Error: '{path}' not found.", file=sys.stderr)
        except IsADirectoryError:
            print(f"Error: '{path}' is a folder, not a file.", file=sys.stderr)
        except PermissionError:
            print(f"Error: no permission to read '{path}'.", file=sys.stderr)
        except OSError as exc:
            print(f"Error reading '{path}': {exc}", file=sys.stderr)
        else:
            reports.append(stats)
            combined.update(counts)
            print_report(stats)
            continue
        exit_code = 1

    if len(reports) > 1:
        total = FileStats(
            path=f"TOTAL ({len(reports)} files)",
            lines=sum(r.lines for r in reports),
            words=sum(r.words for r in reports),
            characters=sum(r.characters for r in reports),
            unique_words=len(combined),
            top_words=top_words(combined, n, not keep_stop, min_length),
        )
        print_report(total)

    if find:
        word = find.lower()
        print(f"\n'{word}' appears {combined.get(word, 0)} time(s) in total.")

    if export and reports:
        try:
            export_report(export, reports)
        except (ValueError, OSError) as exc:
            print(f"Export failed: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


# ======================================================================
# CLI
# ======================================================================
def interactive_menu() -> None:
    board = Leaderboard()
    while True:
        print("\n=== Game & Word Toolkit ===")
        print("  1. Play number guessing game")
        print("  2. View leaderboard")
        print("  3. Word counter")
        print("  4. Exit")
        choice = read_int("Option: ", 1, 4)

        if choice == 1:
            run_game(None, None, None)
        elif choice == 2:
            board.show()
        elif choice == 3:
            raw = input("File path(s), separated by spaces: ").split()
            if raw:
                run_wordcount([Path(p) for p in raw], 10, False, 1, None, None)
        else:
            print("Goodbye!")
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Number guessing game & word counter")
    sub = parser.add_subparsers(dest="command")

    play = sub.add_parser("play", help="play the guessing game")
    play.add_argument("-d", "--difficulty", choices=DIFFICULTIES)
    play.add_argument("--seed", type=int, help="fixed seed for a reproducible number")
    play.add_argument("--name", help="name for the leaderboard")

    lb = sub.add_parser("leaderboard", help="show high scores")
    lb.add_argument("-d", "--difficulty", choices=DIFFICULTIES)

    wc = sub.add_parser("wordcount", help="analyse text files")
    wc.add_argument("files", nargs="+", type=Path)
    wc.add_argument("-n", "--top", type=int, default=10, help="how many top words to show")
    wc.add_argument("--keep-stopwords", action="store_true", help="include words like 'the', 'and'")
    wc.add_argument("--min-length", type=int, default=1, help="ignore shorter words in the top list")
    wc.add_argument("--find", metavar="WORD", help="count occurrences of one word")
    wc.add_argument("--export", type=Path, help="save report as .json or .csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "play":
            run_game(args.difficulty, args.seed, args.name)
        elif args.command == "leaderboard":
            Leaderboard().show(args.difficulty)
        elif args.command == "wordcount":
            return run_wordcount(args.files, args.top, args.keep_stopwords,
                                 args.min_length, args.find, args.export)
        else:
            interactive_menu()
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())