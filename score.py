from pathlib import Path

answers: list[str] = list(
    (Path.cwd().parent / "docmetrics" / "answers.txt").read_text().strip()
)


def score(guesses: str) -> float:
    total = len(answers)
    # assert len(guesses) == len(answers), f"Expected {total} guesses, got {len(guesses)}"
    return sum(guess == answer for guess, answer in zip(guesses, answers)) / total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Score a set of guesses against the answers."
    )
    parser.add_argument(
        "guesses", type=str, help="A string of guesses, one character per answer."
    )
    args = parser.parse_args()

    final_score = score(args.guesses)
    print(f"Final Score: {final_score:.2%}")
