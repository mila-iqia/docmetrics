import questionary
from rich.console import Console

from docmetrics.main import Question

console = Console()


def run_quiz(questions: list[Question]) -> None:
    """Run an interactive multiple-choice quiz in the terminal.

    Presents each question using questionary. The user can exit at any time by
    pressing Ctrl+C or selecting the Quit option.
    """
    total = len(questions)
    score = 0
    answered = 0

    console.print(f"\n[bold]Quiz: {total} question(s)[/bold]")
    console.print("Press [bold]Ctrl+C[/bold] or select [bold]Quit[/bold] to exit.\n")

    for i, question in enumerate(questions, 1):
        console.print(f"[bold][{i}/{total}] {question.question}[/bold]")
        for letter, text in question.options.items():
            console.print(f"  {letter}: {text}")
        console.print()

        choices = [questionary.Choice(title=letter, value=letter) for letter in question.options]
        choices.append(questionary.Choice(title="Quit (q)", value="q"))

        try:
            answer = questionary.select(
                "Your answer:",
                choices=choices,
            ).ask()
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting quiz.[/yellow]")
            _print_score(score, answered)
            return

        if answer is None or answer == "q":
            console.print("\n[yellow]Exiting quiz.[/yellow]")
            _print_score(score, answered)
            return

        answered += 1
        if answer == question.answer:
            score += 1
            console.print(
                f"[green]Correct![/green] ({question.answer}: {question.options[question.answer]})\n"
            )
        else:
            console.print(
                f"[red]Wrong.[/red] Correct answer: {question.answer}: {question.options[question.answer]}\n"
            )

    _print_score(score, answered)


def _print_score(score: int, answered: int) -> None:
    if answered == 0:
        return
    console.print(f"[bold]Score: {score}/{answered}[/bold]")
