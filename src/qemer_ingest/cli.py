import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def _root() -> None:
    """Build local documentation corpora from public GitHub repositories."""


def main() -> None:
    app()
