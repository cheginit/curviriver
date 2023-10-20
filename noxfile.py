"""Nox sessions."""
from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import nox

try:
    import tomllib as tomli
except ImportError:
    import tomli


def get_package_name() -> str:
    """Get the name of the package."""
    with Path("pyproject.toml").open("rb") as f:
        return tomli.load(f)["project"]["name"]


def get_extras() -> list[str]:
    """Get the name of the package."""
    with Path("pyproject.toml").open("rb") as f:
        extras = tomli.load(f)["project"]["optional-dependencies"]
    return [e for e in extras if e not in ("test", "typeguard", "doc")]


def get_deps() -> list[str]:
    """Get the name of the package."""
    with Path("pyproject.toml").open("rb") as f:
        return tomli.load(f)["project"]["dependencies"]


python_versions = ["3.9"]
lint_versions = ["3.11"]
package = get_package_name()
nox.options.sessions = (
    "pre-commit",
    "type-check",
    "tests",
)


def install_deps(session: nox.Session, extra: str | None = None) -> None:
    """Install package dependencies."""
    deps = [f".[{extra}]"] if extra else ["."]
    session.install(*deps)
    dirs = [".pytest_cache", "build", "dist", ".eggs"]
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)

    patterns = ["*.egg-info", "*.egg", "*.pyc", "*~", "**/__pycache__"]
    for p in patterns:
        for f in Path.cwd().rglob(p):
            shutil.rmtree(f, ignore_errors=True)


def activate_virtualenv_in_precommit_hooks(session: nox.Session) -> None:
    """Activate virtualenv in hooks installed by pre-commit.

    This function patches git hooks installed by pre-commit to activate the
    session's virtual environment. This allows pre-commit to locate hooks in
    that environment when invoked from git.

    Parameters
    ----------
    session
        The Session object.
    """
    virtualenv = session.env.get("VIRTUAL_ENV")
    if virtualenv is None:
        return

    hookdir = Path(".git") / "hooks"
    if not hookdir.is_dir():
        return

    for hook in hookdir.iterdir():
        if hook.name.endswith(".sample") or not hook.is_file():
            continue

        text = hook.read_text()
        bindir = repr(session.bin)[1:-1]  # strip quotes
        if not (Path("A") == Path("a") and bindir.lower() in text.lower() or bindir in text):
            continue

        lines = text.splitlines()
        if not (lines[0].startswith("#!") and "python" in lines[0].lower()):
            continue

        header = textwrap.dedent(
            f"""\
            import os
            os.environ["VIRTUAL_ENV"] = {virtualenv!r}
            os.environ["PATH"] = os.pathsep.join((
                {session.bin!r},
                os.environ.get("PATH", ""),
            ))
            """
        )

        lines.insert(1, header)
        hook.write_text("\n".join(lines))


@nox.session(name="pre-commit", python=lint_versions)
def pre_commit(session: nox.Session) -> None:
    """Lint using pre-commit."""
    args = session.posargs or ["run", "--all-files"]
    session.install("pre-commit")
    session.run("pre-commit", *args)
    if args and args[0] == "install":
        activate_virtualenv_in_precommit_hooks(session)


@nox.session(name="type-check", python=python_versions)
def type_check(session: nox.Session) -> None:
    """Run Pyright."""
    extras = get_extras()
    install_deps(session, ",".join(extras))
    session.install("pyright")
    session.run("pyright")


@nox.session(python=python_versions)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    install_deps(session, "test")
    session.run("pytest", "--doctest-modules", *session.posargs)
    session.run("coverage", "report")
    session.run("coverage", "html")
