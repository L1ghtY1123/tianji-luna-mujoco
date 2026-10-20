"""Bundle this project only; never include model credentials, environments, or user runs."""
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / "deploy-bundle.tar.gz"
excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "data"}


def executable_scripts(member):
    # Windows does not retain executable bits; Ubuntu launchers need them after extraction.
    if member.name.endswith(".sh"):
        member.mode = 0o755
    return member


with tarfile.open(target, "w:gz") as archive:
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if not path.is_file() or any(part in excluded or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path == target or path.name == ".env" or path.suffix in {".log", ".pyc"}:
            continue
        archive.add(path, arcname=rel.as_posix(), filter=executable_scripts)
print(str(target))
