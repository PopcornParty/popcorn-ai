"""Restricted file access for isolated candidate workspaces.

This is a local workspace primitive, not an OS sandbox. The caller must
provide a private temporary directory and prevent concurrent untrusted
processes from mutating it while operations are in progress.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath

MAX_FILE_BYTES = 1_000_000
ALLOWED_SUFFIXES = frozenset({
    ".py", ".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".txt",
    ".css", ".html", ".yml", ".yaml", ".toml", ".ini", ".cfg",
})
BLOCKED_COMPONENTS = frozenset({
    ".git", ".env", ".ssh", ".aws", ".config", ".cache", ".venv",
    "venv", "node_modules", "__pycache__", "secrets", "credentials",
})


class WorkspaceError(ValueError):
    """Raised when a candidate workspace operation violates its boundary."""


class CandidateWorkspace:
    def __init__(self, root: str | Path, *, max_file_bytes: int = MAX_FILE_BYTES):
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise WorkspaceError("Workspace root must be an existing directory")
        if root_path.is_symlink():
            raise WorkspaceError("Workspace root must not be a symlink")
        if not isinstance(max_file_bytes, int) or isinstance(max_file_bytes, bool) or not 1 <= max_file_bytes <= MAX_FILE_BYTES:
            raise WorkspaceError("Invalid workspace file-size limit")
        self.root = root_path.resolve(strict=True)
        self.max_file_bytes = max_file_bytes

    def _target(self, relative_path: str) -> Path:
        if not isinstance(relative_path, str) or not relative_path or len(relative_path) > 240:
            raise WorkspaceError("A short relative file path is required")
        if "\x00" in relative_path or "\\" in relative_path:
            raise WorkspaceError("Invalid path characters")
        pure = PurePosixPath(relative_path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise WorkspaceError("Path must stay inside the candidate workspace")
        if any(part.startswith(".") or part.casefold() in BLOCKED_COMPONENTS for part in pure.parts):
            raise WorkspaceError("Hidden, configuration and credential paths are blocked")
        if pure.suffix.casefold() not in ALLOWED_SUFFIXES:
            raise WorkspaceError("File type is not allowlisted")

        target = self.root.joinpath(*pure.parts)
        current = self.root
        for part in pure.parts:
            current = current / part
            if current.is_symlink():
                raise WorkspaceError("Symlinks are not allowed in the candidate workspace")
        resolved = target.resolve(strict=False)
        if not resolved.is_relative_to(self.root):
            raise WorkspaceError("Path escapes the candidate workspace")
        return target

    def read_text(self, relative_path: str) -> str:
        target = self._target(relative_path)
        if not target.exists() or not target.is_file():
            raise WorkspaceError("Candidate file does not exist")
        if target.stat().st_size > self.max_file_bytes:
            raise WorkspaceError("Candidate file exceeds the size limit")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceError("Candidate file is not valid UTF-8 text") from exc

    def write_text(self, relative_path: str, content: str) -> None:
        target = self._target(relative_path)
        if not isinstance(content, str):
            raise WorkspaceError("Candidate content must be text")
        encoded = content.encode("utf-8")
        if len(encoded) > self.max_file_bytes:
            raise WorkspaceError("Candidate content exceeds the size limit")
        parent = target.parent
        parent.mkdir(parents=True, exist_ok=True)
        # Re-check after directory creation before touching the destination.
        checked = self._target(relative_path)
        if checked != target or parent.is_symlink() or not parent.resolve().is_relative_to(self.root):
            raise WorkspaceError("Candidate path changed during validation")
        if target.exists() and not target.is_file():
            raise WorkspaceError("Only regular text files may be replaced")
        target.write_bytes(encoded)

    def list_files(self) -> list[str]:
        found = []
        for path in self.root.rglob("*"):
            relative = path.relative_to(self.root).as_posix()
            try:
                checked = self._target(relative)
            except WorkspaceError:
                continue
            if checked.is_file() and not checked.is_symlink():
                found.append(relative)
        return sorted(found)
