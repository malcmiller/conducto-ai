"""Inspect immutable conducto-ai release artifacts and produce attestable metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_PARTS = {
    ".git",
    ".github",
    ".venv",
    "__pycache__",
    "tests",
    "dist",
    "credentials",
    ".env",
}
REQUIRED_PACKAGE_PREFIX = "conducto/"


def _project_metadata() -> tuple[str, str, list[str], dict[str, list[str]]]:
    """Read canonical package metadata from ``pyproject.toml``."""
    import tomllib

    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    return (
        str(project["name"]),
        str(project["version"]),
        [str(item) for item in project["dependencies"]],
        {
            str(name): [str(item) for item in values]
            for name, values in project.get("optional-dependencies", {}).items()
        },
    )


def _artifact_members(path: Path) -> list[str]:
    """Return sorted archive members without extracting untrusted contents."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return sorted(archive.namelist())
    with tarfile.open(path, "r:gz") as archive:
        return sorted(member.name for member in archive.getmembers())


def _check_members(path: Path, members: Iterable[str], name: str) -> None:
    """Reject repository, development, cache, or credential files."""
    member_list = list(members)
    has_package = any(
        member.endswith(".py")
        and (
            member.startswith(REQUIRED_PACKAGE_PREFIX)
            or "/src/" + REQUIRED_PACKAGE_PREFIX in member
        )
        for member in member_list
    )
    if not has_package:
        raise SystemExit(f"{path.name}: missing packaged conducto/ files")
    forbidden = [
        member
        for member in member_list
        if any(part.lower() in FORBIDDEN_PARTS for part in Path(member).parts)
        or any(token in member.lower() for token in ("password", "secret", "credential"))
    ]
    if forbidden:
        raise SystemExit(f"{path.name}: forbidden members: {', '.join(forbidden)}")
    if name == "wheel" and not any(
        member.endswith(".dist-info/METADATA") for member in member_list
    ):
        raise SystemExit(f"{path.name}: missing wheel METADATA")
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            timestamps: set[object] = {entry.date_time for entry in archive.infolist()}
    else:
        with tarfile.open(path, "r:gz") as archive:
            timestamps = {member.mtime for member in archive.getmembers()}
    if len(timestamps) != 1:
        raise SystemExit(f"{path.name}: archive timestamps are not normalized")


def _digest(path: Path) -> str:
    """Compute the SHA-256 digest used across workflow jobs."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_from_wheel(path: Path) -> str:
    """Read the wheel's versioned metadata without installing it."""
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        return archive.read(metadata_name).decode("utf-8")


def _write_outputs(wheel: Path, sdist: Path, output_dir: Path) -> None:
    """Write checksums, inventory, and a minimal SPDX SBOM for exact artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {"file": path.name, "sha256": _digest(path), "bytes": path.stat().st_size}
        for path in (wheel, sdist)
    ]
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{entry['sha256']}  {entry['file']}\n" for entry in entries),
        encoding="utf-8",
    )
    inventory = {
        "package": _project_metadata()[0],
        "version": _project_metadata()[1],
        "artifacts": entries,
        "wheel_members": _artifact_members(wheel),
        "sdist_members": _artifact_members(sdist),
    }
    (output_dir / "package-inventory.json").write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    name, version, dependencies, extras = _project_metadata()
    packages = [
        {
            "SPDXID": "SPDXRef-Package",
            "name": name,
            "versionInfo": version,
            "downloadLocation": "NOASSERTION",
            "licenseConcluded": "Apache-2.0",
            "licenseDeclared": "Apache-2.0",
            "checksums": [
                {"algorithm": "SHA256", "checksumValue": entry["sha256"]} for entry in entries
            ],
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": f"pkg:pypi/{name}@{version}",
                }
            ],
        }
    ]
    relationships = []
    for index, dependency in enumerate(
        [*dependencies, *[item for values in extras.values() for item in values]]
    ):
        dependency_name = re.split(r"[<=>!~\[]", dependency, maxsplit=1)[0].strip()
        dependency_id = f"SPDXRef-Dependency-{index}"
        packages.append(
            {
                "SPDXID": dependency_id,
                "name": dependency_name,
                "versionInfo": "NOASSERTION",
                "downloadLocation": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
            }
        )
        relationships.append(
            {
                "spdxElementId": "SPDXRef-Package",
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": dependency_id,
            }
        )
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{name}-{version}",
        "documentNamespace": f"https://conducto.ai/sbom/{name}/{version}",
        "creationInfo": {
            "created": "1970-01-01T00:00:00Z",
            "creators": ["Tool: release_verify.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }
    (output_dir / "sbom.spdx.json").write_text(
        json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    """Validate a single wheel/sdist pair and generate release metadata."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    name, version, dependencies, extras = _project_metadata()
    expected_stem = name.replace("-", "_") + f"-{version}"
    if args.wheel.stem != f"{expected_stem}-py3-none-any":
        raise SystemExit(f"wheel name/version does not match pyproject.toml: {args.wheel.name}")
    if args.sdist.name != f"{expected_stem}.tar.gz":
        raise SystemExit(f"sdist name/version does not match pyproject.toml: {args.sdist.name}")
    wheel_members = _artifact_members(args.wheel)
    sdist_members = _artifact_members(args.sdist)
    _check_members(args.wheel, wheel_members, "wheel")
    _check_members(args.sdist, sdist_members, "sdist")
    if not any(member.endswith("/LICENSE") or member == "LICENSE" for member in sdist_members):
        raise SystemExit("sdist is missing LICENSE")
    if not any(member.endswith("/README.md") or member == "README.md" for member in sdist_members):
        raise SystemExit("sdist is missing README.md")
    metadata = _metadata_from_wheel(args.wheel)
    if f"Name: {name}\n" not in metadata or f"Version: {version}\n" not in metadata:
        raise SystemExit("wheel metadata does not match pyproject.toml")
    if not dependencies or not extras:
        raise SystemExit("project metadata must declare base dependencies and optional extras")
    _write_outputs(args.wheel, args.sdist, args.output_dir)
    print(f"verified {args.wheel.name} and {args.sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
