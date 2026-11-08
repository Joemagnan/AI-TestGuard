"""Résumé Markdown des résultats de tests (JUnit) et de la couverture, pour la CI.

python scripts/ci_summary.py reports/tests/junit.xml [coverage.xml]
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def junit_summary(path: Path) -> list[str]:
    if not path.is_file():
        return [f"Rapport JUnit absent : `{path}`"]
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    total = sum(int(s.get("tests", 0)) for s in suites)
    failures = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    duration = sum(float(s.get("time", 0)) for s in suites)
    status = "réussite" if failures + errors == 0 else "ÉCHEC"
    lines = [
        f"### Tests : {status}",
        "",
        "| Exécutés | Réussis | Échecs | Erreurs | Ignorés | Durée (s) |",
        "|---|---|---|---|---|---|",
        f"| {total} | {total - failures - errors - skipped} | {failures} | {errors} | {skipped} "
        f"| {duration:.1f} |",
    ]
    failed = [
        f"- `{case.get('classname')}::{case.get('name')}`"
        for case in root.iter("testcase")
        if case.find("failure") is not None or case.find("error") is not None
    ]
    if failed:
        lines += ["", "Tests en échec :", *failed]
    return lines


def coverage_summary(path: Path) -> list[str]:
    if not path.is_file():
        return []
    rate = float(ET.parse(path).getroot().get("line-rate", 0))
    return ["", f"### Couverture du code (`app/`) : {rate:.1%}"]


def main(argv: list[str]) -> int:
    lines = junit_summary(Path(argv[1]))
    if len(argv) > 2:
        lines += coverage_summary(Path(argv[2]))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
