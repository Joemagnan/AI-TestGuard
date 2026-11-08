"""PARTIE VIII - Sécurité du dépôt : pas de secret, pas de données complètes versionnées,
dépendances figées."""

import re
import subprocess

import pytest

from app.config import ROOT

# Motifs construits par concaténation pour que ce fichier ne se détecte pas lui-même.
SECRET_PATTERNS = {
    "clé AWS": r"AKIA" + r"[0-9A-Z]{16}",
    "jeton GitHub": r"gh[pousr]_" + r"[A-Za-z0-9]{36}",
    "jeton GitHub fin": r"github_pat_" + r"[A-Za-z0-9_]{50,}",
    "clé privée": r"-----BEGIN [A-Z ]*PRIVATE" + r" KEY-----",
    "jeton Slack": r"xox[baprs]-" + r"[A-Za-z0-9-]{10,}",
    "clé Kaggle": r"\"key\"\s*:\s*\"[0-9a-f]{32}\"",
    "mot de passe en clair": r"(?i)(password|passwd|mot_de_passe)\s*[=:]\s*['\"][^'\"]{4,}['\"]",
    # 16 chiffres commençant par 4, non précédés d'un point ou d'un chiffre : exclut
    # les décimales des flottants (par exemple 0.4216119381140392)
    "numéro de carte": r"(?<![\d.])4[0-9]{3}([ -]?[0-9]{4}){3}(?![\d.])",
}
TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".toml", ".txt", ".json", ".cfg", ".ini", ""}


def tracked_files():
    try:
        out = subprocess.run(
            ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True, timeout=30
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pytest.skip("dépôt Git indisponible")
    return [ROOT / name for name in out if (ROOT / name).is_file()]


def test_card_pattern_ignores_float_decimals():
    pattern = SECRET_PATTERNS["numéro de carte"]
    assert re.search(pattern, "carte 4970 1000 0000 0000")
    assert not re.search(pattern, '"threshold": 0.4216119381140392')


def test_no_secret_in_tracked_files():
    findings = []
    for path in tracked_files():
        if path.suffix not in TEXT_SUFFIXES or path.name == "test_security.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in SECRET_PATTERNS.items():
            if label == "numéro de carte" and path.name == "test_api.py":
                continue  # numéro fictif utilisé pour vérifier le rejet d'un champ
            if re.search(pattern, text):
                findings.append(f"{path.relative_to(ROOT)} : {label}")
    assert not findings, findings


def test_full_dataset_and_local_files_are_not_tracked():
    names = {p.relative_to(ROOT).as_posix() for p in tracked_files()}
    assert "data/creditcard.csv" not in names
    assert not any(n.startswith(".venv/") or n.endswith(".env") for n in names)
    csv_files = [n for n in names if n.endswith(".csv")]
    assert csv_files == ["data/sample/creditcard_sample.csv"]


def test_gitignore_protects_data_and_secrets():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for rule in ("data/*.csv", ".env", ".venv/", "*.pem"):
        assert rule in rules


def test_versioned_sample_stays_small():
    size = (ROOT / "data" / "sample" / "creditcard_sample.csv").stat().st_size
    assert size < 5 * 1024 * 1024


def test_dependencies_are_pinned():
    lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines), lines
