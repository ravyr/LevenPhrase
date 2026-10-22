"""Run with: python3 test_levenphrase.py"""

from pathlib import Path

from levenphrase import (
    LevenPhrase,
    LevenPhraseConfig,
    TextPair,
    available_language_profiles,
    load_text_pairs,
    metric_levenphrase_distance,
)
from levenphrase.core import levenshtein_distance
from levenphrase.preview import preview_file, write_results_file


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    print("Running LevenPhrase tests...")

    check({"pl", "cz", "sk", "en"}.issubset(set(available_language_profiles())), "language profiles missing")

    x = list("kitten")
    y = list("sitting")
    check(metric_levenphrase_distance(x, y, gamma=-1.0) == levenshtein_distance(x, y), "gamma=-1 should equal Levenshtein")

    engine = LevenPhrase(LevenPhraseConfig(language="pl", profile_scale_n=29))
    result = engine.compare("here we are now entertain us", "here we are now in containers", mode="wyrazow")
    check(result.edit_distance == 2, "word-level edit distance should be 2")
    check(result.gamma_scores[0] == 2, "gamma=-1 score should be edit distance")
    check(
        all(abs(rel - raw / result.profile_scale_n) < 1e-12 for raw, rel in zip(result.gamma_scores, result.relative_scores)),
        "relative LevenPhrase should be lambda_tilde divided by length n",
    )
    check(result.normalized_scores == result.relative_scores, "normalized_scores should alias the relative view")

    pair_path = Path("examples/pairs.tsv")
    pairs = load_text_pairs(pair_path, format="tsv")
    check(len(pairs) == 3, "example TSV should load 3 pairs")
    check(pairs[0] == TextPair("here we are now entertain us", "here we are now in containers", None, 1), "first TSV pair mismatch")

    rows = preview_file(pair_path, format="tsv", language="pl", mode="wyrazow")
    check(len(rows) == 3, "preview should produce 3 rows")
    check(rows[0].levenphrase_edit_distance == 2, "first preview edit distance should be 2")

    results_path = Path("test_results1.txt")
    if results_path.exists():
        results_path.unlink()
    write_results_file(rows, results_path, kind=1)
    check(results_path.exists(), "results file should be created")
    raw_results = results_path.read_bytes()
    check(b"\r\n" in raw_results, "results file should use Windows CRLF line endings")
    check(b"\n" not in raw_results.replace(b"\r\n", b""), "results file should not contain bare LF line endings")
    content = raw_results.decode("utf-8").splitlines()
    check(content[0].startswith("data godzina : kind"), "results header missing")
    check("lambda_g1 lambda_g2 lambda_g3 lambda_g4 lambda_g5" in content[0], "metric lambda columns missing")
    numeric_rows = [line for line in content[1:] if line]
    check(len(numeric_rows) == 3, "results file should have one row per pair")
    check(numeric_rows[0].count(":") >= 8, "results row should use the colon-separated groups")
    results_path.unlink()

    # Triangle sanity check for the manuscript counterexample shape.
    x, y, z = list("AA"), list("AB"), list("BB")
    d_xz = metric_levenphrase_distance(x, z, gamma=0.0, profile_scale_n=2)
    d_xy = metric_levenphrase_distance(x, y, gamma=0.0, profile_scale_n=2)
    d_yz = metric_levenphrase_distance(y, z, gamma=0.0, profile_scale_n=2)
    check(d_xz <= d_xy + d_yz + 1e-12, "triangle inequality check failed")

    print("OK: all tests passed.")


if __name__ == "__main__":
    main()
