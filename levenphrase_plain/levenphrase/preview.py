"""Jellyfish-style preview and batch evaluation helpers.

If the real `jellyfish` package is installed, this module uses it. If not, it
falls back to small pure-Python implementations for the numeric preview metrics
and marks phonetic encodings as unavailable. The project dependency still lists
jellyfish, so normal installation should use the real library.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - depends on local environment
    import jellyfish as _jellyfish
except Exception:  # pragma: no cover - exercised where jellyfish is absent
    _jellyfish = None

from .core import LevenPhrase, LevenPhraseConfig, Mode, levenshtein_distance
from .io import TextPair, load_text_pairs


@dataclass(frozen=True, slots=True)
class JellyfishPreview:
    backend: str
    levenshtein: int
    damerau_levenshtein: int
    hamming: int
    jaro: float
    jaro_winkler: float
    match_rating: bool | None
    metaphone_reference: str
    metaphone_hypothesis: str
    soundex_reference: str
    soundex_hypothesis: str
    nysiis_reference: str
    nysiis_hypothesis: str


@dataclass(frozen=True, slots=True)
class PreviewRow:
    label: str | None
    reference: str
    hypothesis: str
    levenphrase_edit_distance: int
    profile_distance: float
    gamma_scores: tuple[float, ...]
    metric_weights: tuple[tuple[float, float], ...]
    relative_scores: tuple[float, ...]
    relative_quality_scores: tuple[float, ...]
    # Backward-compatible aliases for scripts created during the previous
    # iteration.  They now mean the relative LevenPhrase view: lambda_tilde / n.
    normalized_scores: tuple[float, ...]
    quality_scores: tuple[float, ...]
    profile_scale_n: int
    profile_sequence_lengths: tuple[int, int]
    reference_word_count: int
    hypothesis_word_count: int
    reference_length: int
    hypothesis_length: int
    jellyfish: JellyfishPreview

    def as_dict(self) -> dict:
        return asdict(self)


def _hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip_longest(left, right, fillvalue=None))


def _damerau_levenshtein_distance(left: str, right: str) -> int:
    # Optimal string alignment distance; close enough as fallback preview.
    rows = len(left) + 1
    cols = len(right) + 1
    matrix = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        matrix[i][0] = i
    for j in range(cols):
        matrix[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if left[i - 1] == right[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and left[i - 1] == right[j - 2] and left[i - 2] == right[j - 1]:
                matrix[i][j] = min(matrix[i][j], matrix[i - 2][j - 2] + 1)
    return matrix[-1][-1]


def _jaro_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    match_distance = max(len(left), len(right)) // 2 - 1
    match_distance = max(match_distance, 0)
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)
    matches = 0
    for i, char in enumerate(left):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len(right))
        for j in range(start, end):
            if right_matches[j] or char != right[j]:
                continue
            left_matches[i] = True
            right_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i, char in enumerate(left):
        if not left_matches[i]:
            continue
        while not right_matches[k]:
            k += 1
        if char != right[k]:
            transpositions += 1
        k += 1
    transpositions /= 2
    return (
        matches / len(left)
        + matches / len(right)
        + (matches - transpositions) / matches
    ) / 3


def _jaro_winkler_similarity(left: str, right: str) -> float:
    jaro = _jaro_similarity(left, right)
    prefix = 0
    for a, b in zip(left, right):
        if a != b or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * 0.1 * (1 - jaro)


def jellyfish_preview(reference: str, hypothesis: str) -> JellyfishPreview:
    if _jellyfish is not None:  # pragma: no cover - depends on local env
        return JellyfishPreview(
            backend="jellyfish",
            levenshtein=_jellyfish.levenshtein_distance(reference, hypothesis),
            damerau_levenshtein=_jellyfish.damerau_levenshtein_distance(reference, hypothesis),
            hamming=_jellyfish.hamming_distance(reference, hypothesis),
            jaro=round(_jellyfish.jaro_similarity(reference, hypothesis), 6),
            jaro_winkler=round(_jellyfish.jaro_winkler_similarity(reference, hypothesis), 6),
            match_rating=_jellyfish.match_rating_comparison(reference, hypothesis),
            metaphone_reference=_jellyfish.metaphone(reference),
            metaphone_hypothesis=_jellyfish.metaphone(hypothesis),
            soundex_reference=_jellyfish.soundex(reference),
            soundex_hypothesis=_jellyfish.soundex(hypothesis),
            nysiis_reference=_jellyfish.nysiis(reference),
            nysiis_hypothesis=_jellyfish.nysiis(hypothesis),
        )

    return JellyfishPreview(
        backend="fallback",
        levenshtein=levenshtein_distance(list(reference), list(hypothesis)),
        damerau_levenshtein=_damerau_levenshtein_distance(reference, hypothesis),
        hamming=_hamming_distance(reference, hypothesis),
        jaro=round(_jaro_similarity(reference, hypothesis), 6),
        jaro_winkler=round(_jaro_winkler_similarity(reference, hypothesis), 6),
        match_rating=None,
        metaphone_reference="",
        metaphone_hypothesis="",
        soundex_reference="",
        soundex_hypothesis="",
        nysiis_reference="",
        nysiis_hypothesis="",
    )


def preview_pairs(
    pairs: Sequence[TextPair],
    *,
    language: str = "pl",
    mode: str | Mode = Mode.WORDS,
    epsilon: float = 0.05,
    gamma_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    profile_scale_n: int | None = None,
) -> list[PreviewRow]:
    bootstrap = LevenPhrase(LevenPhraseConfig(language=language, gamma_values=gamma_values, epsilon=epsilon))
    if profile_scale_n is None:
        profile_scale_n = bootstrap.infer_profile_scale_n(
            [text for pair in pairs for text in (pair.reference, pair.hypothesis)], mode
        )
    engine = LevenPhrase(
        LevenPhraseConfig(
            language=language,
            gamma_values=gamma_values,
            epsilon=epsilon,
            profile_scale_n=profile_scale_n,
        )
    )
    rows: list[PreviewRow] = []
    for pair in pairs:
        result = engine.compare(pair.reference, pair.hypothesis, mode)
        rows.append(
            PreviewRow(
                label=pair.label,
                reference=pair.reference,
                hypothesis=pair.hypothesis,
                levenphrase_edit_distance=result.edit_distance,
                profile_distance=result.profile_distance,
                gamma_scores=tuple(result.gamma_scores),
                metric_weights=tuple(result.metric_weights),
                relative_scores=tuple(result.relative_scores),
                relative_quality_scores=tuple(result.relative_quality_scores),
                normalized_scores=tuple(result.normalized_scores),
                quality_scores=tuple(result.quality_scores),
                profile_scale_n=result.profile_scale_n,
                profile_sequence_lengths=result.profile_sequence_lengths,
                reference_word_count=len(pair.reference.split()),
                hypothesis_word_count=len(pair.hypothesis.split()),
                reference_length=len(pair.reference),
                hypothesis_length=len(pair.hypothesis),
                jellyfish=jellyfish_preview(pair.reference, pair.hypothesis),
            )
        )
    return rows


def preview_file(
    path: str,
    *,
    format: str = "auto",
    encoding: str = "utf-8",
    language: str = "pl",
    mode: str | Mode = Mode.WORDS,
    epsilon: float = 0.05,
    gamma_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0),
) -> list[PreviewRow]:
    pairs = load_text_pairs(path, format=format, encoding=encoding)  # type: ignore[arg-type]
    return preview_pairs(
        pairs,
        language=language,
        mode=mode,
        epsilon=epsilon,
        gamma_values=gamma_values,
    )


def print_preview_table(rows: Sequence[PreviewRow], *, gamma_index: int = 2) -> None:
    header = (
        "label | LP raw | LP relative | LP rel. quality | Lev | D-Lev | Ham | "
        "Jaro | J-Winkler | MRA | backend | reference -> hypothesis"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        raw = row.gamma_scores[gamma_index]
        normalized = row.relative_scores[gamma_index]
        quality = row.relative_quality_scores[gamma_index]
        jf = row.jellyfish
        print(
            f"{row.label or ''} | {raw:.3f} | {normalized:.3f} | {100*quality:.1f}% | "
            f"{jf.levenshtein} | {jf.damerau_levenshtein} | {jf.hamming} | "
            f"{jf.jaro:.3f} | {jf.jaro_winkler:.3f} | {jf.match_rating} | {jf.backend} | "
            f"{row.reference} -> {row.hypothesis}"
        )





def _gamma_label(gamma: float) -> str:
    """Return a compact label for a gamma value."""
    if gamma == -1.0:
        return "g1"
    if gamma == -0.5:
        return "g2"
    if gamma == 0.0:
        return "g3"
    if gamma == 0.5:
        return "g4"
    if gamma == 1.0:
        return "g5"
    return "g" + str(gamma).replace("-", "m").replace(".", "p")


def results_columns(gamma_values: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0)) -> list[str]:
    """Return the fixed-width MATLAB export columns.

    The number of columns is the same as the historical `drukuj()` export, but
    names and contents now describe the current metric algorithm instead of the
    retired liwlip/RB diagnostics.
    """
    return [
        "data godzina",
        "kind",
        "seq",
        "words",
        "elements",
        "profile_d_delta",
        "relative_g3",
        "quality_g3",
        "Lev.",
        "D-Lev.",
        "Ham.",
        "Jar.",
        "J-Win.",
        "M.R.A.",
        "lambda_g1",
        "lambda_g2",
        "lambda_g3",
        "lambda_g4",
        "lambda_g5",
        "profile_scale_n",
        "ref_profile_len",
        "hyp_profile_len",
        "dL_word_or_symbol",
        "a_g3",
        "b_g3",
        "rho_profile",
    ]


def _numeric_bool(value: bool | None) -> int:
    if value is None:
        return -1
    return 1 if value else 0


def _format_number(value: float | int | bool | None) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, int):
        return str(value)
    # Match the old compact style reasonably closely while avoiding long tails.
    return str(round(float(value), 3))


def _c2(value: str) -> str:
    return value if len(value) >= 2 else value + " "


def _c5(value: str) -> str:
    if "." in value:
        while len(value) < 5:
            value += "0"
    return value


def _s5(value: str) -> str:
    return value if len(value) >= 5 else value + " "


def _match_rating_text(value: bool | None) -> str:
    # The original jellyfish call returned True/False. If jellyfish is absent,
    # keep the old visual shape by writing False rather than None.
    return "True" if value is True else "False"


def _debug_similarity_fields(row: PreviewRow) -> tuple[float, float, float]:
    """Return the three current-debug values in the old liwlip slots.

    The column count is intentionally unchanged, but names/content now describe
    the current metric algorithm:

    * profile_d_delta = profile distance d_Δ,
    * relative_g3     = lambda_tilde at gamma=0 divided by n,
    * quality_g3      = 1 - clipped(relative_g3).
    """
    mid = min(2, len(row.gamma_scores) - 1)
    return (
        row.profile_distance,
        row.relative_scores[mid],
        row.relative_quality_scores[mid],
    )


def _debug_tail_fields(row: PreviewRow) -> list[float | int]:
    """Return seven current-debug values in the old RB_* slots.

    The column count is intentionally unchanged.  These columns now decompose
    the current metric score at gamma=0:

    * profile_scale_n, ref_profile_len, hyp_profile_len,
    * dL_word_or_symbol,
    * a_g3, b_g3, rho_profile = cosine similarity of profiles.

    ``profile_part_g3`` was removed because it is redundant: it equals
    b_g3 * profile_d_delta and can be reconstructed from other columns.
    ``rho_profile`` is more useful for debugging the profile component because
    d_delta = sqrt(2 * (1 - rho_profile)).
    """
    mid = min(2, len(row.gamma_scores) - 1)
    a_g3, b_g3 = row.metric_weights[mid]
    return [
        row.profile_scale_n,
        row.profile_sequence_lengths[0],
        row.profile_sequence_lengths[1],
        row.levenphrase_edit_distance,
        a_g3,
        b_g3,
        1.0 - (row.profile_distance * row.profile_distance) / 2.0,
    ]


def results_row_values(row: PreviewRow, *, seq: int, kind: int = 1) -> list[object]:
    """Return values in the original `drukuj()` order.

    This is intended mostly for tests and debugging. `write_results_file()`
    formats the row in the exact colon-separated historical style.
    """
    jf = row.jellyfish
    liwlipmax, mbll, betterliwlip = _debug_similarity_fields(row)
    seqlen = max(row.reference_length, row.hypothesis_length)
    values: list[object] = [
        kind,
        seq,
        row.reference_word_count,
        seqlen,
        liwlipmax,
        mbll,
        betterliwlip,
        jf.levenshtein,
        jf.damerau_levenshtein,
        jf.hamming,
        jf.jaro,
        jf.jaro_winkler,
        jf.match_rating,
    ]
    values.extend(row.gamma_scores[:5])
    values.extend(_debug_tail_fields(row))
    return values


def _historical_header() -> str:
    return (
        "data godzina : kind : seq : words,elements : profile_d_delta :"
        " relative_g3 quality_g3 : Lev. D-Lev. Ham. Jar. J-Win. M.R.A. :"
        " lambda_g1 lambda_g2 lambda_g3 lambda_g4 lambda_g5 :"
        " profile_scale_n ref_profile_len hyp_profile_len dL_word_or_symbol a_g3 b_g3 rho_profile :\n"
    )


def _historical_row_text(row: PreviewRow, *, seq_zero_based: int, kind: int) -> str:
    jf = row.jellyfish
    liwlipmax, mbll, betterliwlip = _debug_similarity_fields(row)
    seqlen = max(row.reference_length, row.hypothesis_length)
    rb = _debug_tail_fields(row)
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    parts = []
    parts.append(timestamp + " : ")
    parts.append(str(kind) + " : ")
    parts.append(_c2(str(seq_zero_based + 1)) + " : ")
    parts.append(_c2(str(row.reference_word_count)) + " ")
    parts.append(_c2(str(seqlen)) + " : ")
    parts.append(_c5(str(round(liwlipmax, 3))) + " : ")
    parts.append(_c5(str(round(mbll, 3))) + " ")
    parts.append(_c5(str(round(betterliwlip, 3))) + " : ")
    parts.append(_c2(str(jf.levenshtein)) + " ")
    parts.append(_c2(str(jf.damerau_levenshtein)) + " ")
    parts.append(_c2(str(jf.hamming)) + " ")
    parts.append(_c5(str(round(jf.jaro, 3))) + " ")
    parts.append(_c5(str(round(jf.jaro_winkler, 3))) + " ")
    parts.append(_s5(_match_rating_text(jf.match_rating)) + " : ")
    for score in row.gamma_scores[:5]:
        parts.append(str(round(score, 3)) + " ")
    parts.append(": ")
    for value in rb:
        parts.append(str(round(float(value), 3)) + " ")
    parts.append(":\n")
    return "".join(parts)


def write_results_file(
    rows: Sequence[PreviewRow],
    path: str | Path = "results1.txt",
    *,
    gamma_values: Sequence[float] = (-1.0, -0.5, 0.0, 0.5, 1.0),
    kind: int = 1,
    append: bool = False,
    include_header: bool = True,
) -> None:
    """Write a MATLAB-friendly results file.

    The file is always written with Windows CRLF line endings (`\r\n`)
    so Windows-based tools read rows reliably.

    The colon-separated group shape and number of columns match the historical
    `drukuj()` export, but the formerly liwlip/RB slots now contain current
    metric-debug diagnostics.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    write_header = include_header and (not append or not target.exists() or target.stat().st_size == 0)
    with target.open(mode, encoding="utf-8", newline="\r\n") as file:
        if write_header:
            file.write(_historical_header())
        for seq_zero_based, row in enumerate(rows):
            file.write(_historical_row_text(row, seq_zero_based=seq_zero_based, kind=kind))
