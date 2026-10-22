"""Metric LevenPhrase implementation.

Only the repaired metric family is implemented:

    lambda_tilde_gamma(x, y) = a_gamma d_L(x, y) + b_gamma d_Delta(p_x, p_y)

where d_L is Levenshtein edit distance and d_Delta is the chord distance between
normalized probability profiles on the simplex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .languages import LanguageProfile, get_language_profile

Symbol = str


class Matrix:
    """Small 2D matrix with matrix[row, col] indexing; avoids numpy dependency."""

    def __init__(self, rows: int, columns: int, fill: int = 0) -> None:
        self.data = [[fill for _ in range(columns)] for _ in range(rows)]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return self.data[row][column]
        return self.data[key]

    def __setitem__(self, key, value) -> None:
        if isinstance(key, tuple):
            row, column = key
            self.data[row][column] = value
        else:
            self.data[key] = value

    def __repr__(self) -> str:
        return repr(self.data)



class Mode(str, Enum):
    LETTERS = "liter"
    CHARS = "znakow"
    GLOSEK = "glosek"
    WORDS = "wyrazow"
    QUIET = "quiet"


@dataclass(slots=True)
class LevenPhraseConfig:
    language: str | LanguageProfile = "pl"
    placeholder: str = "#"
    gamma_values: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    epsilon: float = 0.05
    inner_mode_for_words: Mode = Mode.GLOSEK
    profile_scale_n: int | None = None
    verbose: bool = False

    @property
    def language_profile(self) -> LanguageProfile:
        return get_language_profile(self.language)


@dataclass(slots=True)
class AlignmentResult:
    mode: Mode
    edit_distance: int
    inserted: list[Symbol]
    deleted: list[Symbol]
    modified: list[str]
    unchanged: list[Symbol]
    full_text: list[str]
    binary_mask: list[int]
    detailed_binary_mask: list[str]
    phrase1_extended: list[str]
    phrase2_extended: list[str]
    profile_distance: float
    metric_weights: list[tuple[float, float]]
    gamma_scores: list[float]
    upper_bounds: list[float]
    relative_scores: list[float]
    relative_quality_scores: list[float]
    normalized_scores: list[float]
    quality_scores: list[float]
    profile_scale_n: int
    profile_sequence_lengths: tuple[int, int]
    profile_scale_is_local: bool
    measure_10: float = 0.0

    @property
    def distance(self) -> int:
        return self.edit_distance

    @property
    def binary_code(self) -> str:
        return "".join(str(value) for value in self.binary_mask)

    @property
    def detailed_binary_code(self) -> str:
        return "".join(self.detailed_binary_mask)

    @property
    def insertion_deletion_count(self) -> int:
        return len(self.inserted) + len(self.deleted)

    def as_tuple(self) -> tuple[int, str, list[str], list[str], list[float], list[tuple[float, float]]]:
        return (
            self.insertion_deletion_count,
            self.binary_code,
            self.phrase1_extended,
            self.phrase2_extended,
            self.gamma_scores,
            self.metric_weights,
        )


@dataclass
class _BacktrackState:
    inserted: list[Symbol] = field(default_factory=list)
    deleted: list[Symbol] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[Symbol] = field(default_factory=list)
    full_text: list[str] = field(default_factory=list)
    binary_mask: list[int] = field(default_factory=list)
    detailed_binary_mask: list[str] = field(default_factory=list)
    phrase1_extended: list[str] = field(default_factory=list)
    phrase2_extended: list[str] = field(default_factory=list)

    def reverse(self) -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, list(reversed(getattr(self, name))))


def validate_metric_parameters(gamma: float, epsilon: float, profile_scale_n: int) -> None:
    if not -1.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [-1, 1], got {gamma!r}")
    if not 0.0 < epsilon <= 1.0:
        raise ValueError(f"epsilon must be in (0, 1], got {epsilon!r}")
    if profile_scale_n <= 0:
        raise ValueError(f"profile_scale_n must be positive, got {profile_scale_n!r}")


def metric_weights(gamma: float, epsilon: float, profile_scale_n: int) -> tuple[float, float]:
    """Return (a_gamma, b_gamma) from the current manuscript.

    b_gamma includes the n/sqrt(2) scaling, so the profile part is on the same
    order as an edit distance on a space of size n.
    """
    validate_metric_parameters(gamma, epsilon, profile_scale_n)
    a_gamma = epsilon + (1.0 - epsilon) * (1.0 - gamma) / 2.0
    b_gamma = profile_scale_n / math.sqrt(2.0) * (1.0 + gamma) / 2.0
    return a_gamma, b_gamma


def metric_upper_bound(gamma: float, epsilon: float, profile_scale_n: int) -> float:
    """Theoretical upper bound of lambda_tilde for the current parameterization.

    This is kept as a diagnostic helper.  The main relative view requested in
    the project is `lambda_tilde / length`, implemented by `relative_levenphrase`.
    """
    a_gamma, b_gamma = metric_weights(gamma, epsilon, profile_scale_n)
    return a_gamma * profile_scale_n + b_gamma * math.sqrt(2.0)


def relative_levenphrase(score: float, length: int) -> float:
    """Return the relative LevenPhrase view: lambda_tilde divided by length.

    This is not a replacement for the metric.  It is a reporting view used to
    compare scores across strings/corpora with different lengths.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length!r}")
    return score / length


def levenshtein_matrix(sequence1: Sequence[Symbol], sequence2: Sequence[Symbol]) -> Matrix:
    rows = len(sequence1) + 1
    columns = len(sequence2) + 1
    matrix = Matrix(rows, columns, fill=0)
    for row in range(rows):
        matrix[row, 0] = row
    for column in range(columns):
        matrix[0, column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = 0 if sequence1[row - 1] == sequence2[column - 1] else 1
            matrix[row, column] = min(
                matrix[row - 1, column] + 1,
                matrix[row - 1, column - 1] + substitution_cost,
                matrix[row, column - 1] + 1,
            )
    return matrix


def levenshtein_distance(sequence1: Sequence[Symbol], sequence2: Sequence[Symbol]) -> int:
    return int(levenshtein_matrix(sequence1, sequence2)[len(sequence1), len(sequence2)])


def probability_profile(symbols: Sequence[Symbol], weights: dict[str, float] | None = None) -> dict[Symbol, float]:
    """Return a simplex profile for a symbol sequence.

    With no weights, this is the empirical symbol distribution. With language
    weights, each occurrence contributes the symbol's configured weight; the
    final vector is still normalized to sum to one.
    """
    if not symbols:
        return {}
    weights = weights or {}
    scores: dict[Symbol, float] = {}
    for symbol in symbols:
        value = float(weights.get(symbol, 1.0))
        if value < 0:
            raise ValueError(f"Negative profile weight for symbol {symbol!r}: {value}")
        scores[symbol] = scores.get(symbol, 0.0) + value
    total = sum(scores.values())
    if total == 0.0:
        return {}
    return {symbol: value / total for symbol, value in scores.items()}


def simplex_chord_distance(profile1: dict[Symbol, float], profile2: dict[Symbol, float]) -> float:
    if not profile1 and not profile2:
        return 0.0
    if not profile1 or not profile2:
        return 1.0
    norm1 = math.sqrt(sum(value * value for value in profile1.values()))
    norm2 = math.sqrt(sum(value * value for value in profile2.values()))
    if norm1 == 0.0 and norm2 == 0.0:
        return 0.0
    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0
    alphabet = set(profile1) | set(profile2)
    squared = 0.0
    for symbol in alphabet:
        squared += (profile1.get(symbol, 0.0) / norm1 - profile2.get(symbol, 0.0) / norm2) ** 2
    return math.sqrt(max(0.0, squared))


def profile_distance(
    sequence1: Sequence[Symbol],
    sequence2: Sequence[Symbol],
    *,
    weights: dict[str, float] | None = None,
) -> float:
    return simplex_chord_distance(
        probability_profile(sequence1, weights),
        probability_profile(sequence2, weights),
    )


def metric_levenphrase_distance(
    sequence1: Sequence[Symbol],
    sequence2: Sequence[Symbol],
    *,
    gamma: float = 0.0,
    epsilon: float = 0.05,
    profile_scale_n: int | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    scale_n = profile_scale_n or max(len(sequence1), len(sequence2), 1)
    a_gamma, b_gamma = metric_weights(gamma, epsilon, scale_n)
    return a_gamma * levenshtein_distance(sequence1, sequence2) + b_gamma * profile_distance(
        sequence1, sequence2, weights=weights
    )


class LevenPhrase:
    def __init__(self, config: LevenPhraseConfig | None = None) -> None:
        self.config = config or LevenPhraseConfig()

    def tokenize(self, phrase: str | Sequence[Symbol], mode: str | Mode) -> list[Symbol]:
        mode = Mode(mode)
        if mode == Mode.QUIET:
            return list(phrase) if isinstance(phrase, str) else list(phrase)
        if isinstance(phrase, str):
            return self.config.language_profile.tokenize(phrase, mode.value)
        return list(phrase)

    def compare(
        self,
        phrase1: str | Sequence[Symbol],
        phrase2: str | Sequence[Symbol],
        mode: str | Mode = Mode.CHARS,
        *,
        emit_report: bool | None = None,
    ) -> AlignmentResult:
        mode = Mode(mode)
        sequence1 = self.tokenize(phrase1, mode)
        sequence2 = self.tokenize(phrase2, mode)
        report = self.config.verbose if emit_report is None else emit_report
        matrix = levenshtein_matrix(sequence1, sequence2)
        state = self._backtrack(sequence1, sequence2, matrix, mode)
        state.reverse()
        edit_distance = int(matrix[len(sequence1), len(sequence2)])
        result = self._build_result(mode, edit_distance, state, sequence1, sequence2)
        if report and mode != Mode.QUIET:
            self.print_report(result, sequence1, sequence2)
        return result

    def infer_profile_scale_n(self, phrases: Sequence[str | Sequence[Symbol]], mode: str | Mode) -> int:
        """Infer a fixed n for a batch/corpus.

        Use this before batch evaluation if no external domain-wide n has been
        chosen. The inferred n is the maximum profile sequence length.
        """
        mode = Mode(mode)
        maximum = 1
        for phrase in phrases:
            sequence = self.tokenize(phrase, mode)
            maximum = max(maximum, len(self._profile_sequence(sequence, mode)))
        return maximum

    def _build_result(
        self,
        mode: Mode,
        edit_distance: int,
        state: _BacktrackState,
        sequence1: Sequence[Symbol],
        sequence2: Sequence[Symbol],
    ) -> AlignmentResult:
        if mode == Mode.QUIET:
            return AlignmentResult(
                mode=mode,
                edit_distance=edit_distance,
                inserted=state.inserted,
                deleted=state.deleted,
                modified=state.modified,
                unchanged=state.unchanged,
                full_text=state.full_text,
                binary_mask=state.binary_mask,
                detailed_binary_mask=state.detailed_binary_mask,
                phrase1_extended=state.phrase1_extended,
                phrase2_extended=state.phrase2_extended,
                profile_distance=0.0,
                metric_weights=[],
                gamma_scores=[],
                upper_bounds=[],
                relative_scores=[],
                relative_quality_scores=[],
                normalized_scores=[],
                quality_scores=[],
                profile_scale_n=1,
                profile_sequence_lengths=(0, 0),
                profile_scale_is_local=True,
            )

        prof_seq1 = self._profile_sequence(sequence1, mode)
        prof_seq2 = self._profile_sequence(sequence2, mode)
        local_scale = max(len(prof_seq1), len(prof_seq2), 1)
        scale_n = self.config.profile_scale_n or local_scale
        profile_scale_is_local = self.config.profile_scale_n is None
        prof_distance = profile_distance(
            prof_seq1,
            prof_seq2,
            weights=dict(self.config.language_profile.symbol_weights),
        )

        weights: list[tuple[float, float]] = []
        scores: list[float] = []
        upper_bounds: list[float] = []
        relative_scores: list[float] = []
        relative_quality_scores: list[float] = []
        normalized_scores: list[float] = []
        quality_scores: list[float] = []
        for gamma in self.config.gamma_values:
            a_gamma, b_gamma = metric_weights(gamma, self.config.epsilon, scale_n)
            upper = metric_upper_bound(gamma, self.config.epsilon, scale_n)
            score = a_gamma * edit_distance + b_gamma * prof_distance
            # Project meeting decision: keep lambda_tilde unchanged and add a
            # relative view obtained by dividing by the chosen length n.
            relative = relative_levenphrase(score, scale_n)

            # `normalized_scores` / `quality_scores` are retained as aliases for
            # older scripts, but they now use the meeting-approved relative view
            # rather than division by the theoretical upper bound.
            clipped_relative = min(1.0, max(0.0, relative))
            weights.append((a_gamma, b_gamma))
            scores.append(score)
            upper_bounds.append(upper)
            relative_scores.append(relative)
            relative_quality_scores.append(1.0 - clipped_relative)
            normalized_scores.append(relative)
            quality_scores.append(1.0 - clipped_relative)

        detailed_binary = "".join(state.detailed_binary_mask)
        compared_units = max(len(prof_seq1), len(prof_seq2), 1)
        measure_10 = detailed_binary.count("1") / compared_units

        return AlignmentResult(
            mode=mode,
            edit_distance=edit_distance,
            inserted=state.inserted,
            deleted=state.deleted,
            modified=state.modified,
            unchanged=state.unchanged,
            full_text=state.full_text,
            binary_mask=state.binary_mask,
            detailed_binary_mask=state.detailed_binary_mask,
            phrase1_extended=state.phrase1_extended,
            phrase2_extended=state.phrase2_extended,
            profile_distance=prof_distance,
            metric_weights=weights,
            gamma_scores=scores,
            upper_bounds=upper_bounds,
            relative_scores=relative_scores,
            relative_quality_scores=relative_quality_scores,
            normalized_scores=normalized_scores,
            quality_scores=quality_scores,
            profile_scale_n=scale_n,
            profile_sequence_lengths=(len(prof_seq1), len(prof_seq2)),
            profile_scale_is_local=profile_scale_is_local,
            measure_10=measure_10,
        )

    def _profile_sequence(self, sequence: Sequence[Symbol], mode: Mode) -> list[Symbol]:
        if mode == Mode.WORDS:
            text = " ".join(sequence)
            return self.config.language_profile.tokenize(text, self.config.inner_mode_for_words.value)
        return list(sequence)

    def _backtrack(self, sequence1: Sequence[Symbol], sequence2: Sequence[Symbol], matrix: Matrix, mode: Mode) -> _BacktrackState:
        row = len(sequence1)
        column = len(sequence2)
        state = _BacktrackState()
        while row > 0 or column > 0:
            step = self._choose_step(matrix, sequence1, sequence2, row, column, mode)
            if step in {"diag_mod", "diag_same"}:
                token1 = sequence1[row - 1]
                token2 = sequence2[column - 1]
                if step == "diag_mod":
                    self._append_modified(state, token1, token2, mode)
                else:
                    self._append_unchanged(state, token1, token2, mode)
                row -= 1
                column -= 1
            elif step == "left":
                self._append_inserted(state, sequence2[column - 1], mode)
                column -= 1
            elif step == "up":
                self._append_deleted(state, sequence1[row - 1], mode)
                row -= 1
            else:
                raise RuntimeError(f"Unknown backtracking step: {step}")
        return state

    def _choose_step(self, matrix: Matrix, sequence1: Sequence[Symbol], sequence2: Sequence[Symbol], row: int, column: int, mode: Mode) -> str:
        current = matrix[row, column]
        candidates: list[tuple[str, int]] = []
        if row > 0 and column > 0:
            diagonal = matrix[row - 1, column - 1]
            candidates.append(("diag_same" if diagonal == current else "diag_mod", diagonal))
        if column > 0:
            candidates.append(("left", matrix[row, column - 1]))
        if row > 0:
            candidates.append(("up", matrix[row - 1, column]))
        best_cost = min(cost for _, cost in candidates)
        best_steps = [step for step, cost in candidates if cost == best_cost]
        if mode == Mode.WORDS and len(best_steps) > 1:
            return self._choose_word_tie_break(best_steps, sequence1, sequence2, row, column)
        priority = {"diag_same": 0, "diag_mod": 1, "left": 2, "up": 3}
        return min(best_steps, key=lambda step: priority[step])

    def _choose_word_tie_break(self, steps: list[str], sequence1: Sequence[Symbol], sequence2: Sequence[Symbol], row: int, column: int) -> str:
        scores: dict[str, int] = {}
        if any(step in steps for step in ("diag_same", "diag_mod")) and row > 0 and column > 0:
            step = "diag_same" if "diag_same" in steps else "diag_mod"
            scores[step] = levenshtein_distance(self._inner_tokens(sequence1[row - 1]), self._inner_tokens(sequence2[column - 1]))
        if "left" in steps and row > 0 and column > 1:
            scores["left"] = levenshtein_distance(self._inner_tokens(sequence1[row - 1]), self._inner_tokens(sequence2[column - 2]))
        if "up" in steps and row > 1 and column > 0:
            scores["up"] = levenshtein_distance(self._inner_tokens(sequence1[row - 2]), self._inner_tokens(sequence2[column - 1]))
        return min(scores, key=scores.get) if scores else steps[0]

    def _inner_tokens(self, token: Symbol) -> list[Symbol]:
        mode = self.config.inner_mode_for_words
        if mode == Mode.GLOSEK:
            return self.config.language_profile.tokenize_glosek(token, keep_spaces=False)
        if mode == Mode.LETTERS:
            return self.config.language_profile.tokenize_chars(token, keep_spaces=False)
        return list(token)

    def _append_modified(self, state: _BacktrackState, token1: Symbol, token2: Symbol, mode: Mode) -> None:
        state.modified.append(f"{token1}>{token2}")
        state.full_text.append(f"{token1}>{token2}")
        state.binary_mask.append(0)
        if mode == Mode.QUIET:
            state.phrase1_extended.extend([self.config.placeholder, token1])
            state.phrase2_extended.extend([token2, self.config.placeholder])
            return
        inner = self.compare(self._inner_tokens(token1), self._inner_tokens(token2), Mode.QUIET, emit_report=False)
        state.detailed_binary_mask.append(inner.binary_code)
        state.phrase1_extended.append("".join(inner.phrase1_extended))
        state.phrase2_extended.append("".join(inner.phrase2_extended))

    def _append_unchanged(self, state: _BacktrackState, token1: Symbol, token2: Symbol, mode: Mode) -> None:
        state.unchanged.append(token1)
        state.full_text.append(token1)
        state.binary_mask.append(1)
        if mode == Mode.QUIET:
            state.phrase1_extended.append(token1)
            state.phrase2_extended.append(token2)
            return
        inner = self.compare(self._inner_tokens(token1), self._inner_tokens(token2), Mode.QUIET, emit_report=False)
        state.detailed_binary_mask.append(inner.binary_code)
        state.phrase1_extended.append(token1)
        state.phrase2_extended.append(token2)

    def _append_inserted(self, state: _BacktrackState, token: Symbol, mode: Mode) -> None:
        state.inserted.append(token)
        state.full_text.append(f"+{token}")
        state.binary_mask.append(0)
        if mode == Mode.QUIET:
            state.phrase1_extended.append(self.config.placeholder)
            state.phrase2_extended.append(token)
            return
        inner_len = len(self._inner_tokens(token))
        state.detailed_binary_mask.append("0" * inner_len)
        state.phrase1_extended.append(self.config.placeholder * inner_len)
        state.phrase2_extended.append(token)

    def _append_deleted(self, state: _BacktrackState, token: Symbol, mode: Mode) -> None:
        state.deleted.append(token)
        state.full_text.append(f"-{token}")
        state.binary_mask.append(0)
        if mode == Mode.QUIET:
            state.phrase1_extended.append(token)
            state.phrase2_extended.append(self.config.placeholder)
            return
        inner_len = len(self._inner_tokens(token))
        state.detailed_binary_mask.append("0" * inner_len)
        state.phrase1_extended.append(token)
        state.phrase2_extended.append(self.config.placeholder * inner_len)

    def print_report(self, result: AlignmentResult, sequence1: Sequence[Symbol], sequence2: Sequence[Symbol]) -> None:
        print()
        print(f"string 1 ma {len(sequence1)} {result.mode.value}, a drugi {len(sequence2)}")
        print(f"metryczny LevenPhrase edit_distance = {result.edit_distance}")
        print("inserted\t:", len(result.inserted), ":", result.inserted)
        print("deleted \t:", len(result.deleted), ":", result.deleted)
        print("modified\t:", len(result.modified), ":", result.modified)
        print("ok      \t:", len(result.unchanged), ":", result.unchanged)
        print("alignment\t:", result.full_text)
        print("binary mask\t:", result.binary_mask)
        print("profile_distance d_delta =", round(result.profile_distance, 6))
        for gamma, (a_gamma, b_gamma), score, normalized, quality in zip(
            self.config.gamma_values,
            result.metric_weights,
            result.gamma_scores,
            result.relative_scores,
            result.relative_quality_scores,
        ):
            print(
                f"lambda_tilde@gamma={gamma}: {score:.6f}  "
                f"relative_LevenPhrase={normalized:.6f} relative_quality={100*quality:.2f}% "
                f"(a={a_gamma:.3f}, b={b_gamma:.3f}, n={result.profile_scale_n})"
            )


def levenPhrase(phrase1, phrase2, what="znakow"):
    return LevenPhrase().compare(phrase1, phrase2, what).as_tuple()
