"""Language profiles for Metric LevenPhrase."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

_PUNCTUATION_RE = re.compile(r"[\.,;:!?\"'“”‘’`´…]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    code: str
    name: str
    alphabet: tuple[str, ...]
    glosek_multigraphs: tuple[str, ...] = ()
    replacements: tuple[tuple[str, str], ...] = ()
    symbol_weights: Mapping[str, float] = field(default_factory=dict)

    def normalize(self, text: str) -> str:
        value = text.lower()
        for source, target in self.replacements:
            value = value.replace(source, target)
        value = _PUNCTUATION_RE.sub("", value)
        value = value.replace("-", " ")
        value = _MULTI_SPACE_RE.sub(" ", value).strip()
        return value

    def tokenize_chars(self, text: str, *, keep_spaces: bool = True) -> list[str]:
        value = self.normalize(text)
        if keep_spaces:
            return list(value)
        return [symbol for symbol in value if symbol != " "]

    def tokenize_words(self, text: str) -> list[str]:
        value = self.normalize(text)
        return value.split() if value else []

    def tokenize_glosek(self, text: str, *, keep_spaces: bool = True) -> list[str]:
        value = self.normalize(text)
        multigraphs = sorted(self.glosek_multigraphs, key=len, reverse=True)
        tokens: list[str] = []
        index = 0
        while index < len(value):
            if value[index] == " ":
                if keep_spaces:
                    tokens.append(" ")
                index += 1
                continue
            matched = None
            for graph in multigraphs:
                if value.startswith(graph, index):
                    matched = graph
                    break
            if matched is None:
                tokens.append(value[index])
                index += 1
            else:
                tokens.append(matched)
                index += len(matched)
        return tokens

    def tokenize(self, text: str, mode: str) -> list[str]:
        mode = mode.lower()
        if mode == "wyrazow":
            return self.tokenize_words(text)
        if mode == "glosek":
            return self.tokenize_glosek(text, keep_spaces=True)
        if mode == "liter":
            return self.tokenize_chars(text, keep_spaces=False)
        if mode == "znakow":
            return self.tokenize_chars(text, keep_spaces=True)
        raise ValueError(f"Unsupported tokenization mode: {mode!r}")


_REGISTRY: dict[str, LanguageProfile] = {}


def register_language_profile(profile: LanguageProfile) -> None:
    _REGISTRY[profile.code.lower()] = profile


def get_language_profile(code: str | LanguageProfile = "pl") -> LanguageProfile:
    if isinstance(code, LanguageProfile):
        return code
    key = code.lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown language profile {code!r}. Known profiles: {known}") from exc


def available_language_profiles() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def _letters(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(text))


PL = LanguageProfile(
    code="pl",
    name="Polish",
    alphabet=_letters("aąbcćdeęfghijklłmnńoóprsśtuwyzźż "),
    glosek_multigraphs=("dź", "dż", "dzi", "sz", "cz", "rz", "ch", "dz", "ci"),
    replacements=(("ó", "u"),),
)

CZ = LanguageProfile(
    code="cz",
    name="Czech",
    alphabet=_letters("aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzž "),
    glosek_multigraphs=("dž", "ch", "dz"),
)

SK = LanguageProfile(
    code="sk",
    name="Slovak",
    alphabet=_letters("aáäbcčdďdžeéfghiíjklĺľmnňoóôpqrŕsštťuúvwxyýzž "),
    glosek_multigraphs=("dž", "ch", "dz"),
)

EN = LanguageProfile(
    code="en",
    name="English",
    alphabet=_letters("abcdefghijklmnopqrstuvwxyz "),
    glosek_multigraphs=("sh", "ch", "th", "ph", "ng", "ck", "qu"),
)

for _profile in (PL, CZ, SK, EN):
    register_language_profile(_profile)
