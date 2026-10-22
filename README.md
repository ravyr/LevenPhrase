# LevenPhrase

**LevenPhrase (LP)** is a novel phrase-distance metric designed to address limitations of existing similarity measures when applied to spoken-language transcriptions.

Unlike character-level approaches such as **Levenshtein distance** or metrics such as **Word Error Rate (WER)**, LevenPhrase operates on word-level structures and incorporates language-specific character-frequency distributions into a bi-space correlation model.

The method was originally motivated by discrepancies observed in automatic speech recognition (**ASR**) systems used in human–robot interaction. However, LevenPhrase is not limited to ASR evaluation.

LP provides finer-grained sensitivity to linguistic plausibility, particularly for short phrases, where small lexical deviations can have a disproportionately large semantic impact.

## Package

The repository contains a standalone Python implementation in:

```text
levenphrase_plain/
```

This version does **not** require `pip install`, `pyproject.toml`, or shell entry points. It can be run directly with Python.

For implementation details, metric definitions, result fields, debug output, and MATLAB compatibility, see:

```text
levenphrase_plain/readme.txt
```

## Quick start

Enter the package directory:

```bash
cd levenphrase_plain
```

Run the test suite:

```bash
python3 test_levenphrase.py
```

Expected output:

```text
Running LevenPhrase tests...
OK: all tests passed.
```

## Command-line usage

### Recommended example

Process a file containing triples:

```bash
python3 preview_file.py /path/to/teksty.txt \
    --format triples \
    --language pl \
    --mode glosek \
    --results-file results2.txt \
    --results-kind 2
```

Replace `/path/to/teksty.txt` with the path to your input file.

### TSV example

```bash
python3 preview_file.py examples/pairs.tsv \
    --format tsv \
    --language pl \
    --mode wyrazow \
    --results-file results1.txt
```

For TSV input, each row has the form:

```text
reference sentence<TAB>hypothesis sentence
```

Supported input formats include:

```text
tsv       reference<TAB>hypothesis
pipe      reference || hypothesis
pairs     reference line, hypothesis line, repeated
triples   id, reference, hypothesis, repeated
auto      automatic format detection
```

## Using LevenPhrase from Python

LevenPhrase can also be used directly from your own Python code:

```python
from levenphrase import LevenPhrase, LevenPhraseConfig

engine = LevenPhrase(
    LevenPhraseConfig(
        language="pl",
        profile_scale_n=29,
    )
)

result = engine.compare(
    "text A",
    "text B",
    mode="wyrazow",
)

print(result.gamma_scores)
print(result.relative_scores)
```

The main result fields are:

```text
result.gamma_scores
    Raw LevenPhrase metric values.

result.relative_scores
    Relative LevenPhrase values, normalized by the selected profile scale n.

result.relative_quality_scores
    Quality representation derived from the relative scores.
```

More detailed definitions are available in `levenphrase_plain/readme.txt`.

## Optional dependency

If the Python package `jellyfish` is installed, `preview_file.py` uses it for additional comparison metrics and phonetic encodings.

Without `jellyfish`, LevenPhrase still works using the built-in fallback metrics, while unavailable phonetic encodings are marked accordingly.

## Results export

`preview_file.py` can write experiment/debug results to a text file:

```bash
python3 preview_file.py examples/pairs.tsv \
    --format tsv \
    --language pl \
    --mode wyrazow \
    --results-file results1.txt
```

Useful options include:

```text
--results-kind N
--append-results
--no-results-header
```

The exported format is designed to remain compatible with the historical MATLAB/debug workflow used during development.

For the exact column definitions and a MATLAB parser example, see:

```text
levenphrase_plain/readme.txt
levenphrase_plain/matlab_read_results_example.m
```

## Publication

**Status:** In review.

## Citation

Citation information will be provided after publication.

**Cite:** In review.