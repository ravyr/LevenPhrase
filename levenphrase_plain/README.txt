LevenPhrase plain package
=========================

This version does NOT require pip install, pyproject.toml, or shell entry points.
Unzip it, cd into the directory, and run Python scripts directly.

Current decision
----------------

The package keeps the current manuscript math for raw Metric LevenPhrase:

    lambda_tilde_gamma = a_gamma * d_L + b_gamma * d_delta

and adds a reporting view agreed in the meeting:

    relative_LevenPhrase_gamma = lambda_tilde_gamma / n

where n is the chosen length/scale stored as result.profile_scale_n.

Important names in code:

    result.gamma_scores              raw lambda_tilde values
    result.relative_scores           lambda_tilde / n
    result.relative_quality_scores   1 - clipped(relative_scores)

For compatibility with the previous local package iteration:

    result.normalized_scores == result.relative_scores
    result.quality_scores    == result.relative_quality_scores

1. Run tests
------------

    cd levenphrase_plain
    python3 test_levenphrase.py

Expected result:

    Running LevenPhrase tests...
    OK: all tests passed.

2. Run one example
------------------

    python3 run_example.py

3. Preview a TSV file
---------------------

    python3 preview_file.py examples/pairs.tsv --format tsv --language pl --mode wyrazow

Input TSV format:

    reference sentence<TAB>hypothesis sentence

Other supported formats:

    --format pipe      reference || hypothesis
    --format pairs     reference line, hypothesis line, repeated
    --format triples   old format: id, reference, hypothesis, repeated
    --format auto      detect automatically

4. Use from your own script
---------------------------

    from levenphrase import LevenPhrase, LevenPhraseConfig

    engine = LevenPhrase(LevenPhraseConfig(language="pl", profile_scale_n=29))
    result = engine.compare("text A", "text B", mode="wyrazow")
    print(result.gamma_scores)       # raw lambda_tilde
    print(result.relative_scores)    # lambda_tilde / n

5. About jellyfish
------------------

If the Python package 'jellyfish' is installed, preview_file.py uses it.
If it is not installed, the preview still works with built-in fallback metrics.
Phonetic encodings are then marked as unavailable.


6. Write MATLAB/debug results1.txt
------------------------------------

The old project had a results1.txt file produced by the historical drukuj()
function. This package keeps the same number of values and colon-separated
groups, but the column names/content now describe the current metric algorithm:

    data godzina : kind : seq : words,elements : profile_d_delta :
    relative_g3 quality_g3 : Lev. D-Lev. Ham. Jar. J-Win. M.R.A. :
    lambda_g1 lambda_g2 lambda_g3 lambda_g4 lambda_g5 :
    profile_scale_n ref_profile_len hyp_profile_len dL_word_or_symbol
    a_g3 b_g3 rho_profile :

Run:

    python3 preview_file.py examples/pairs.tsv --format tsv --language pl --mode wyrazow --results-file results1.txt

Useful options:

    --results-kind 1       first numeric group; use 1/2/etc. as experiment labels
    --append-results       append rows instead of overwriting
    --no-results-header    write rows only

Current debug meaning of the non-jellyfish groups:

    profile_d_delta  = profile metric d_delta
    relative_g3      = lambda_tilde(gamma=0) / n
    quality_g3       = 1 - clipped(relative_g3)
    lambda_g1..g5    = raw metric lambda_tilde values for the five gammas
    profile_scale_n  = n used in b_gamma and relative_LevenPhrase
    ref_profile_len  = profile-token length of the reference
    hyp_profile_len  = profile-token length of the hypothesis
    dL_word_or_symbol = edit-distance component used by lambda_tilde
    a_g3             = a_gamma at gamma=0
    b_g3             = b_gamma at gamma=0
    rho_profile      = 1 - profile_d_delta^2 / 2

See matlab_read_results_example.m for a simple parser example.

Results export note:
  results1.txt is always written with Windows CRLF line endings (\r\n).


Update: the last debug column is now rho_profile instead of profile_part_g3.
The number of columns did not change. profile_part_g3 was redundant because it
can be reconstructed as b_g3 * profile_d_delta. rho_profile is the cosine
similarity between the two probability profiles and is linked to d_delta by:

    d_delta = sqrt(2 * (1 - rho_profile))
    rho_profile = 1 - d_delta^2 / 2
