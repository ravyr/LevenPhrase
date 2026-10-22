"""Run with: python3 run_example.py"""

from levenphrase import LevenPhrase, LevenPhraseConfig

engine = LevenPhrase(LevenPhraseConfig(language="pl", profile_scale_n=29, verbose=True))
engine.compare(
    "here we are now entertain us",
    "here we are now in containers",
    mode="wyrazow",
)
