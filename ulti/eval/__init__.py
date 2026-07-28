"""
Reproducible evaluation suites.

A *suite* is a versioned, git-tracked list of seeds (and possibly expected
metrics) that any agent can be benchmarked against. Running the same suite
against the same agent twice must produce identical numbers.

Layout:
    suites/         — versioned suite definitions (one file per suite)
    runners.py      — "run suite X against agent Y, emit report"
"""
