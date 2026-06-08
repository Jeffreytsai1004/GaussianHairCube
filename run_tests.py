#!/usr/bin/env python3
"""
Convenience runner for the GaussianHairCube test suite.

Usage:
    python run_tests.py           # run all tests
    python run_tests.py -v        # verbose
    python run_tests.py test_project_io   # run one module
"""

import sys
import unittest


def main():
    args = sys.argv[1:]
    verbosity = 2 if "-v" in args else 1
    args = [a for a in args if a != "-v"]

    loader = unittest.TestLoader()
    if args:
        suite = loader.loadTestsFromNames([f"tests.{name}" for name in args])
    else:
        suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
