#!/bin/bash
set -e

# If the first argument is "test", run the test suite
if [ "$1" = "test" ]; then
    shift
    exec python tests/unit_tests.py "$@"
# If the first argument is "lint", run pylint
elif [ "$1" = "lint" ]; then
    shift
    exec pylint --rcfile=.pylintrc modules pipeline.py tests "$@"
# Otherwise execute the given command
elif [ "${1:0:1}" = "-" ]; then
    exec python pipeline.py "$@"
else
    exec "$@"
fi