#!/bin/bash
# This script stages only the modified files affected by pre-commit hooks
FILES=$(git diff --name-only --cached)
git add $FILES
