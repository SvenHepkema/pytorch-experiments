#!/bin/sh

if ! black . > /dev/null 2>&1
then
  echo
  echo
  echo "===> The Black python file formatter failed ('black .' failed)"
  echo
  echo
  exit 1
fi

if ! ruff . --fix 
then
  echo
  echo
  echo "===> The ruff code linter failed ('ruff . --fix' failed)"
  echo
  echo
  exit 1
fi

echo "No errors were encountered during formatting and linting"
exit 0
