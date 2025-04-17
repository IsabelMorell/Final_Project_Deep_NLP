pytest .
mypy --cache-dir=/dev/null --check-untyped-defs --ignore-missing-imports .
flake8 --max-line-length 89 --extend-ignore=E203