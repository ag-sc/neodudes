#!/bin/sh
mkdir -p docs
cd docs
sphinx-apidoc -f -o source/ ../src/
make html