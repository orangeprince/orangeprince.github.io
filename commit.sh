#!/usr/bin/env bash
jekyll build -s . -d ../www/
git add .
git commit
git push https://orangeprince:dcdlab525@github.com/orangeprince/orangeprince.github.io
