#/bin/bash

for file in `git diff-tree --no-commit-id --name-only -r HEAD | grep ".py"` ; do autopep8 -i $file ; done && exit `git diff | wc -l`
