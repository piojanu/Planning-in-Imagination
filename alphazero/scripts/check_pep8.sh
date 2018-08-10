#/bin/bash
for file in `find alphazero -name "*\.py" | grep -v "alphazero/src/third_party/humblerl"`; do [ -f $file ] && autopep8 -i $file ; done && git status && exit `git diff | wc -l` || exit 1
