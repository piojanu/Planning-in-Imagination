#/bin/bash
git fetch --depth=1 origin master
for file in `git show --name-only --oneline FETCH_HEAD..HEAD | cut -d " " -f 1 | sort | uniq | grep ".py"` ; do [ -f $file ] && autopep8 -i $file ; done && exit `git diff | wc -l` || exit 1
