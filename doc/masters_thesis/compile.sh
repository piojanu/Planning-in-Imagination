#!/bin/bash

echo -e "\033[0;32m\nCompile bibliography...\033[0m"
xelatex master.tex &> /dev/null
bibtex master.aux

echo -e "\033[0;32m\nCompile document...\033[0m"
xelatex master.tex &> /dev/null
xelatex master.tex

/bin/rm master.aux
/bin/rm master.bbl
/bin/rm master.blg
/bin/rm master.idx
/bin/rm master.lof
/bin/rm master.log
/bin/rm master.lot
/bin/rm master.out
/bin/rm master.toc
/bin/rm master-blx.bib
/bin/rm master.run.xml
