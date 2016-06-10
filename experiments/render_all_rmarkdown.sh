#!/bin/sh

for rmdfile in */run_*.Rmd */experiment_*.Rmd
do
    rcommand="rmarkdown::render('"$rmdfile"')"
    echo $rcommand
    time R -e $rcommand >> log_render_all_rmarkdown.txt
    echo "============================================"
done
