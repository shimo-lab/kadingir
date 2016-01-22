#!/bin/sh


logfile="log_render_all_rmarkdown.txt"

for rmdfile in run_*.Rmd
do
    rcommand="rmarkdown::render('"$rmdfile"')"
    echo $rcommand
    time R -e $rcommand >> $logfile
done

for rmdfile in experiment_*.Rmd
do
    rcommand="rmarkdown::render('"$rmdfile"')"
    echo $rcommand
    time R -e $rcommand >> $logfile
done
