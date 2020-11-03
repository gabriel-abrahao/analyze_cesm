#!/bin/bash
tempfname="tempfile.ncl"
minfolder="../input/rcp_daily/"

scen="rcp2.6_seg_005"

#st="'""scenario="\"$scen\""'"
#cmd=ncl" "$st" "cesm_estacao_chuvosa_v0.7.ncl
#echo $cmd
#$cmd

for scen in $(ls $minfolder); do
	echo $scen
	st="scenario="\"$scen\"
	echo "begin" >$tempfname
	echo $st >>$tempfname
	echo "end" >>$tempfname
	ncl cesm_estacao_chuvosa_v0.7.ncl
done
	
st="scenario="\"$scen\"
echo "begin" >$tempfname
echo $st >>$tempfname
echo "end" >>$tempfname
