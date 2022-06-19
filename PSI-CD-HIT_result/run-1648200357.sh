#!/bin/sh
#$ -S /bin/bash
#$ -v PATH=/home/data/webcomp/RAMMCAP-ann/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
#$ -v BLASTMAT=/home/data/webcomp/RAMMCAP-ann/blast/bin/data
#$ -v LD_LIBRARY_PATH=/home/data/webcomp/RAMMCAP-ann/gnuplot-install/lib
#$ -v PERL5LIB=/home/hying/programs/Perl_Lib
#$ -q cdhit_webserver.q,fast.q
#$ -pe orte 4
#$ -l h_rt=24:00:00


#$ -e /data5/data/webcomp/web-session/1648200357/1648200357.err
#$ -o /data5/data/webcomp/web-session/1648200357/1648200357.out
cd /data5/data/webcomp/web-session/1648200357
sed -i "s/\x0d/\n/g" 1648200357.fas.0

faa_stat.pl 1648200357.fas.0

/data5/data/NGS-ann-project/apps/cd-hit/psi-cd-hit/psi-cd-hit.pl -i 1648200357.fas.0 -o 1648200357.fas.1 -c 0.25 -P /data5/data/NGS-ann-project/apps/blast+/bin -para 4
rm -rf 1648200357.fas.0-bl
faa_stat.pl 1648200357.fas.1
/data5/data/NGS-ann-project/apps/cd-hit/clstr_sort_by.pl no < 1648200357.fas.1.clstr > 1648200357.fas.1.clstr.sorted
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list.pl 1648200357.fas.1.clstr 1648200357.clstr.dump
gnuplot1.pl < 1648200357.fas.1.clstr > 1648200357.fas.1.clstr.1; gnuplot2.pl 1648200357.fas.1.clstr.1 1648200357.fas.1.clstr.1.png
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1648200357.clstr.dump 1648200357.clstr_no.dump
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1648200357.clstr.dump 1648200357.clstr_len.dump len
/data5/data/NGS-ann-project/apps/cd-hit/clstr_list_sort.pl 1648200357.clstr.dump 1648200357.clstr_des.dump des
tar -zcf 1648200357.result.tar.gz * --exclude=*.dump --exclude=*.env
echo hello > 1648200357.ok
