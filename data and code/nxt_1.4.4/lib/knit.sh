#!/bin/bash

# A script to make it easier to work with knit.xsl

infile=$1
outfile=$2
shift
shift

# now the remaining parameters are those to be passed to the main script

if test -z $infile; then
    echo "Missing infile";
    echo "usage: knit.sh infile outfile xalan-parameters";
    exit 1
fi

if test -z $outfile; then
    echo "Missing outfile";
    echo "usage: knit.sh infile outfile xalan-parameters";
    exit 1
fi

   
case $infile in
    /*) # Absolute
	;;	
    *)
	infile=`pwd`/$infile ;;
esac

fileurl="file://${infile}"
# echo "Executing   java org.apache.xalan.xslt.Process -in $fileurl -xsl ${NXT}/lib/knit.xsl -param idatt id  -out $tmpname"

java org.apache.xalan.xslt.Process -in $fileurl -xsl ${NXT}/lib/knit.xsl  -out $outfile -param docbase $fileurl $*

