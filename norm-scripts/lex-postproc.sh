#!/bin/sh

thisdir=`dirname $0`

bash $thisdir/pre-normalise.sh | perl $thisdir/alModFrNormalise.pl -l
