#!/bin/sh

# normalise quotes including when there are several consecutive ones, apostrophes, repeated spaces
perl -CS -Mutf8 -pe "s/[«»\“\”]+/\"/g" | \
    perl -CS -Mutf8 -pe "s/[‘’]+/'/g" | \
    perl -CS -Mutf8 -pe "s/ +/ /g" | \
    perl -CS -Mutf8 -pe "s/\"+/\"/g" | \
    perl -CS -Mutf8 -pe "s/'+/'/g" | \
    perl -CS -Mutf8 -pe "s/^ *//g" | \
    perl -CS -Mutf8 -pe "s/ *$//g"
    
    
