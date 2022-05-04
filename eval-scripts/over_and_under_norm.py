#!/usr/bin/python                                                                                                                                                         
import re
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('gold_align')
parser.add_argument('sys_align')
args = parser.parse_args()

stats = {'keep': {'keep': 0, 'nokeep': 0}, 'nokeep': {'keep': 0, 'nokeep': 0}}

# check how conservative models are
with open(args.gold_align) as gfp, open(args.sys_align) as sfp:
    for gline, sline in zip(gfp, sfp):

        gold_toks = re.split(' +', gline)
        sys_toks = re.split(' +', sline)
        if len(gold_toks) != len(sys_toks):
            print(gline)
            print(sline)
            print(gold_toks)
            print(sys_toks)
        assert len(gold_toks) == len(sys_toks)

        for g, s in zip(gold_toks, sys_toks):
            # if no change in gold
            gtype, stype = 'keep', 'keep'
            if '|||' in g:
                gtype = 'nokeep'
            if '|||' in s:
                stype = 'nokeep'
            stats[gtype][stype] += 1

print('Correctly did not change = ', stats['keep']['keep'])
print('% of those that should have been kept = ', stats['keep']['keep']/sum(stats['keep'].values()))
print('Should have kept same and modified = ', stats['keep']['nokeep'])
print('Should have modified and did not = ', stats['nokeep']['keep'])
print('Correctly did change', stats['nokeep']['nokeep'])
