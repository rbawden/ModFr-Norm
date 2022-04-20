#!/usr/bin/python
from nltk.metrics.distance import edit_distance_align
from utils import *
from align_levenshtein import read_file, homogenise, align
import os

def word_acc_final(ref, pred, align_types, cache_file=None):
    s1, s2 = read_file(ref), read_file(pred)
    scores = {'ref': None, 'pred': None, 'both': None}

    # do this unless only 'pred' is chosen
    if align_types != ['pred']:
        alignment_fwd = align(s1, s2, cache_file=cache_file)
        scores['ref'] = word_acc(alignment_fwd)

    # do this unless only 'ref' is chosen
    if align_types != ['ref']:
        alignment_bckwd = align(s2, s1, cache_file=cache_file)
        scores['pred'] = word_acc(alignment_bckwd)

    if 'both' in align_types:
        scores['both'] = (scores['ref'] + scores['pred']) / 2
        
    return scores


def word_acc(alignments):
    correct, total = 0, 0
    for sent in alignments:
        for word in sent:
            if word[0] == word[1]:
                correct += 1
            total += 1
    return (correct / total) * 100

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('pred')
    parser.add_argument('-a', '--align_types', help='Which file\'s tokenisation to use as reference for alignment. Valid choices are both, ref, pred. Multiple choices are possible (comma separated)', required=True)
    parser.add_argument('-c', '--cache', help='pickle file containing cached alignments', default=None)
    args = parser.parse_args()
    align_types = args.align_types.split(',')
    assert all([x in ['both', 'ref', 'pred'] for x in align_types]), 'Align types must belong to "both", "ref", "pred"'
    
    args = parser.parse_args()
    scores = word_acc_final(args.ref, args.pred, align_types, cache_file=args.cache)
    print(' '.join([str(scores[x]) for x in align_types]))


                      
