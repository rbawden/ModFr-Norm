#!/usr/bin/python
from nltk.metrics.distance import edit_distance
from utils import *
from align_levenshtein import homogenise, read_file
import pickle
import os

def levenshtein_score(sents_ref, sents_pred, align_type='ref', cache_file=None):
    alignments = []
    score = 0
    num_chars = 0
    cache = {}
    if cache_file is not None and os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, 'rb'))
    for sent_ref, sent_pred in zip(sents_ref, sents_pred):
        if align_type == 'ref':
            if (sent_ref, sent_pred) in cache and 'score' in cache[(sent_ref, sent_pred)]:
                score = cache[(sent_ref, sent_pred)]['score']
            else:
                score = edit_distance(homogenise(sent_ref), homogenise(sent_pred))
                if (sent_ref, sent_pred) not in cache:
                    cache[(sent_ref, sent_pred)] = {}
                cache[(sent_ref, sent_pred)]['score'] = score
            num_chars += len(sent_ref)
        else:
            if (sent_pred, sent_ref) in cache and 'score' in cache[(sent_pred, sent_ref)]:
                score =	cache[(sent_pred, sent_ref)]['score']
            else:
                score = edit_distance(homogenise(sent_pred), homogenise(sent_ref))
                if (sent_pred, sent_ref) not in cache:
                    cache[(sent_pred, sent_ref)] = {}
                cache[(sent_pred, sent_ref)]['score'] = score
            num_chars += len(sent_pred)
    # dump cache file
    if cache_file is not None:
        pickle.dump(cache, open(cache_file, 'wb'))
    return (score/num_chars) * 100
        

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('pred')
    parser.add_argument('-a', '--align_type', choices=('ref', 'pred'),
                        help='Which file\'s tokenisation to use as reference for alignment.')
    parser.add_argument('-c', '--cache', help='pickle cache file to store scores/alignments', default=None)
    args = parser.parse_args()
    sents_ref, sents_pred = read_file(args.ref), read_file(args.pred)
    score = levenshtein_score(sents_ref, sents_pred, cache_file=args.cache)
    print(score)


                      
