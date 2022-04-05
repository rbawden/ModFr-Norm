#!/usr/bin/python
from nltk.metrics.distance import edit_distance_align
from utils import *
import pickle
import os

def read_file(filename):
    contents = []
    with open(filename) as fp:
        for line in fp:
            contents.append(re.sub('[  ]+', ' ', basic_tokenise(line).strip()))
    return contents


def homogenise(sent):
    sent = sent.lower()
    sent = sent.replace("oe", "œ").replace("OE", "Œ")
    replace_from = "ǽǣáàâäąãăåćčçďéèêëęěğìíîĩĭıïĺľłńñňòóôõöøŕřśšşťţùúûũüǔỳýŷÿźẑżžÁÀÂÄĄÃĂÅĆČÇĎÉÈÊËĘĚĞÌÍÎĨĬİÏĹĽŁŃÑŇÒÓÔÕÖØŔŘŚŠŞŤŢÙÚÛŨÜǓỲÝŶŸŹẐŻŽſ"
    replace_into = "ææaaaaaaaacccdeeeeeegiiiiiiilllnnnoooooorrsssttuuuuuuyyyyzzzzAAAAAAAACCCDEEEEEEGIIIIIIILLLNNNOOOOOORRSSSTTUUUUUUYYYYZZZZs"
    table = sent.maketrans(replace_from, replace_into)
    return sent.translate(table)


def align(sents_ref, sents_pred, cache_file=None):
    alignments, cache = [], {}
    if cache_file is not None and os.path.exists(cache_file):
        cache = pickle.load(open(cache_file, 'rb'))
    for sent_ref, sent_pred in zip(sents_ref, sents_pred):
        if (sent_ref, sent_pred) in cache and 'align' in cache[(sent_ref, sent_pred)]:
            alignment = cache[(sent_ref, sent_pred)]['align']
            alignments.append(alignment)
        else:
            backpointers = edit_distance_align(homogenise(sent_ref), homogenise(sent_pred))
            alignment, current_word, seen1, seen2 = [], ['', ''], [], []
            for i_ref, i_pred in backpointers:
                # spaces in both, add straight away
                if i_ref < len(sent_ref) and sent_ref[i_ref] == ' ' and i_pred < len(sent_pred) and sent_pred[i_pred] == ' ':
                    alignment.append((current_word[0].strip(), current_word[1].strip()))
                    current_word = ['', '']
                    seen1.append(i_ref)
                    seen2.append(i_pred)
                else:
                    end_space = '▁'
                    if i_ref < len(sent_ref) and i_ref not in seen1:
                        current_word[0] += sent_ref[i_ref]
                        seen1.append(i_ref)
                    if i_pred < len(sent_pred) and i_pred not in seen2:
                        current_word[1] += sent_pred[i_pred]
                        end_space = '' if space_after(i_pred, sent_pred) else '▁'
                        seen2.append(i_pred)
                    if i_ref < len(sent_ref) and sent_ref[i_ref] == ' ':
                        alignment.append((current_word[0].strip(), current_word[1].strip() + end_space))
                        current_word = ['', '']
            # final word
            alignment.append((current_word[0].strip(), current_word[1].strip()))
            # check that both strings are entirely covered
            recovered1 = re.sub(' +', ' ', ' '.join([x[0] for x in alignment]))
            recovered2 = re.sub(' +', ' ', ' '.join([x[1] for x in alignment]))

            assert recovered1 == sent_ref, sent_ref
            assert re.sub('[▁ ]', '', recovered2) == re.sub('[▁ ]', '', sent_pred), sent_pred
            alignments.append(alignment)
            if cache is not None:
                if (sent_ref, sent_pred) not in cache:
                    cache[(sent_ref, sent_pred)] = {}
                cache[(sent_ref, sent_pred)]['align'] = alignment
    # dump cache if specified
    if cache_file is not None:
        pickle.dump(cache, open(cache_file, 'wb'))
    return alignments

def space_after(idx, sent):
    if idx < len(sent) -1 and sent[idx + 1] == ' ':
        return True
    return False

def space_before(idx, sent):
    if idx > 0 and sent[idx - 1] == ' ':
        return True
    return False

def prepare_for_print(alignments):
    sents = []
    for align_sent in alignments:
        sent = ''
        for word1, word2 in align_sent:
            if word1 == word2:
                sent += word1 + ' '
            else:
                sent += word1 + '|||' + word2 + ' '
        sents.append(sent.strip(' '))
    return '\n'.join(sents)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ref')
    parser.add_argument('pred')
    parser.add_argument('-a', '--align_type', choices=('ref', 'pred'),
                        help='Which file\'s tokenisation to use as reference for alignment.')
    parser.add_argument('-c', '--cache', help='pickle cache file containing alignments', default=None)
    args = parser.parse_args()
    sents_ref, sents_pred = read_file(args.ref), read_file(args.pred)
    alignment = align(sents_ref, sents_pred, args.cache)
    print(prepare_for_print(alignment))


                      
