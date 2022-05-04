#!/usr/bin/python
import seaborn as sns
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pylab import savefig
import os, re
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('folder') # final_outputs/align_gold
args = parser.parse_args()

#gold-aba.align      gold-identity.align      gold-lstm.align      gold-rule-based.align      gold-smt.align      gold-transformer.align
#gold-aba+lex.align  gold-identity+lex.align  gold-lstm+lex.align  gold-rule-based+lex.align  gold-smt+lex.align  gold-transformer+lex.align

def read_all(folder):
    systems = ['Rule-based', 'ABA', 'SMT', 'LSTM', 'Transformer']
    s2s = OrderedDict()
    for sys1 in reversed(systems):
        s2s[sys1] = {}
        for sys2 in systems:
            #if sys1 == sys2:
            #    continue
            name1 = sys1.lower()
            name2 = sys2.lower()

            s2s[sys1][sys2] = compare(folder + '/gold-' + name1 + '.align',
                                          folder + '/gold-' + name2 + '.align')

    sns.set_style({'font.sans-serif':['Avenir']})

                
    cm = pd.DataFrame(s2s)
    cm = cm.reindex(index=systems.reverse(), columns=systems[:]).transpose()
    print(cm)
    svm = sns.heatmap(cm, annot=True, fmt='.2f', cmap="Blues", cbar=None, vmin=85)
    svm.xaxis.tick_top()
    svm.xaxis.set_label_position('top')
    figure = svm.get_figure()    
    figure.savefig(folder + '/sys-compare.pdf', dpi=400)

def compare(sys1, sys2):
    total, same = 0, 0
    with open(sys1) as s1fp, open(sys2) as s2fp:
        for s1line, s2line in zip(s1fp, s2fp):
            #print(s1line)
            #print(s2line)
            s1_toks = re.split(' +', s1line.strip())
            s2_toks = re.split(' +', s2line.strip())
            #s1_toks = re.sub(' (?! )', '', s1line.strip().split('\t')[-1]).split(' ')
            #s2_toks = re.sub(' (?! )', '', s2line.strip().split('\t')[-1]).split(' ')
            

            if len(s1_toks) != len(s2_toks):
                print(sys1, sys2)
                print(s1_toks)
                print(s2_toks)
                print(len(s1_toks))
                print(len(s2_toks))
                input()

            for s1_tok, s2_tok in zip(s1_toks, s2_toks):
                if s1_tok == s2_tok:
                    same += 1
                total += 1
    return round((same/total)*100, 2)

read_all(args.folder)
