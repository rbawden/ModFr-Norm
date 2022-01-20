#!/usr/bin/python
import os

def read_file(filename):
    contents = []
    with open(filename) as fp:
        for line in fp:
            contents.append(line.strip())
    return contents


def threshold_met(sent, threshold):
    if threshold is None or len(sent.split()) >= int(threshold):
        return True
    return False

def filter_train(train_prefix, srctest, trgtest, threshold=None):
    lines_skipped = 0
    num_toks_in_lines_skipped = []
    with open(train_prefix + '.src') as sfp, open(train_prefix + '.trg') as tfp, open(train_prefix + '.meta') as mfp:
        for s, t, m in zip(sfp, tfp, mfp):
            s, t, m = s.strip(' \n'), t.strip(' \n'), m.strip(' \n')
            if (s in srctest or s in trgtest) and threshold_met(s, threshold):
                lines_skipped += 1
                num_toks_in_lines_skipped.append(len(s.split()))
                continue
            elif (t in srctest or t in trgtest) and threshold_met(t, threshold):
                num_toks_in_lines_skipped.append(len(t.split()))
                lines_skipped += 1
                continue
            elif s == '':
                num_toks_in_lines_skipped.append(len(s.split()))
                lines_skipped += 1
                continue
            elif t == '':
                num_toks_in_lines_skipped.append(len(t.split()))
                lines_skipped += 1
                continue
            else:
                print(s + '\t' + t + '\t' + m)
    os.sys.stderr.write('Number of lines skipped: ' + str(lines_skipped) + '\n')
    linelen2occs = {linelen: num_toks_in_lines_skipped.count(linelen) for linelen in set(num_toks_in_lines_skipped)}
    os.sys.stderr.write('Distribution of sentence lengths (in tokens) of lines skipped: ' + str(linelen2occs) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('testset_prefix')
    parser.add_argument('trainset_prefix')
    parser.add_argument('-t', '--threshold', default=None,
                        help='Identical sentences are discarded if they contain this number of tokens or above. If None, all identical sentences are discarded')
    args = parser.parse_args()

    srctest = read_file(args.testset_prefix + '.src')
    trgtest = read_file(args.testset_prefix + '.trg')

    filter_train(args.trainset_prefix, srctest, trgtest, args.threshold)
