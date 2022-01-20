#!/usr/bin/python
import os
import re

def read_toc(toc_file, header=True):
    headers = None
    toc = {}
    with open(toc_file) as fp:
        for line in fp:
            if headers is None:
                headers = line.strip('\n').split('\t')
            else:
                info = line.strip('\n').split('\t')
                filename = info[headers.index('file')]
                toc[filename] = {h: info[i] for i, h in enumerate(headers)}
    return toc, headers

def create_set(subfolder, toc, headers, output_file):
    # look up all files in the folder
    with open(output_file, 'w') as ofp:
        for filename in sorted(os.listdir(subfolder)):
            meta_info = toc[filename]

            # remove outer square brackets
            for aspect in 'Publication date', 'publication place':
                meta_info[aspect] = re.sub('^\[(.+?)\]$', r'\1', meta_info[aspect])
            with open(subfolder + '/' + filename) as fp:
                for line in fp:
                    #print(line)
                    #print( '\t'.join([meta_info[h].strip() for h in headers]))
                    #input()
                    ofp.write(line.strip(' \n') + '\t' + \
                              '\t'.join([meta_info[h].strip() for h in headers]) + '\n')
    return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('table_of_contents')
    parser.add_argument('splits_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    toc, headers = read_toc(args.table_of_contents)
    create_set(args.splits_dir + '/train/', toc, headers, args.output_dir + '/train/train.tsv')
    create_set(args.splits_dir + '/dev/', toc, headers, args.output_dir + '/dev/dev.tsv')
    create_set(args.splits_dir + '/test/', toc, headers, args.output_dir + '/test/test.tsv')
