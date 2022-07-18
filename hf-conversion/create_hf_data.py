#!/bin/python
import json

def create_json(src_file, trg_file, langpair):
    src, trg = langpair.split('-')
    with open(src_file) as sfp, open(trg_file) as tfp:
        for sline, tline in zip(sfp, tfp):
            #example = {'translation': {src: '<s> ' + sline.strip() + ' </s>', trg: '<s> ' + tline.strip() + ' </s>'}}
            #example = {'translation': {src: sline.strip(), trg: tline.strip()}}
            example = {'translation': {src: sline.strip() + ' </s>', trg: tline.strip() + ' </s>'}}
            print(json.dumps(example))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('src_file')
    parser.add_argument('trg_file')
    parser.add_argument('langpair', help='Dash-separated language pair (e.g. en-fr, src-trg)')
    args = parser.parse_args()

    create_json(args.src_file, args.trg_file, args.langpair)
