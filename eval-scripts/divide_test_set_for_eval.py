#!/usr/bin/python

SUBCORPUS_IDX=10 # the index in the meta file of the subcorpus info

def process_translation(translation_file, meta_info_file, output_prefix):
    # divide translation file and meta info into separate files
    subcorpus2text = {}
    with open(translation_file) as tfp, open(meta_info_file) as mfp:
        for t, m in zip(tfp, mfp):
            subcorpus = m.split('\t')[SUBCORPUS_IDX]

            if subcorpus not in subcorpus2text:
                subcorpus2text[subcorpus] = []
            subcorpus2text[subcorpus].append(t.strip())

    # output subcorpus files
    for subcorpus in subcorpus2text:
        with open(output_prefix + '.' + subcorpus, 'w') as fp:
            for line in subcorpus2text[subcorpus]:
                fp.write(line + '\n')
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('translation_file')
    parser.add_argument('meta_info_file')
    parser.add_argument('output_prefix')
    args = parser.parse_args()

    process_translation(args.translation_file, args.meta_info_file, args.output_prefix)

