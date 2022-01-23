#!/usr/bin/python
from plot_val import read_file

def select(data, metric, reverse=False):
    best_score = None
    best_checkpoint = None
    for checkpoint in sorted(data['all'], key=lambda x: x['checkpoint']):
        if checkpoint['metric'] == metric:
            score = checkpoint['score']
            if reverse:
                if best_score is None or score < best_score:
                    best_score = score
                    best_checkpoint = checkpoint['checkpoint']
            else:
                if best_score is None or score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint['checkpoint']
    return best_checkpoint, best_score


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('valid_eval_file',
                        help='the validation file containing the scores of each checkpoint')
    parser.add_argument('metric',
                        help='metric to be maximised')
    args = parser.parse_args()
    data = read_file(args.valid_eval_file)
    best_checkpoint, best_score = select(data, args.metric)

    print(best_checkpoint)
