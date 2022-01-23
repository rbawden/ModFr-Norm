#!/usr/bin/python
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
import re

ignore_these_metrics = ['lev_char', 'lev_tok', 'wordacc_macro_h2r', 'wordacc_macro_r2h', 'wordacc_micro_h2r', 'wordacc_micro_r2h']

def closest_factors(number):
    a, b, i = 1, number, 0
    while a < b:
        i += 1
        if number % i == 0:
            a = i
            b = number//a
    return b, a

def read_file(filename):
    all_data = {}
    with open(filename) as fp:
        for line in fp:
            line = line.strip().split('\t')
            checkpoint_match = re.match('.*?checkpoint(\d+)\.', line[0])
            if not checkpoint_match:
                continue
            checkpoint = int(checkpoint_match.group(1))
            for info in line[1].split(' '):
                name, metric = info.split('=')[0].split(',')
                # ignore certain metrics
                if metric in ignore_these_metrics:
                    continue
                score = info.split('=')[1]
                data = {}
                data['checkpoint'] = checkpoint
                data['metric'] = metric
                data['score'] = float(score)
                if metric == 'bleu':
                    data['score'] = float(score)/100
                if name not in all_data:
                    all_data[name] = []
                all_data[name].append(data)
    return all_data


def plot(all_data):
    num_rows, num_cols = closest_factors(len(all_data.keys()))
    fig, axes = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(8,5))
    for i, datatype in enumerate(all_data):
        row, col = (int(i/num_rows), i%num_rows)
        axes[row, col].set_title(datatype)
        axes[row, col].set_ylim(0, 1)

        df_data = DataFrame(all_data[datatype])
        sns.lineplot(ax=axes[row, col], data=df_data, x="checkpoint", y="score", style="metric", hue="metric")
        if i != 0:
            axes[row, col].get_legend().remove()
#        else:
#            axes[row, col].legend(loc='upper right', bbox_to_anchor=(1.25, 0.5), ncol=3)

    plt.savefig("valid_eval_plot.pdf")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('valid_eval',
                        help='the validation file containing the scores of each checkpoint')
    args = parser.parse_args()
    all_data = read_file(args.valid_eval)
    plot(all_data)
