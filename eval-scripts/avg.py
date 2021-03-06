#!/usr/bin/python
import statistics
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('scores', help='space-separated scores')
parser.add_argument('-p', action='store_true', default=False, help='turn into a percentage by multiplying by 100')
parser.add_argument('--plain_text', action='store_true', default=False, help='return plain text rather than latex-escaped')
args = parser.parse_args()

roundnum=2
scores = [float(score) for score in args.scores.split()]
if args.p:
    scores = [score * 100 for score in scores]

if len(scores) == 1:
    print("{:.2f}".format(round(scores[0], roundnum)))
else:
    avg = "{:.2f}".format(round(statistics.mean(scores), roundnum))
    stddev = "{:.2f}".format(round(statistics.stdev(scores), roundnum))
    if args.plain_text:
        print(avg + r'±' + stddev)
    else:
        print(avg + r'\\textpm' + stddev)
