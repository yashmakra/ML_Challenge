"""Rebuild matching_results.tsv from persisted candidate scores at a new threshold.

predict.py saves one float32 score per candidate, in the same order the candidate
IDs are written to candidate_pairs.tsv, so changing the operating point costs a
single streaming pass instead of a full re-inference run.
"""
import argparse, csv, sys
from pathlib import Path

import numpy as np


def apply_threshold(candidate_path, scores_dir, out_path, threshold):
    shards = sorted(Path(scores_dir).glob('scores_*.npy'))
    if not shards:
        raise SystemExit(f'no score shards in {scores_dir}; re-run predict to persist them')
    rows = links = 0
    shard_i = 0
    buf = np.load(shards[0])
    pos = 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(candidate_path, encoding='utf-8') as fin, out_path.open('w', encoding='utf-8', newline='') as fout:
        fin.readline()
        w = csv.writer(fout, delimiter='\t', lineterminator='\n')
        w.writerow(['source1_entity_id', 'matched_entity_ids'])
        for line in fin:
            sid, _, rest = line.rstrip('\n').partition('\t')
            ids = [x for x in rest.split(',') if x]
            take = []
            for cid in ids:
                while pos >= len(buf):
                    shard_i += 1
                    if shard_i >= len(shards):
                        raise SystemExit('score shards exhausted before candidates ran out')
                    buf = np.load(shards[shard_i])
                    pos = 0
                if buf[pos] >= threshold:
                    take.append(cid)
                pos += 1
            rows += 1
            links += len(take)
            w.writerow([sid, ','.join(take)])
    leftover = len(buf) - pos + sum(len(np.load(s)) for s in shards[shard_i + 1:])
    if leftover:
        raise SystemExit(f'{leftover} scores unconsumed; candidate file and scores are misaligned')
    print(f'threshold {threshold} rows {rows} links {links}', flush=True)
    return dict(rows=rows, links=links, threshold=threshold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--scores-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', required=True, type=float)
    a = ap.parse_args()
    apply_threshold(a.candidates, a.scores_dir, a.out, a.threshold)


if __name__ == '__main__':
    sys.exit(main())
