"""Reproduce indexing, entity-level validation, training, and submission outputs."""
import argparse,subprocess,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from build_index import build
from train_model import train
from predict import predict
from resolve_conflicts import resolve
from validate_outputs import validate_streaming
from extra_experiments import blocking_study,run_ablations,second_holdout,singleton_rules

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['all','train','predict','resolve','validate'])
    ap.add_argument('--data-root',required=True,type=Path,help='Path containing train/, test/, and sibling utils/')
    ap.add_argument('--cache-dir',type=Path,default=Path(__file__).resolve().parent/'cache')
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'output')
    ap.add_argument('--top-k',type=int,default=50)
    ap.add_argument('--final-candidates',type=int,default=75)
    ap.add_argument('--threshold',type=float,default=None,help='Inference cutoff; defaults to the F0.5-optimal value in selection.json')
    a=ap.parse_args();root=Path(__file__).resolve().parent;a.cache_dir.mkdir(parents=True,exist_ok=True)
    if a.stage in ('all','train'):
        build(a.data_root,'train',a.cache_dir/'train_targets.sqlite')
        if a.top_k==50 and a.final_candidates==75:
            baseline=a.cache_dir/'baseline_50'
            train(a.data_root,a.cache_dir/'train_targets.sqlite',a.cache_dir/'sample_pairs_50_50.npz',baseline/'models',baseline/'reports',50,50)
        sample=a.cache_dir/f'sample_pairs_{a.top_k}_{a.final_candidates}.npz'
        train(a.data_root,a.cache_dir/'train_targets.sqlite',sample,root/'models',root/'reports',a.top_k,a.final_candidates)
        if a.top_k==50 and a.final_candidates==75:
            for filename in ('experiment_results.csv','threshold_search.csv'):
                main=pd.read_csv(root/'reports'/filename);earlier=pd.read_csv(baseline/'reports'/filename)
                pd.concat([main,earlier],ignore_index=True).to_csv(root/'reports'/filename,index=False)
            blocking_study(a.data_root,a.cache_dir/'train_targets.sqlite',root/'reports')
            run_ablations(sample,root/'reports')
            second_holdout(a.data_root,a.cache_dir/'train_targets.sqlite',sample,root/'models',root/'reports')
            singleton_rules(sample,root/'reports')
    if a.stage in ('all','predict'):
        build(a.data_root,'test',a.cache_dir/'test_targets.sqlite')
        predict(a.data_root,a.cache_dir/'test_targets.sqlite',root/'models',a.output_dir,threshold=a.threshold)
    if a.stage in ('all','predict','resolve'):
        resolve(a.output_dir/'matching_results.tsv',a.data_root,root/'models',a.output_dir/'matching_results.tsv',root/'reports'/'conflict_resolution.json')
    if a.stage in ('all','validate'):
        validate_streaming(a.data_root,a.output_dir)
        # The supplied validator loads all candidate IDs into RAM. The streaming
        # check above covers both files; run its memory-light matching check too.
        cmd=[sys.executable,str(a.data_root.parent/'utils'/'validate_submission.py'),'--matching',str(a.output_dir/'matching_results.tsv'),'--candidate',str(a.cache_dir/'candidate_check_done_by_streaming_validator.tsv'),'--test-dir',str(a.data_root/'test')]
        subprocess.run(cmd,check=True)
if __name__=='__main__':main()
