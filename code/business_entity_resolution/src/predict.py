"""Parallel, ordered test inference over disk-backed FTS5 indexes."""
import csv,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from blocking import connect,retrieve
from features import pair_features,rank_candidates

_CON=None;_MODEL=None;_CFG=None;_SHARD=None;_SCORES=None;_THRESHOLD=None
def _init_worker(index_path,model_dir,shard_dir,scores_dir,threshold):
    global _CON,_MODEL,_CFG,_SHARD,_SCORES,_THRESHOLD
    _CON=connect(index_path);_CFG=json.loads((Path(model_dir)/'selection.json').read_text(encoding='utf-8'))
    _MODEL=None if _CFG['experiment']=='weighted_similarity' else joblib.load(Path(model_dir)/'best_model.joblib')
    _SHARD=Path(shard_dir);_SCORES=Path(scores_dir);_THRESHOLD=threshold

def _work(task):
    idx,records=task;start=time.time();features=[];entities=[]
    for sid,name,addr,country in records:
        cands=rank_candidates(name,addr,retrieve(_CON,name,addr,country,_CFG['candidate_k'],_CFG['candidate_k']),_CFG['final_candidates'])
        ids=[]
        for (mid,mn,ma,mc),nf,af in cands:
            ids.append(mid);features.append(pair_features(name,addr,country,mn,ma,mc,nf,af))
        entities.append((sid,ids))
    X=np.vstack(features) if features else np.empty((0,len(_CFG['feature_names'])),dtype=np.float32)
    scores=(.58*X[:,0]+.42*X[:,14]) if _MODEL is None else _MODEL.predict_proba(X)[:,1] if len(X) else np.empty(0)
    # Persisted so a threshold change can be applied without re-running inference.
    np.save(_SCORES/f'scores_{idx:05d}.npy',np.asarray(scores,dtype=np.float32))
    links=0;pos=0
    with (_SHARD/f'candidate_{idx:05d}.tsv').open('w',encoding='utf-8',newline='') as fc,(_SHARD/f'matching_{idx:05d}.tsv').open('w',encoding='utf-8',newline='') as fm:
        wc=csv.writer(fc,delimiter='\t',lineterminator='\n');wm=csv.writer(fm,delimiter='\t',lineterminator='\n')
        for sid,ids in entities:
            take=[ids[j] for j,p in enumerate(scores[pos:pos+len(ids)]) if p>=_THRESHOLD]
            wc.writerow([sid,','.join(ids)]);wm.writerow([sid,','.join(take)])
            pos+=len(ids);links+=len(take)
    return idx,len(entities),links,round(time.time()-start,1)

def _tasks(data_root,chunk_size):
    for idx,chunk in enumerate(pd.read_csv(Path(data_root)/'test'/'test_source1.tsv',sep='\t',dtype=str,keep_default_na=False,chunksize=chunk_size)):
        yield idx,list(chunk.itertuples(index=False,name=None))

def predict(data_root,index_path,model_dir,output_dir,workers=6,chunk_size=2000,threshold=None):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    shard_dir=output_dir.parent/'cache'/'shards';shard_dir.mkdir(parents=True,exist_ok=True)
    scores_dir=output_dir.parent/'cache'/'scores';scores_dir.mkdir(parents=True,exist_ok=True)
    if threshold is None:
        threshold=json.loads((Path(model_dir)/'selection.json').read_text(encoding='utf-8'))['threshold']
    print('inference threshold',threshold,flush=True)
    start=time.time();n=links=chunks=0
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_worker,initargs=(str(Path(index_path).resolve()),str(Path(model_dir).resolve()),str(shard_dir.resolve()),str(scores_dir.resolve()),float(threshold))) as pool:
        source=iter(_tasks(data_root,chunk_size));pending={}
        for _ in range(workers*2):
            try:task=next(source)
            except StopIteration:break
            pending[pool.submit(_work,task)]=task[0]
        while pending:
            future=next(as_completed(pending));pending.pop(future)
            idx,count,found,seconds=future.result()
            n+=count;links+=found;chunks+=1
            if chunks%10==0:print('predicted entities',n,'links',links,'chunks',chunks,'seconds',round(time.time()-start),'last chunk seconds',seconds,flush=True)
            try:task=next(source)
            except StopIteration:continue
            pending[pool.submit(_work,task)]=task[0]
    for prefix,filename,header in [('candidate','candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids']),('matching','matching_results.tsv',['source1_entity_id','matched_entity_ids'])]:
        with (output_dir/filename).open('w',encoding='utf-8',newline='') as out:
            out.write('\t'.join(header)+'\n')
            for idx in range(chunks):
                part=shard_dir/f'{prefix}_{idx:05d}.tsv'
                with part.open('r',encoding='utf-8') as inp:
                    for line in inp:out.write(line)
                part.unlink()
    print('DONE entities',n,'links',links,'seconds',round(time.time()-start),flush=True)
