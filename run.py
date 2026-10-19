"""Epoch30 release CLI: result verification, official-data check, cache build, train, evaluate."""
from pathlib import Path
import argparse,json,hashlib,sys,random,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def grouped(c,groups):
    import numpy as np
    c=np.asarray(c,dtype=np.int64);B=np.zeros((18,len(groups)),np.int64)
    for i,(_,ids) in enumerate(groups):B[ids,i]=1
    if not np.all(B.sum(1)==1):raise ValueError('Groups must partition all18 labels')
    m=B.T@c@B;u=m.sum(0)+m.sum(1)-m.diagonal()
    if not np.all(u>0):raise ValueError('Absent group: full taxonomy score not defined')
    v=m.diagonal()/u
    return {'miou':float(v.mean()),'groups':[n for n,_ in groups],'iou':v.tolist(),'confusion':m.tolist()}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['verify-results','check-data','evaluate','build-cache','train'])
    p.add_argument('--data-root',type=Path);p.add_argument('--cache-root',type=Path);p.add_argument('--backbone',type=Path)
    p.add_argument('--split',choices=['train','val','test'],default='test');p.add_argument('--scene-limit',type=int,default=0)
    p.add_argument('--output',type=Path,default=ROOT/'runs/output');p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--start-training',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();recipe=read(ROOT/'configs/recipe.json');defs=read(ROOT/'configs/groups.json');splits=read(ROOT/'data/splits.json')
    if a.command=='verify-results':
        import numpy as np
        from metrics import evaluate
        value=read(ROOT/'results/test_epoch30.json');c=np.asarray(value['confusion18'],dtype=np.int64)
        assert c.sum()==value['points']==2_294_339_698
        assert np.array_equal(sum((np.asarray(s['confusion18'],dtype=np.int64) for s in value['per_scene']),np.zeros((18,18),np.int64)),c)
        actual=evaluate(c)
        expected=read(ROOT/'results/grouped_epoch30.json')
        for k,g in defs.items():assert abs(grouped(c,g)['miou']-expected[k]['miou'])<1e-12
        print(json.dumps({'status':'PASS','scope':'Saved result arithmetic; not GPU inference','groups':{k:grouped(c,g)['miou'] for k,g in defs.items()}},indent=2));return
    selected=splits[a.split][:a.scene_limit or None]
    if a.command=='check-data':
        if not a.data_root:p.error('--data-root required')
        for part,scenes in splits.items():
            for s in scenes:
                for f in ['coord.npy','color.npy','normal.npy','class.npy']:
                    if not (a.data_root/s/f).is_file():raise FileNotFoundError(a.data_root/s/f)
        print(json.dumps({'status':'PASS','split_counts':{k:len(v) for k,v in splits.items()},'scope':'File presence, not full checksum'}));return
    if a.cache_root is None:p.error('--cache-root required; choose a directory separate from original data')
    if a.data_root and (a.cache_root.resolve()==a.data_root.resolve() or a.cache_root.resolve().is_relative_to(a.data_root.resolve())):raise ValueError('Never write cache inside original dataset')
    if a.command=='train' and a.dry_run:
        print(json.dumps({'operation':'train','executed':False,'epochs':a.epochs,'train_scenes':len(splits['train']),'validation_scenes':len(splits['val']),'recipe':recipe},indent=2));return
    import numpy as np
    import torch
    from head import MultiScaleSegHead
    from scene_cache import ensure_scene,evaluate_split
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for model execution')
    torch.set_num_threads(8);torch.manual_seed(recipe['seed']);np.random.seed(recipe['seed']);random.seed(recipe['seed'])
    if a.command=='build-cache':
        if not a.backbone or not a.data_root:p.error('--backbone and --data-root required')
        if sha(a.backbone)!=recipe['backbone_sha256']:raise ValueError('Wrong backbone weights')
        sys.path.insert(0,str(ROOT/'src/_vendor'));import utonia
        encoder=utonia.load(str(a.backbone),custom_config={'enable_flash':False,'enc_patch_size':[64]*5}).cuda().eval();encoder.requires_grad_(False)
        spec={'head_code_sha256':sha(ROOT/'src/head.py'),'cache_code_sha256':sha(ROOT/'src/scene_cache.py'),'encoder_sha256':sha(a.backbone),'seed':recipe['seed'],'rebuild_scope':'Fresh build; no claim of bitwise equivalence to historical cached features'}
        cache_id=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest();spec['cache_id']=cache_id
        existing=a.cache_root/'cache_spec.json'
        if existing.exists() and read(existing)!=spec:raise ValueError('Existing cache differs; use a new empty directory')
        write(existing,spec)
        for i,s in enumerate(selected):ensure_scene(a.data_root,s,a.cache_root,encoder,cache_id,16384,lambda **kw:None);print(f'Cache {i+1}/{len(selected)} {s}',flush=True)
        return
    spec=read(a.cache_root/'cache_spec.json')
    if spec['head_code_sha256']!=sha(ROOT/'src/head.py') or spec['cache_code_sha256']!=sha(ROOT/'src/scene_cache.py'):raise ValueError('Feature-cache code fingerprint differs')
    if spec['encoder_sha256']!=recipe['backbone_sha256']:raise ValueError('Feature-cache backbone differs')
    if a.command=='evaluate':
        if not a.data_root:p.error('--data-root required')
        ck=torch.load(ROOT/'checkpoints/head_epoch30.pth',map_location='cpu',weights_only=True)
        h=MultiScaleSegHead(ck['channels']).cuda().eval();h.load_state_dict(ck['head']);h.requires_grad_(False)
        result=evaluate_split(a.data_root,selected,a.cache_root,h,lambda **kw:None)
        result.update(epoch=30,split=a.split,partial_data=bool(a.scene_limit),grouped={k:grouped(result['confusion18'],g) for k,g in defs.items()})
        result.pop('primary',None);result.pop('auxiliary',None);write(a.output/'evaluation.json',result);print(json.dumps({'scene_count':len(selected),'points':result['points'],'groups':{k:v['miou'] for k,v in result['grouped'].items()}},indent=2));return
    if not a.start_training:p.error('Training is opt-in: --start-training required; --dry-run prints the recipe')
    if not a.data_root:p.error('--data-root required')
    if not 1<=a.epochs<=30:raise ValueError('Release recipe supports1..30epochs; new recipes require separate experiments')
    if a.output.resolve().is_relative_to(a.data_root.resolve()) or a.output.resolve().is_relative_to(a.cache_root.resolve()):raise ValueError('Output must be separate from original data and feature cache')
    # Re-training adapter for the published recipe; historical cache/RNG chronology is not reconstructed.
    channels=read(a.cache_root/splits['train'][0]/'SCENE_COMPLETE.json')['channels'];h=MultiScaleSegHead(channels).cuda()
    opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.01);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
    blocks=[]
    for s in splits['train']:
        m=read(a.cache_root/s/'SCENE_COMPLETE.json');assert m['cache_id']==spec['cache_id'];blocks.extend(a.cache_root/s/f for f in m['blocks'])
    history=[]
    for ep in range(1,a.epochs+1):
        h.train();random.shuffle(blocks);loss_sum=n=0
        for path in blocks:
            b=torch.load(path,map_location='cpu',weights_only=True);fs=[x.float().cuda() for x in b['features']];inv=[x.long().cuda() for x in b['inverses']];target=b['labels'].long().cuda()
            opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(h(fs,inv),target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();opt.step();loss_sum+=float(loss.detach())*len(target);n+=len(target)
        sched.step()
        for group in opt.param_groups:group['lr']=max(group['lr'],1e-5)
        row={'epoch':ep,'loss':loss_sum/n}
        if ep%5==0:
            validation=evaluate_split(a.data_root,splits['val'],a.cache_root,h,lambda **kw:None)
            row['validation']={'points':validation['points'],'grouped':{k:grouped(validation['confusion18'],g) for k,g in defs.items()}}
        history.append(row);a.output.mkdir(parents=True,exist_ok=True)
        torch.save({'epoch':ep,'head':h.state_dict(),'channels':channels,'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'history':history},a.output/f'epoch_{ep:04d}.pth');write(a.output/'history.json',history)
        print(f'Epoch {ep}/{a.epochs} loss={row["loss"]:.5f}',flush=True)
if __name__=='__main__':main()
