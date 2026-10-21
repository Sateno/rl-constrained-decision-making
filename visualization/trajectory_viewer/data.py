"""Read-only access to paired evaluation_trajectory_v1 archives."""
from pathlib import Path
import csv
import hashlib
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def outcome(d):
    success, collision = bool(d['state_success'][-1]), bool(d['state_collision'][-1])
    require(not (success and collision), 'Conflicting terminal flags')
    if success: return 'SUCCESS'
    if collision: return 'COLLISION'
    require(bool(d['truncated'][-1]), 'Episode has no recognized terminal outcome')
    return 'TIMEOUT'


def catalog(root):
    """Index paired rows without opening every trajectory archive."""
    records=[]
    for table in sorted((Path(root)/'results'/'tables').glob('*/evaluation_episode_results.csv')):
        suite=table.parent.name
        pairs={}
        with table.open(newline='',encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                key=(row['method'],int(row['train_seed']),int(row['episode']))
                modes=pairs.setdefault(key,{})
                require(row['projection_mode'] not in modes, f'Duplicate row in {table}: {key}')
                modes[row['projection_mode']]=row
        for (method,seed,episode),modes in sorted(pairs.items()):
            if set(modes)!={'disabled','enabled'}: continue
            off,on=modes['disabled'],modes['enabled']
            for field in ('checkpoint_sha256','evaluation_seed','layout_id','evaluation_policy_mode'):
                require(off[field]==on[field],f'Pair mismatch in {field}: {suite}/{method}/{seed}/{episode}')
            def terminal(row):
                return 'SUCCESS' if row['success']=='True' else 'COLLISION' if row['collision']=='True' else 'TIMEOUT'
            records.append({'suite':suite,'method':method,'train_seed':seed,'episode':episode,
                            'layout_id':off['layout_id'],'evaluation_seed':int(off['evaluation_seed']),
                            'execution':off['evaluation_policy_mode'],'off':terminal(off),'on':terminal(on),
                            'rows':modes,
                            'available':all((Path(root)/'runs'/'evaluation'/'final'/suite/method/f'seed{seed}'/f'projection_{mode}_trajectories.npz').is_file() for mode in ('disabled','enabled'))})
    require(records,'No paired results found. Select the repository root containing results/tables and runs/evaluation/final.')
    return records


def load_pair(root,suite,method,train_seed,episode,record=None):
    root=Path(root)
    if record is None:
        matches=[r for r in catalog(root) if (r['suite'],r['method'],r['train_seed'],r['episode'])==(suite,method,int(train_seed),int(episode))]
        require(len(matches)==1,'Selected pair is missing or ambiguous')
        record=matches[0]
    pair=[]
    for mode in ('disabled','enabled'):
        path=root/'runs'/'evaluation'/'final'/suite/method/f'seed{train_seed}'/f'projection_{mode}_trajectories.npz'
        prefix=f'episode_{episode:04d}_'
        with np.load(path,allow_pickle=False) as archive:
            require(str(archive['trajectory_archive_version'])=='evaluation_trajectory_v1','Unsupported trajectory schema')
            d={k[len(prefix):]:archive[k].copy() for k in archive.files if k.startswith(prefix)}
            require(bool(d),f'Episode {episode} missing from {path}')
            d['checkpoint_sha256']=str(archive['run_checkpoint_sha256'])
            d['goal_radius']=float(archive['run_goal_radius'])
            d['stochastic']=bool(archive['run_stochastic'])
        d.update(mode=mode,row=record['rows'][mode],source_archive=path.relative_to(root).as_posix(),
                 source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        row=d['row'];n=len(d['positions'])-1
        require(n>0 and n==int(row['episode_length']),'Episode length mismatch')
        require(d['positions'].shape==(n+1,2) and np.isfinite(d['positions']).all(),'Invalid positions')
        for field in ('headings','state_success','state_collision'):
            require(len(d[field])==n+1,f'Misaligned {field}')
        for field in ('action_exec_physical','projection_intervened','projection_slack_max','truncated'):
            require(len(d[field])==n,f'Misaligned {field}')
        require(float(d['dt'])>0,'Invalid timestep')
        require(int(d['seed'])==int(row['evaluation_seed']),'Evaluation seed mismatch')
        require(d['checkpoint_sha256']==row['checkpoint_sha256'],'Checkpoint hash mismatch')
        require(bool(d['stochastic'])==(row['evaluation_policy_mode']=='stochastic'),'Policy mode mismatch')
        require(bool(d['state_collision'][-1])==(row['collision']=='True'),'Collision mismatch')
        require(bool(d['state_success'][-1])==(row['success']=='True'),'Success mismatch')
        require(bool(np.all(d['projection_enabled'])) if mode=='enabled' else not bool(np.any(d['projection_enabled'])),'Projection mode mismatch')
        outcome(d)
        pair.append(d)
    for field in ('goal','obstacle_centers','obstacle_radii','obstacle_mask','seed','agent_radius','dt'):
        require(np.array_equal(pair[0][field],pair[1][field]),f'Pair mismatch: {field}')
    require(np.array_equal(pair[0]['positions'][0],pair[1]['positions'][0]),'Initial positions differ')
    require(pair[0]['headings'][0]==pair[1]['headings'][0],'Initial headings differ')
    require(pair[0]['checkpoint_sha256']==pair[1]['checkpoint_sha256'],'Paired checkpoint hashes differ')
    return pair
