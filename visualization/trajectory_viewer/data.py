"""Read-only access to published or local evaluation_trajectory_v1 archives."""
from pathlib import Path, PurePosixPath
import csv
import hashlib
import io
import json
import re
import numpy as np


BUNDLE = Path('results/trajectories/pap_evaluation_v1')
SOURCES = ('auto', 'published', 'local')
SOURCE_LABELS = {'published': 'Published evaluation · original recorded outcomes',
                 'local': 'Local evaluation archives'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _component(value):
    return isinstance(value, str) and bool(re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*', value))


def _relative(value):
    require(isinstance(value, str) and bool(value) and '\\' not in value,
            'Invalid published archive path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and '..' not in path.parts and str(path) == value,
            'Invalid published archive path')
    return path


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _table_hash(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def _dataset(root, source):
    """Choose one complete data source; never substitute local files for a bundle."""
    require(source in SOURCES, 'Unknown data source; choose auto, published or local')
    manifest_path = root / BUNDLE / 'manifest.json'
    if source == 'auto':
        source = 'published' if manifest_path.exists() else 'local'
    if source == 'local':
        tables = [(p.parent.name, p) for p in sorted(
            (root / 'results/tables').glob('*/evaluation_episode_results.csv'))]
        return source, None, tables
    require(manifest_path.is_file(),
            'Published trajectory manifest is missing. Restore results/trajectories/pap_evaluation_v1, '
            'or select --data-source local for preserved local archives.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(isinstance(manifest, dict), 'Invalid published trajectory manifest')
    require(manifest.get('schema_version') == 'pap_trajectory_bundle_v1',
            'Unsupported published trajectory manifest schema')
    require(manifest.get('dataset_id') == 'pap_evaluation_v1' and
            manifest.get('trajectory_archive_version') == 'evaluation_trajectory_v1',
            'Unsupported published trajectory dataset')
    require(isinstance(manifest.get('tables'), dict) and bool(manifest['tables']),
            'Published trajectory manifest has no tables')
    require(isinstance(manifest.get('archives'), dict) and bool(manifest['archives']),
            'Published trajectory manifest has no archives')
    tables = []
    for suite, item in sorted(manifest['tables'].items()):
        require(_component(suite) and isinstance(item, dict), 'Invalid published table entry')
        expected = f'results/tables/{suite}/evaluation_episode_results.csv'
        require(item.get('path') == expected and _hash(item.get('sha256')),
                'Invalid published table path or hash')
        path = root / expected
        require(path.is_file(), f'Published evaluation table is missing: {expected}')
        require(_table_hash(path) == item['sha256'], f'Published table hash mismatch: {expected}')
        tables.append((suite, path))
    for relative, item in manifest['archives'].items():
        parts = _relative(relative).parts
        require(len(parts) == 4 and parts[0] in manifest['tables'] and
                _component(parts[1]) and re.fullmatch(r'seed\d+', parts[2]) and
                parts[3] in ('projection_disabled_trajectories.npz',
                             'projection_enabled_trajectories.npz'),
                f'Invalid published archive path: {relative}')
        require(isinstance(item, dict) and _hash(item.get('sha256')) and
                type(item.get('size_bytes')) is int and item['size_bytes'] > 0 and
                type(item.get('episodes')) is int and item['episodes'] > 0 and
                item.get('source_archive') == 'runs/evaluation/final/' + relative,
                f'Invalid published archive metadata: {relative}')
        candidate = root / BUNDLE / relative
        require(candidate.resolve().is_relative_to((root / BUNDLE).resolve()),
                f'Published archive escapes dataset: {relative}')
    return source, manifest, tables


def outcome(d):
    success, collision = bool(d['state_success'][-1]), bool(d['state_collision'][-1])
    require(not (success and collision), 'Conflicting terminal flags')
    if success: return 'SUCCESS'
    if collision: return 'COLLISION'
    require(bool(d['truncated'][-1]), 'Episode has no recognized terminal outcome')
    return 'TIMEOUT'


def catalog(root, source='auto'):
    """Index paired rows and verify bundle/table identity without loading arrays."""
    root = Path(root).resolve()
    source, manifest, tables = _dataset(root, source)
    records=[]
    archive_counts={}
    for suite, table in tables:
        pairs={}
        with table.open(newline='',encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                require(_component(row['method']) and int(row['train_seed']) >= 0 and
                        int(row['episode']) >= 0, 'Invalid evaluation row identity')
                key=(row['method'],int(row['train_seed']),int(row['episode']))
                modes=pairs.setdefault(key,{})
                require(row['projection_mode'] not in modes, f'Duplicate row in {table}: {key}')
                require(row['projection_mode'] in ('disabled', 'enabled'), 'Unknown projection mode')
                modes[row['projection_mode']]=row
                relative=f"{suite}/{row['method']}/seed{int(row['train_seed'])}/projection_{row['projection_mode']}_trajectories.npz"
                archive_counts[relative]=archive_counts.get(relative,0)+1
        table_digest = _table_hash(table)
        for (method,seed,episode),modes in sorted(pairs.items()):
            if set(modes)!={'disabled','enabled'}:
                require(source != 'published', f'Unpaired published result: {suite}/{method}/{seed}/{episode}')
                continue
            off,on=modes['disabled'],modes['enabled']
            for field in ('checkpoint_sha256','evaluation_seed','layout_id','evaluation_policy_mode'):
                require(off[field]==on[field],f'Pair mismatch in {field}: {suite}/{method}/{seed}/{episode}')
            def terminal(row):
                return 'SUCCESS' if row['success']=='True' else 'COLLISION' if row['collision']=='True' else 'TIMEOUT'
            files={}
            for mode in ('disabled','enabled'):
                relative=f'{suite}/{method}/seed{seed}/projection_{mode}_trajectories.npz'
                if source == 'published':
                    require(relative in manifest['archives'], f'Published archive is not listed: {relative}')
                    files[mode]={**manifest['archives'][relative], 'path':(BUNDLE / relative).as_posix()}
                else:
                    files[mode]={'path':'runs/evaluation/final/'+relative}
            records.append({'suite':suite,'method':method,'train_seed':seed,'episode':episode,
                            'layout_id':off['layout_id'],'evaluation_seed':int(off['evaluation_seed']),
                            'execution':off['evaluation_policy_mode'],'off':terminal(off),'on':terminal(on),
                            'rows':modes,'data_source':source,
                            'dataset_id':manifest['dataset_id'] if manifest else None,
                            'source_table':table.relative_to(root).as_posix(),
                            'source_table_sha256':table_digest,'archive_files':files,
                            'available':all((root/files[mode]['path']).is_file() for mode in ('disabled','enabled'))})
    if manifest:
        require(set(archive_counts) == set(manifest['archives']),
                'Published archive inventory differs from the evaluation tables')
        for relative,count in archive_counts.items():
            require(count == manifest['archives'][relative]['episodes'],
                    f'Published episode count mismatch: {relative}')
    require(records,'No paired results found. The repository must contain results/tables and published or local evaluation archives.')
    return records


def load_pair(root,suite,method,train_seed,episode,record=None,source='auto'):
    root=Path(root).resolve()
    if record is None:
        matches=[r for r in catalog(root,source=source) if (r['suite'],r['method'],r['train_seed'],r['episode'])==(suite,method,int(train_seed),int(episode))]
        require(len(matches)==1,'Selected pair is missing or ambiguous')
        record=matches[0]
    require(source in SOURCES and (source == 'auto' or source == record['data_source']),
            'Selected pair belongs to a different data source')
    require((record['suite'],record['method'],record['train_seed'],record['episode']) ==
            (suite,method,int(train_seed),int(episode)), 'Selected pair identity differs')
    pair=[]
    for mode in ('disabled','enabled'):
        item=record['archive_files'][mode]
        path=root/item['path']
        require(path.is_file(), f"{SOURCE_LABELS[record['data_source']]} archive is missing: {item['path']}")
        contents=path.read_bytes()
        archive_digest=hashlib.sha256(contents).hexdigest()
        if record['data_source'] == 'published':
            require(len(contents)==item['size_bytes'] and archive_digest==item['sha256'],
                    f"Published archive hash/size mismatch: {item['path']}")
        prefix=f'episode_{episode:04d}_'
        with np.load(io.BytesIO(contents),allow_pickle=False) as archive:
            require(str(archive['trajectory_archive_version'])=='evaluation_trajectory_v1','Unsupported trajectory schema')
            d={k[len(prefix):]:archive[k].copy() for k in archive.files if k.startswith(prefix)}
            require(bool(d),f'Episode {episode} missing from {path}')
            d['checkpoint_sha256']=str(archive['run_checkpoint_sha256'])
            d['goal_radius']=float(archive['run_goal_radius'])
            d['stochastic']=bool(archive['run_stochastic'])
        d.update(mode=mode,row=record['rows'][mode],source_archive=path.relative_to(root).as_posix(),
                 source_sha256=archive_digest,data_source=record['data_source'],
                 dataset_id=record['dataset_id'],source_table=record['source_table'],
                 source_table_sha256=record['source_table_sha256'])
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
