"""Small synthetic archives test pairing guards, not scientific outcomes."""
from pathlib import Path
import csv
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import catalog, load_pair


class PairValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.suite='example_suite';self.method='example_method'
        self.folder=self.root/'runs/evaluation/final'/self.suite/self.method/'seed1';self.folder.mkdir(parents=True)
        rows=[]
        for mode in ('disabled','enabled'):
            values={'trajectory_archive_version':np.asarray('evaluation_trajectory_v1'),
                    'run_checkpoint_sha256':np.asarray('checkpoint_hash'),'run_goal_radius':np.asarray(.25),
                    'run_stochastic':np.asarray(False)}
            fields={'positions':np.array([[0.,0.],[.1,0.],[.2,0.]]),'headings':np.zeros(3),
                    'state_success':np.array([False,False,True]),'state_collision':np.zeros(3,dtype=bool),
                    'action_exec_physical':np.zeros((2,2)),'projection_intervened':np.zeros(2,dtype=bool),
                    'projection_slack_max':np.zeros(2),'projection_enabled':np.full(2,mode=='enabled'),
                    'truncated':np.zeros(2,dtype=bool),'seed':np.asarray(1000),'dt':np.asarray(.1),
                    'goal':np.array([.2,0.]),'obstacle_centers':np.zeros((1,2)),
                    'obstacle_radii':np.array([.3]),'obstacle_mask':np.array([False]),'agent_radius':np.asarray(.1)}
            values.update({'episode_0000_'+k:v for k,v in fields.items()})
            np.savez_compressed(self.folder/f'projection_{mode}_trajectories.npz',**values)
            rows.append({'method':self.method,'train_seed':'1','episode':'0','projection_mode':mode,
                         'checkpoint_sha256':'checkpoint_hash','evaluation_seed':'1000','layout_id':'example',
                         'evaluation_policy_mode':'deterministic','success':'True','collision':'False','episode_length':'2'})
        table=self.root/'results/tables'/self.suite/'evaluation_episode_results.csv';table.parent.mkdir(parents=True)
        with table.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def load(self):return load_pair(self.root,self.suite,self.method,1,0)

    def change(self,key,value):
        path=self.folder/'projection_enabled_trajectories.npz'
        with np.load(path,allow_pickle=False) as a:values={k:a[k].copy() for k in a.files}
        values[key]=value;np.savez_compressed(path,**values)

    def test_valid_pair(self):
        self.assertEqual(len(self.load()),2)
        self.assertTrue(catalog(self.root)[0]['available'])

    def test_changed_initial_pose_rejected(self):
        self.change('episode_0000_positions',np.array([[.01,0],[.1,0],[.2,0]]))
        with self.assertRaisesRegex(ValueError,'Initial positions'):self.load()

    def test_changed_checkpoint_rejected(self):
        self.change('run_checkpoint_sha256',np.asarray('different'))
        with self.assertRaisesRegex(ValueError,'Checkpoint hash'):self.load()

    def test_wrong_alignment_rejected(self):
        self.change('episode_0000_headings',np.zeros(2))
        with self.assertRaisesRegex(ValueError,'Misaligned headings'):self.load()

    def test_csv_outcome_disagreement_rejected(self):
        self.change('episode_0000_state_success',np.zeros(3,dtype=bool))
        with self.assertRaisesRegex(ValueError,'Success mismatch'):self.load()

    def test_wrong_mode_rejected(self):
        self.change('episode_0000_projection_enabled',np.zeros(2,dtype=bool))
        with self.assertRaisesRegex(ValueError,'Projection mode mismatch'):self.load()

    def test_missing_archive_is_not_available(self):
        (self.folder/'projection_enabled_trajectories.npz').unlink()
        self.assertFalse(catalog(self.root)[0]['available'])


def publish_fixture(fixture):
    """Package a tiny fixture without touching the scientific dataset."""
    import hashlib
    import json
    import shutil
    from data import BUNDLE
    destination = fixture.root / BUNDLE
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 'pap_trajectory_bundle_v1',
                'dataset_id': 'pap_evaluation_v1',
                'trajectory_archive_version': 'evaluation_trajectory_v1',
                'tables': {}, 'archives': {}}
    relative_table = f'results/tables/{fixture.suite}/evaluation_episode_results.csv'
    table = fixture.root / relative_table
    manifest['tables'][fixture.suite] = {'path': relative_table,
        'sha256': hashlib.sha256(table.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    for path in fixture.folder.glob('*.npz'):
        relative = f'{fixture.suite}/{fixture.method}/seed1/{path.name}'
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        manifest['archives'][relative] = {
            'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'size_bytes': output.stat().st_size, 'episodes': 1,
            'source_archive': 'runs/evaluation/final/' + relative}
    (destination / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return destination


class PublishedBundleTests(unittest.TestCase):
    def setUp(self):
        self.fixture = PairValidationTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.bundle = publish_fixture(self.fixture)

    def load(self, source='auto'):
        return load_pair(self.root, self.fixture.suite, self.fixture.method, 1, 0, source=source)

    def test_fresh_checkout_replays_without_runs(self):
        import shutil
        shutil.rmtree(self.root / 'runs')
        records = catalog(self.root)
        self.assertTrue(records[0]['available'])
        self.assertEqual(records[0]['data_source'], 'published')
        pair = self.load()
        self.assertEqual(pair[0]['dataset_id'], 'pap_evaluation_v1')
        self.assertTrue(pair[0]['source_archive'].startswith('results/trajectories/'))
        self.assertEqual(pair[0]['source_table_sha256'], records[0]['source_table_sha256'])

    def test_published_precedence_and_explicit_local_selection(self):
        self.fixture.change('run_checkpoint_sha256', np.asarray('changed-local-file'))
        self.assertEqual(self.load()[1]['checkpoint_sha256'], 'checkpoint_hash')
        with self.assertRaisesRegex(ValueError, 'Checkpoint hash'):
            self.load(source='local')

    def test_missing_published_mode_never_uses_local_copy(self):
        next(self.bundle.rglob('projection_enabled_trajectories.npz')).unlink()
        self.assertFalse(catalog(self.root)[0]['available'])
        self.assertTrue(catalog(self.root, source='local')[0]['available'])
        with self.assertRaisesRegex(ValueError, 'archive is missing'):
            self.load()

    def test_corrupt_published_bytes_never_fall_back(self):
        path = next(self.bundle.rglob('projection_enabled_trajectories.npz'))
        path.write_bytes(path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'hash/size mismatch'):
            self.load()
        self.assertEqual(self.load(source='local')[1]['data_source'], 'local')

    def test_changed_table_rejected(self):
        path = self.root / 'results/tables' / self.fixture.suite / 'evaluation_episode_results.csv'
        path.write_bytes(path.read_bytes().replace(b'example', b'changed'))
        with self.assertRaisesRegex(ValueError, 'table hash mismatch'):
            catalog(self.root)

    def test_windows_table_newlines_are_accepted(self):
        path = self.root / 'results/tables' / self.fixture.suite / 'evaluation_episode_results.csv'
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'))
        self.assertEqual(len(self.load()), 2)

    def test_bad_manifest_path_and_episode_count_are_rejected(self):
        import json
        path = self.bundle / 'manifest.json'
        original = json.loads(path.read_text())
        for corruption in ('path', 'count'):
            with self.subTest(corruption=corruption):
                manifest = json.loads(json.dumps(original))
                key = next(iter(manifest['archives']))
                if corruption == 'path':
                    manifest['archives']['../outside.npz'] = manifest['archives'].pop(key)
                else:
                    manifest['archives'][key]['episodes'] = 2
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    catalog(self.root)

    def test_extra_manifest_path_cannot_override_bundle_path(self):
        import json
        path = self.bundle / 'manifest.json'
        manifest = json.loads(path.read_text())
        for item in manifest['archives'].values():
            item['path'] = 'outside/redirected.npz'
        path.write_text(json.dumps(manifest))
        pair = self.load()
        self.assertTrue(pair[0]['source_archive'].startswith('results/trajectories/'))

    def test_explicit_published_requires_manifest(self):
        (self.bundle / 'manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'manifest is missing'):
            catalog(self.root, source='published')
        self.assertEqual(catalog(self.root)[0]['data_source'], 'local')


if __name__=='__main__':unittest.main()
