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


if __name__=='__main__':unittest.main()
