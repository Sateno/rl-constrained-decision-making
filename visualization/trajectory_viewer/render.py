"""Replay archived PAP trajectories as a 3D rendering of a planar experiment.

No policy inference, physics rollout, or QP solve occurs in this script.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import subprocess
import shutil

from data import load_pair as read_pair, outcome

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'trajectory_exports'
METHOD = 'ppo_train_projection'
TRAIN_SEED = 1
BG = '#f5f7fa'
INK = '#182b42'
MUTED = '#54677c'
OFF = '#cb5957'
ON = '#008b83'
GOLD = '#ebad32'
OBSTACLE = '#8392a8'
FPS = 20
PLAYBACK_SPEED = 1.0
CASES = [
    ('fixed_training_geometry', 2, 'Fixed geometry', 'Stochastic evaluation', 'Collision → success'),
    ('core_layout_transfer', 6, 'Transfer geometry', 'Actor-mean evaluation', 'Collision → success'),
    ('core_layout_transfer', 8, 'Transfer geometry', 'Actor-mean evaluation', 'Collision → timeout'),
]


def load_pair(suite, episode):
    return read_pair(ROOT, suite, METHOD, TRAIN_SEED, episode)


def cylinder_faces(xy, radius, height, z0=0.0, segments=32):
    theta=np.linspace(0,2*np.pi,segments,endpoint=False)
    ring=np.column_stack((xy[0]+radius*np.cos(theta),xy[1]+radius*np.sin(theta)))
    bottom=np.column_stack((ring,np.full(segments,z0)))
    top=np.column_stack((ring,np.full(segments,z0+height)))
    faces=[top]
    faces += [np.array([bottom[i],bottom[(i+1)%segments],top[(i+1)%segments],top[i]]) for i in range(segments)]
    return faces


def disk(ax, xy, radius, color, z=.015, alpha=.2):
    theta=np.linspace(0,2*np.pi,100)
    ring=np.column_stack((xy[0]+radius*np.cos(theta),xy[1]+radius*np.sin(theta),np.full(100,z)))
    p=Poly3DCollection([ring],facecolor=color,edgecolor=color,alpha=alpha,linewidth=.7)
    ax.add_collection3d(p)


def prepare_scene(index, pair, case=None):
    suite, episode, label, execution, comparison = case or CASES[index]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK})
    fig=plt.figure(figsize=(14,8),dpi=100,facecolor=BG)
    fig.text(.04,.952,'PAP  /  RECORDED TRAJECTORIES',size=13,weight='bold',color=ON)
    fig.text(.04,.900,comparison,size=26,weight='bold')
    fig.text(.96,.906,(f'{index+1:02d} / {len(CASES):02d}' if case is None else 'PAIR REPLAY'),ha='right',size=17,color=MUTED)
    seed=int(pair[0]['seed'])
    fig.text(.04,.858,f'{label}  ·  {execution}  ·  Evaluation seed {seed}',size=13,color=MUTED)
    fig.text(.04,.046,'3D view of recorded 2D motion. Heights are illustrative. Playback: 1× simulation time.',size=10,color=MUTED)
    fig.text(.04,.022,f"Selected example. {pair[0]['row']['method']}, training seed {pair[0]['row']['train_seed']}; same checkpoint and initial state within the pair.",size=9.2,color=MUTED)
    fig.text(.96,.073,'●  Gold rim: projector intervened on the latest transition',ha='right',size=10.5,color='#996c12')
    # Fixed, identical camera and limits within each matched pair.
    points=np.concatenate([d['positions'] for d in pair]+[pair[0]['goal'][None,:]])
    for center,radius in zip(pair[0]['obstacle_centers'][pair[0]['obstacle_mask']],pair[0]['obstacle_radii'][pair[0]['obstacle_mask']]):
        points=np.concatenate([points,center[None,:]+[radius,radius],center[None,:]-[radius,radius]])
    lower=points.min(0)-.55
    upper=points.max(0)+.55
    ym=(lower[1]+upper[1])/2
    yr=max(upper[1]-lower[1],3.3)
    lower[1],upper[1]=ym-yr/2,ym+yr/2
    artists=[]
    for col,d in enumerate(pair):
        left=.03+.49*col
        color=OFF if col==0 else ON
        card=FancyBboxPatch((left,.105),.45,.70,boxstyle='round,pad=0.007,rounding_size=0.015',
                           transform=fig.transFigure,facecolor='white',edgecolor='#dce3eb',linewidth=1,zorder=-2)
        fig.patches.append(card)
        fig.text(left+.02,.766,'PROJECTION OFF' if col==0 else 'PROJECTION ON',size=14,weight='bold',color=color)
        fig.text(left+.02,.734,'Actor proposal executed' if col==0 else 'Actor proposal + QP correction',size=11,color=MUTED)
        ax=fig.add_axes([left-.01,.222,.48,.49],projection='3d',facecolor='white',computed_zorder=False)
        ax.view_init(elev=53,azim=-76)
        ax.set_proj_type('ortho')
        ax.set_xlim(lower[0],upper[0]);ax.set_ylim(lower[1],upper[1]);ax.set_zlim(0,1.0)
        ax.set_box_aspect((upper[0]-lower[0],upper[1]-lower[1],1.0),zoom=1.2)
        ax.set_axis_off()
        floor=np.array([[lower[0],lower[1],-.01],[upper[0],lower[1],-.01],[upper[0],upper[1],-.01],[lower[0],upper[1],-.01]])
        ax.add_collection3d(Poly3DCollection([floor],facecolor='#f1f4f8',edgecolor='#d9e0e9',linewidth=.8,zorder=0))
        for x in np.arange(np.ceil(lower[0]*2)/2,upper[0],.5):
            ax.plot([x,x],[lower[1],upper[1]],[0,0],color='#dce3ec',lw=.65,zorder=1)
        for y in np.arange(np.ceil(lower[1]*2)/2,upper[1],.5):
            ax.plot([lower[0],upper[0]],[y,y],[0,0],color='#dce3ec',lw=.65,zorder=1)
        solids=[]
        for c,r in zip(d['obstacle_centers'][d['obstacle_mask']],d['obstacle_radii'][d['obstacle_mask']]):
            mesh=Poly3DCollection(cylinder_faces(c,float(r),.38),facecolor=OBSTACLE,edgecolor='#6d7e96',alpha=.87,linewidth=.2,zorder=3)
            ax.add_collection3d(mesh)
            theta=np.linspace(0,2*np.pi,100)
            top,=ax.plot(c[0]+float(r)*np.cos(theta),c[1]+float(r)*np.sin(theta),np.full(100,.39),color='#61758e',lw=.9,zorder=4)
            solids.append((c,mesh,top))
        disk(ax,d['goal'],d['goal_radius'],'#27a773',alpha=.17)
        ax.text(*d['goal'],.065,'GOAL',ha='center',va='center',size=9,color='#147f57',weight='bold',zorder=5)
        start=d['positions'][0]
        ax.text(start[0]-.06,start[1]-.30,.015,'START',ha='center',size=8.5,color=MUTED,zorder=5)
        trail,=ax.plot([],[],[],color=color,lw=2.3,zorder=2)
        body=Poly3DCollection(cylinder_faces(start,float(d['agent_radius']),.16,z0=.025,segments=24),
                             facecolor=color,edgecolor=color,linewidth=.3,zorder=7)
        ax.add_collection3d(body)
        heading,=ax.plot([],[],[],color='white',lw=2.5,zorder=9)
        rim,=ax.plot([],[],[],color=GOLD,lw=3,zorder=8)
        status=fig.text(left+.02,.205,'READY',size=15,weight='bold',color=color)
        timing=fig.text(left+.02,.169,'',size=11,color=MUTED)
        diagnostic=fig.text(left+.02,.132,'',size=10.5,color=MUTED)
        artists.append((d,ax,trail,body,heading,rim,status,timing,diagnostic,color,solids))

    def update(sim_time):
        for d,ax,trail,body,heading,rim,status,timing,diagnostic,color,solids in artists:
            n=len(d['positions'])-1
            sample=min(max(sim_time/float(d['dt']),0),n)
            k=int(np.floor(sample+1e-8));fraction=sample-k
            if k>=n:k=n;fraction=0.
            pos=d['positions'][k].copy();angle=float(d['headings'][k])
            if k<n:
                pos+=(d['positions'][k+1]-pos)*fraction
                delta=np.arctan2(np.sin(d['headings'][k+1]-angle),np.cos(d['headings'][k+1]-angle))
                angle+=fraction*delta
            xy=np.vstack([d['positions'][:k+1],pos])
            trail.set_data_3d(xy[:,0],xy[:,1],np.full(len(xy),.035))
            body.set_verts(cylinder_faces(pos,float(d['agent_radius']),.16,z0=.025,segments=24))
            # The floor always remains behind the scene; order the solid objects
            # by camera depth to avoid mplot3d's floor-plane occlusion artifact.
            objects=[(center,mesh,top) for center,mesh,top in solids]+[(pos,body,heading)]
            projection=ax.get_proj()
            objects.sort(key=lambda o:proj3d.proj_transform(o[0][0],o[0][1],.12,projection)[2],reverse=True)
            for rank,(_,mesh,top) in enumerate(objects):
                mesh.set_zorder(3+rank);top.set_zorder(3+rank+.05)
            rim.set_zorder(body.get_zorder()+.03)
            radius=float(d['agent_radius'])
            heading.set_data_3d([pos[0],pos[0]+.92*radius*np.cos(angle)],
                               [pos[1],pos[1]+.92*radius*np.sin(angle)],[.19,.19])
            transition=k-1
            intervened=transition>=0 and bool(d['projection_intervened'][transition])
            if intervened:
                theta=np.linspace(0,2*np.pi,50)
                rim.set_data_3d(pos[0]+radius*1.12*np.cos(theta),pos[1]+radius*1.12*np.sin(theta),np.full(50,.20))
            else:rim.set_data_3d([],[],[])
            ended=sample>=n
            status.set_text(outcome(d) if ended else ('READY' if sim_time<=0 else 'RUNNING'))
            status.set_color('#17865e' if ended and outcome(d)=='SUCCESS' else '#ae7320' if ended and outcome(d)=='TIMEOUT' else color)
            timing.set_text(f't = {min(sim_time,n*float(d["dt"])):4.1f} s    |    Step {k:3d} / {n}')
            if d['mode']=='enabled':
                slack=float(d['projection_slack_max'][transition]) if transition>=0 else 0.
                state='YES' if intervened else 'NO'
                diagnostic.set_text(f'Latest transition: intervention {state}    |    max slack {slack:.2e}')
            else:
                diagnostic.set_text('Episode ended; final pose held' if ended else 'Grid spacing: 0.5 m')
        return fig
    return fig,update


def main():
    global ROOT, OUT, METHOD, TRAIN_SEED, CASES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--method',default=METHOD)
    parser.add_argument('--train-seed',type=int,default=1)
    parser.add_argument('--suite',choices=['fixed_training_geometry','core_layout_transfer'])
    parser.add_argument('--episode',type=int,help='Zero-based episode index; requires --suite')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args()
    if (args.suite is None)!=(args.episode is None):
        parser.error('--suite and --episode must be supplied together')
    ROOT=args.repo_root.resolve();OUT=(args.output_dir or ROOT/'trajectory_exports').resolve()
    METHOD=args.method;TRAIN_SEED=args.train_seed
    if args.suite:
        p=load_pair(args.suite,args.episode)
        CASES=[(args.suite,args.episode,args.suite.replace('_',' ').title(),
                'Stochastic evaluation' if p[0]['stochastic'] else 'Actor-mean evaluation',
                f'{outcome(p[0]).title()} → {outcome(p[1]).title()}')]
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.preview and not shutil.which('ffmpeg'):
        parser.error('FFmpeg is not on PATH. Install FFmpeg or use --preview.')
    pairs=[load_pair(c[0],c[1]) for c in CASES]
    CASES=[(c[0],c[1],c[2],
            'Stochastic evaluation' if pair[0]['stochastic'] else 'Actor-mean evaluation',
            f'{outcome(pair[0]).title()} → {outcome(pair[1]).title()}')
           for c,pair in zip(CASES,pairs)]
    manifest=[]
    for i,pair in enumerate(pairs):
        for d in pair:
            manifest.append({k:d[k] for k in ('mode','checkpoint_sha256','source_sha256','source_archive')} |
                            {'suite':CASES[i][0],'episode':CASES[i][1],'layout_id':d['row']['layout_id'],
                             'evaluation_seed':int(d['seed']),'outcome':outcome(d),'transitions':len(d['positions'])-1})
    (OUT/'selected_episodes.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.preview:
        for i,pair in enumerate(pairs):
            fig,update=prepare_scene(i,pair)
            update(max((len(d['positions'])-1)*float(d['dt']) for d in pair))
            fig.savefig(OUT/f'preview_{i+1}.png',dpi=100,facecolor=BG)
            plt.close(fig)
        print(f'{len(pairs)} previews written; saved-array alignment and pairing checks passed.',flush=True)
        return
    video=OUT/'PAP_Recorded_Trajectories_3D.mp4'
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgba','-s','1400x800','-r',str(FPS),
         '-i','-','-an','-vcodec','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',
         '-metadata','title=PAP: recorded OFF/ON trajectories in 3D',
         '-metadata','comment=Replay of saved 2D evaluation data; 3D heights decorative. See selected_episodes.json for exact provenance. Linear visual interpolation; no resimulation.',str(video)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i,pair in enumerate(pairs):
            fig,update=prepare_scene(i,pair)
            duration=max((len(d['positions'])-1)*float(d['dt']) for d in pair)
            times=np.concatenate([np.zeros(FPS),np.arange(1,round(duration*FPS/PLAYBACK_SPEED)+1)*PLAYBACK_SPEED/FPS,
                                  np.full(FPS*2,duration)])
            for time in times:
                update(float(time));fig.canvas.draw();process.stdin.write(fig.canvas.buffer_rgba())
            plt.close(fig)
            print(f'Rendered pair {i+1}/{len(pairs)} ({len(times)} frames).',flush=True)
    finally:
        process.stdin.close()
    if process.wait()!=0:raise RuntimeError('ffmpeg failed')
    print(video,video.stat().st_size,flush=True)


if __name__=='__main__':main()

