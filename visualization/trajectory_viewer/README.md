# PAP trajectory explorer

An offline, browser-based viewer for matched Projection OFF / ON recordings. The interface uses dark navy panels, coral and teal trajectories, and synchronized playback. It reads saved arrays; it never trains a policy, runs an environment, or solves a QP.

## Install this overlay

Copy the ZIP's `visualization/` folder into your repository root and allow the included files to replace their counterparts. Keep the other files already in `visualization/trajectory_viewer/`, including `data.py`, `render.py`, and `requirements.txt`. No files need to be deleted.

This overlay contains four files:

- `visualization/trajectory_viewer/viewer.py` — local launcher and recording access.
- `visualization/trajectory_viewer/viewer.html` — the new browser interface.
- `visualization/trajectory_viewer/README.md` — these instructions.
- `visualization/trajectory_viewer/tests/test_viewer.py` — local-server checks using synthetic data.

No new Python dependency is required. From your repository root, with your project environment active:

```powershell
python -m pip install -r visualization/trajectory_viewer/requirements.txt
python visualization/trajectory_viewer/viewer.py
```

Your default browser opens automatically. Keep the terminal open while using the viewer; press **Ctrl+C** there to stop it. If the browser does not open, copy the `http://127.0.0.1:...` address printed in the terminal into your browser. Closing a browser tab does not stop the Python process.

The viewer finds the repository from its installed location under `visualization/trajectory_viewer/`. There is no repository chooser. The Windows `launch_viewer.bat` also continues to work from an activated environment.

No account, internet connection, CDN, or web framework is needed. All browser assets are included in `viewer.html`; the Python server listens only on this computer. Tkinter is no longer needed. Use a current Chrome, Edge, Firefox, or Safari browser.

For another checkout or manual browser opening:

```powershell
python visualization/trajectory_viewer/viewer.py --repo-root PATH --no-browser
```

`--port 8765` optionally chooses a fixed local port. By default an available port is selected automatically.

## Browse and play

Choose an evaluation suite, training method, training seed, and OFF → ON outcome. The trajectory selector lists all matching pairs whose two trajectory archives are present. Selecting a pair loads it immediately at the first frame. There is no separate Load button or layout/evaluation-seed search field.

The initial example, when available, is projection-trained PPO, training seed 1, transfer episode 6 (collision → success). This is an illustrative selection by outcome, not a performance summary. Filters retain the current pair when it still matches; otherwise the first matching pair loads. An empty selection clears the scene and disables playback and export.

- **Play / Pause**, **Restart**, and the timeline control both sides together.
- **Speed** changes playback speed, not recorded simulation time.
- **3D / Top-down** and **Angle** change the shared camera. The underlying motion is always planar.
- The ON robot's amber ring marks intervention on the latest completed transition.
- The card footer shows each side's own time, completed step, status, and diagnostic.
- Each side holds its recorded final pose while the other episode continues.

Dropdowns and sliders support normal keyboard interaction. Space toggles playback when focus is outside an input control. Resize the browser or use its usual zoom controls to change the interface size.

## Export a selected pair

**Export MP4** renders the selected pair and downloads a ZIP through your browser. Your browser's download preferences determine whether it goes to **Downloads** or opens a **Save As** dialog. Enable the browser's “Ask where to save each file before downloading” setting if you want to choose a folder each time.

Export uses the existing Matplotlib video renderer's standard 3D camera, original light video theme, and 1× simulation time; it does not capture the current browser view or playback speed. The interactive viewer and video renderer share the validated archive loader.

Video export requires FFmpeg on PATH. Playback does not. For a Conda environment:

```powershell
conda install -c conda-forge ffmpeg
```

The downloaded ZIP contains:

- `PAP_Recorded_Trajectories_3D.mp4`
- `selected_episodes.json`, recording archive paths and SHA-256 hashes, checkpoint hash, suite, episode, evaluation seed, layout and outcomes.
- `export.log`, containing the renderer's diagnostics.

When the ZIP is ready, the interface requests its download automatically and leaves a **Download ZIP** link available if the browser blocks or postpones the automatic download. “Download requested” does not confirm that the browser has saved the file; check the browser's downloads list. If rendering fails, **Download export log** provides the available diagnostic log.

Browser exports are rendered in the operating system's temporary directory, never in the repository. Those temporary files are removed when the viewer exits. Keep the viewer's terminal running until the download finishes. The ZIP you download is retained in the destination chosen by your browser. Changing the selected pair during an export does not change the export's selection. Only one export runs at a time.

The separate command-line renderer remains available. These PowerShell examples explicitly save outside the repository:

```powershell
python visualization/trajectory_viewer/render.py --preview --output-dir "$HOME/Downloads/PAP_previews"
python visualization/trajectory_viewer/render.py --output-dir "$HOME/Downloads/PAP_examples"
python visualization/trajectory_viewer/render.py --suite core_layout_transfer --method ppo_baseline --train-seed 2 --episode 8 --output-dir "$HOME/Downloads/PAP_baseline_seed2_episode8"
```

Its default episode selection reproduces the three original examples: projection-trained PPO, seed 1; fixed episode 2 and transfer episodes 6 and 8. Episode numbers are zero-based. This standalone script is unchanged: it writes directly to `--output-dir`, does not create a download ZIP, and overwrites existing output files in that directory. If `--output-dir` is omitted, it still defaults to the repository's `trajectory_exports/` folder. Always supply an external output directory to keep command-line renders outside the repository; browser exports already do this automatically.

## Data and display contract

- Reads `results/tables/<suite>/evaluation_episode_results.csv` and `runs/evaluation/final/<suite>/<method>/seed<N>/projection_{disabled,enabled}_trajectories.npz`.
- Validates archive schema, checkpoint identity, evaluation seed, initial pose, geometry, state/transition alignment, policy mode, projection mode and terminal outcomes against the CSV.
- Each controller follows its own recorded observations and actions. Pairing does not imply that the raw action sequences remain identical after the paths diverge.
- Positions are linearly interpolated for display; headings follow the shortest angular arc. Interpolated frames are not newly simulated states.
- The robot footprint and obstacle radii use recorded geometry. Cylinder heights are decorative; the experiment is planar.
- Both panels use identical bounds and camera settings, fixed for the selected pair.
- Intervention and maximum slack refer to the latest completed transition, not the next transition being interpolated. Slack has barrier-residual units, not metres or collision probability.
- Recording files are read-only. Restart the viewer after changing source evaluation files, since the catalog and up to four loaded pairs are cached for the session.

This add-on contains no paper sources, evaluation data copies, checkpoints, or generated videos.

## Code checks

Run from the repository root:

```powershell
python -m unittest discover -s visualization/trajectory_viewer/tests -v
python -m compileall -q visualization/trajectory_viewer
```

The tests use small synthetic archives and do not rerun evaluation. Browser checks for this release used the supplied recordings, including both evaluation suites and the collision → success and collision → timeout examples. The interface was checked in headless Chromium at desktop and narrow widths; native Windows browser appearance was not directly tested here.
