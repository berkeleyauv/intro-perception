# Step-by-step: finishing and submitting the project

Everything except training YOLO is already done on this branch. These steps
take about 30–45 minutes, most of it waiting for Colab.

You need: the two mentor zips (`images.zip`, `gate_images_labeled.zip`), the
branch bundle `perception-intro.bundle` (or the pushed branch), `uv` installed,
and a Google account for Colab.

---

## 1. Get the code (5 min)

```bash
git clone --recurse-submodules https://github.com/berkeleyauv/intro-perception.git
cd intro-perception
git pull                                   # make sure main has the mentors' latest changes
git fetch ~/Downloads/perception-intro.bundle \
    satyajith/perception-intro:satyajith/perception-intro
git switch satyajith/perception-intro
./scripts/setup.sh
source .venv/bin/activate
./scripts/test.sh                          # expect: 103 passed
```

`git log --oneline main..` should list 7 commits.

## 2. Unzip the data outside the repo (1 min)

```bash
mkdir -p ~/gate_data && cd ~/gate_data
unzip -q ~/Downloads/images.zip                 # -> ~/gate_data/images/ (1,833 .jpg)
unzip -q ~/Downloads/gate_images_labeled.zip    # -> ~/gate_data/intro_detect/ (labels + manifest.csv)
cd -                                            # back to intro-perception
```

Never copy these images into the repo's tracked files (`data/` is ignored by
Git, which is fine).

## 3. Run step 1: split + classical + baseline (1–2 min)

```bash
./scripts/full_pipeline.sh ~/gate_data/images ~/gate_data/intro_detect
```

Check the printed numbers match the report: test split 183 images (146 gate,
37 no-gate, 106 cut-off); classical precision ≈ 0.83, recall ≈ 0.73. FPS will
differ on your laptop. This also creates **`output/generated.zip`**.

## 4. Train YOLO in Colab (15–25 min)

1. Open <https://colab.research.google.com> → *File → Upload notebook* →
   choose `notebooks/train_colab.ipynb` from the repo.
2. *Runtime → Change runtime type → T4 GPU → Save*.
3. *Runtime → Run all*. When the **Choose Files** button appears in the third
   cell, pick `output/generated.zip` (110 MB; the upload takes a minute or two).
4. Wait for training (30 epochs) and the last cell. Your browser downloads
   **`yolo_results.zip`**. If it doesn't, open the folder icon on the left and
   download `/content/yolo_results.zip` by hand.

Do not change the epochs, image size or model: the project requires
yolo26n.pt, 30 epochs, 640 px.

## 5. Run step 2: score YOLO and build the comparison (1 min)

```bash
./scripts/finish_yolo.sh ~/Downloads/yolo_results.zip
```

This fills the YOLO columns of the table in `REPORT.md`, writes
`output/side_by_side/` (classical | YOLO for every test image) and
`output/failures/yolo/`.

## 6. Finish the report (10 min, or send the results to Claude)

Open `REPORT.md` and replace the three _(after YOLO)_ parts:

- **YOLO**: training time, final val mAP50 (last row of
  `artifacts/yolo/colab/results.csv`).
- **Discussion**: which method is more accurate, how much faster classical is
  than YOLO on CPU (`yolo_cpu` column), whether YOLO's recall on cut-off gates
  is better or worse than on whole gates, and which you would deploy.
- **Contributions**: who did what.

Or zip `output/metrics_*.json`, `output/failures_both.txt` and a few
images from `output/failures/yolo/`, send them to Claude, and paste back the
finished sections.

Then commit:

```bash
git add REPORT.md
git commit -m "Add YOLO results and the comparison discussion"
```

## 7. Open the draft PR (5 min)

```bash
git push -u origin satyajith/perception-intro
```

On GitHub: *Compare & pull request* → base `main` → **Create draft pull
request**. In the description paste the report's summary and the comparison
table, then drag in these images (they are dataset images, so they stay out of
the repo):

- `output/report_images/*.jpg` (baseline vs classical successes and failures)
- 3–4 images from `output/side_by_side/` (classical vs YOLO)

Do **not** commit `artifacts/`, `output/`, `data/` or `best.pt` (all ignored).

---

## If something goes wrong

| problem | fix |
|---|---|
| `uv: command not found` | install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh`, open a new terminal |
| `data/generated already exists; reusing it` | fine; delete `data/generated` to re-split from scratch |
| Colab: `CUDA: False` | the runtime is not a GPU; redo step 4.2 |
| Colab upload very slow | upload `generated.zip` to Google Drive and replace the upload cell with `from google.colab import drive; drive.mount('/content/drive')` and `!cp /content/drive/MyDrive/generated.zip .` |
| `finish_yolo.sh`: no prediction for N images | the Colab run used a different split; use the `generated.zip` from step 3 |
| tests fail after `git pull` on main | rebase: `git rebase main`, then `./scripts/test.sh` |
