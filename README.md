# Shot Boundary Detection & Scene Grouping
> A Computer Vision project comparing classical dissimilarity signals, SIFT/RANSAC geometric verification and keyframe similarity to detect cuts and group recurring scenes in edited video.

## The Project
An edited video is a sequence of shots separated by cuts, and editors often return to a scene shown earlier. This project tackles three connected problems: (1) detecting the cuts, (2) avoiding false cuts caused by camera flashes and fast camera motion, and (3) grouping shots that belong to the same recurring scene.

Developed for the Computer Vision course (2026–2027) at **NOVA School of Science and Technology (FCT NOVA)**, by Pedro Pequeno and Rodrigo Sinde.

## Methodology
The system is built in three stages, each evaluated quantitatively against ground-truth annotations:

1. **Cut candidates (frame dissimilarity):** three signals are computed per frame — mean pixel difference, an HSV colour-histogram χ² distance, and the Edge Change Ratio (ECR). The colour histogram (brightness channel dropped) proved the most robust to flashes and camera motion and was combined with non-maximum suppression to produce cut candidates.
2. **False-alarm verification (SIFT + RANSAC):** every candidate cut is checked by comparing the frames a few steps before and after it. SIFT keypoints are matched with Lowe's ratio test, and RANSAC counts how many matches agree with a single homography. If the two frames still show the same content (many inliers), the candidate is rejected as a flash or motion artifact rather than a real cut.
3. **Scene grouping (recurring scenes):** each shot is represented by several evenly-spaced keyframes. Keyframes are compared by colour similarity, the best-matching pair sets the similarity between two shots, and shots are linked into scenes via connected components above a learned threshold.

All parameters were tuned exclusively on two development videos; two held-out test videos were evaluated only once, after every design choice was frozen.

## Results

| | Dev set | Held-out test set |
|---|---|---|
| Cut detection (F1) | 0.979 | 0.885 |
| Scene grouping (F1) | 0.984 | 0.884 |

Cut recall on the test set stayed near-perfect (0.987); most of the precision loss came from a single scene (fireworks) whose fast colour changes mimic a cut. Scene-grouping precision was 1.0 on every video — the method never merged two different scenes, only occasionally missed linking shots of the same one. Full failure-case analysis (with figures) is in the report notebook.

## Tech Stack
* **Language:** Python 3
* **Computer Vision:** OpenCV (SIFT, RANSAC homography estimation, Canny edges), NumPy
* **Analysis & Reporting:** Jupyter Notebook, Matplotlib

## How to Run (Reproducibility)
1. Clone the repo: `git clone https://github.com/Pequeno00/shot-boundary-detection.git`
2. Install dependencies: `pip install opencv-python numpy matplotlib jupyter`
3. Place the input videos and their ground-truth JSON annotations in an `input/` folder alongside this code folder (course-provided test data is not redistributed here for copyright reasons).
4. Run the pipeline: `python tp1.py` — processes all videos in `input/`, writes predictions to `../output/`, and prints per-video timing and metrics (under a minute for four videos).
5. Open `report.ipynb` and run it top to bottom to reproduce every figure, table and the full analysis.

**Code layout:** `tp1.py` orchestrates the pipeline; `signals.py` implements the three dissimilarity signals and cut detection; `verify.py` implements SIFT/RANSAC verification; `recurring.py` implements keyframe-based scene grouping; `metrics.py`, `grouping.py` and `plotting.py` are supporting utilities.

## Limitations & Future Work
* **Content that changes colour quickly without a cut** (e.g. fireworks) fools the histogram signal, and the dark, low-texture frames around it defeat SIFT verification too. A secondary structural signal (e.g. ECR, or background feature matching) could help distinguish this from a real cut.
* **RANSAC can fit a geometrically meaningless homography** to a handful of coincidental matches, occasionally causing a real cut to be rejected. A sanity check on the recovered homography (e.g. requiring it to be close to a small, plausible camera motion) would make the verification step more robust.
* **The scene-similarity threshold sits in a narrow safe range** on the development videos, so it does not always transfer perfectly to new footage. Combining colour similarity with a secondary SIFT-based check (explored as an extra in the report) recovered some of the missed scene links on the test set.
