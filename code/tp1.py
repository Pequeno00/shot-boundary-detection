###############################################################################
## Computer Vision 2026-2027 - NOVA FCT
## Project Stage 1 - Shot boundary detection and recurring scenes
##
## Rodrigo Sinde (68614)
## Pedro Pequeno (79987)
##
###############################################################################

import glob
import json
import os
import time

import cv2
import numpy as np

import grouping   # group_shots: union-find grouping of shots (Part 3, Task 7)
import metrics
import signals    # parte 1: sinais e detecao de cortes
import verify     # parte 2: verificacao dos cortes candidatos com SIFT
import recurring  # parte 3: agrupar os shots em cenas

INPUT_DIR = os.path.join("..", "input")
OUTPUT_DIR = os.path.join("..", "output")

# partes 1 e 2: os candidatos sao os picos do histograma acima deste threshold (o percentil 0.985 nos videos dev),
# mais baixo do que o da parte 1 porque a verificacao com SIFT elimina os falsos alarmes (notebook, seccao 3.4)
T_HIST = 0.0416


def read_frames(path, size=(320, 180)):
    """Read every frame of a video, resized to `size`. Returns a list of BGR images."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    cap.release()
    return frames


def process_video(frames, gt_shots):
    """Return (transitions, scenes) for one video.

    transitions: list of {"type": "cut", "start": t, "end": t}, t = first frame of the new shot (Parts 1 and 2)
    scenes:      dict {str(shot id): label} for the ground-truth shots (Part 3)
    """
    # parte 1: candidatos = picos do histograma acima do threshold (a NMS deixa um candidato por pico)
    d_hist = signals.histogram_signal(frames)
    candidates = signals.local_maxima(d_hist, signals.detect_cuts(d_hist, T_HIST))

    # parte 2: rejeitamos os candidatos em que os frames t - GAP e t + GAP tem muitos matches SIFT (mesmo conteudo)
    matches = verify.candidate_matches(frames, candidates)
    cuts = verify.verify_cuts(candidates, matches, verify.MIN_MATCHES)

    # no formato do enunciado um corte tem start = end = primeiro frame do shot novo
    transitions = [{"type": "cut", "start": t, "end": t} for t in cuts]

    # parte 3: com os shots do ground truth, keyframes -> matriz de semelhanca entre shots -> group_shots
    feats = recurring.shot_features(frames, gt_shots, recurring.N_KEYFRAMES)
    sim = recurring.similarity_matrix(feats, recurring.SIM_KIND, recurring.COMBINE)
    scenes = recurring.scene_labels(sim, gt_shots, recurring.T_SCENE)
    return transitions, scenes


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}                                    # video name -> counts from metrics.evaluate_video
    for video_path in sorted(glob.glob(os.path.join(INPUT_DIR, "*.mp4"))):
        name = os.path.splitext(os.path.basename(video_path))[0]
        gt = json.load(open(os.path.join(INPUT_DIR, name + ".json")))
        start = time.time()
        frames = read_frames(video_path)
        transitions, scenes = process_video(frames, gt["shots"])
        # para irmos controlando o tempo, o enunciado da 5 minutos para os 4 videos
        print(f"{name}: {len(frames)} frames, {len(transitions)} cuts, {time.time() - start:.1f} s")
        pred = {"video": name + ".mp4", "transitions": transitions, "scenes": scenes}
        with open(os.path.join(OUTPUT_DIR, name + ".json"), "w") as f:
            json.dump(pred, f, indent=1, default=int)    # default=int: accepts NumPy integers
        results[name] = metrics.evaluate_video(gt, pred)

    # summary table (Section 8): one row per video, then the dev / test totals
    dev = [results[n] for n in sorted(results) if n.startswith("dev")]
    test = [results[n] for n in sorted(results) if n.startswith("test")]
    metrics.print_table([(n, results[n]) for n in sorted(results)]
                        + [("DEV", metrics.sum_counts(dev)), ("TEST", metrics.sum_counts(test))])
