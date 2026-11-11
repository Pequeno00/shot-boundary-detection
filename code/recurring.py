# Parte 3 do projeto: agrupar os shots que sao da mesma cena.
#
# Aqui usamos os shots do ground truth (gt["shots"]) e nao os cortes que detetamos, assim os erros das
# partes 1 e 2 nao estragam esta parte. Cada shot e representado por um ou mais keyframes, comparamos os
# keyframes de dois shots (por cor ou por SIFT) e isso da uma matriz de semelhanca shot x shot,
# que depois vai para o grouping.group_shots.

import cv2
import numpy as np

import grouping
import metrics
import signals
import verify

# criados uma vez so e reutilizados, porque aqui vamos comparar centenas de pares de keyframes
SIFT = cv2.SIFT_create()
MATCHER = cv2.BFMatcher()

# a configuracao que ficou, escolhida nos videos dev (comparacao no notebook, seccao 4):
# 5 keyframes por shot, comparados pela cor, e dois shots ficam ligados pelo melhor par de keyframes
# o threshold e o meio da zona entre 0.970 e 0.986 onde o F1 dos pares no dev nao muda
N_KEYFRAMES = 5
SIM_KIND = "color"
COMBINE = "max"
T_SCENE = 0.978


def keyframes(shot, n=1):
    # indices dos keyframes de um shot: n frames espalhados por igual dentro do shot
    # com n = 1 da o frame do meio, com n = 3 da os frames a 1/4, 1/2 e 3/4 do shot
    # dividimos por n + 1 e nao por n para nunca apanhar o primeiro nem o ultimo frame,
    # que estao colados aos cortes
    start, end = shot["start"], shot["end"]
    length = end - start + 1
    return [start + int(length * (k + 1) / (n + 1)) for k in range(n)]


def keyframe_features(frame):
    # tudo o que precisamos de um keyframe: o histograma de cor e os descritores SIFT
    # calculamos isto uma vez por keyframe, porque cada keyframe vai ser comparado com todos os outros
    # e o SIFT e a parte lenta (o verify.sift_matches recalcula tudo em cada par, aqui isso era demasiado lento)
    hist = signals.color_histogram(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, des = SIFT.detectAndCompute(gray, None)
    return hist, des


def color_similarity(h1, h2):
    # o chi quadrado e uma distancia entre 0 (iguais) e 1 (nada em comum),
    # por isso 1 - chi2 da uma semelhanca: 1 = mesmas cores, 0 = nenhuma cor em comum
    return 1 - signals.histogram_distance(h1, h2)


def sift_similarity(des1, des2, ratio=verify.RATIO):
    # numero de matches SIFT que passam o ratio test de Lowe, como pede o enunciado (sem RANSAC)
    # e o mesmo teste do verify.sift_matches mas a partir dos descritores ja calculados
    if des1 is None or des2 is None:   # keyframe sem keypoints (tipo frame preto)
        return 0
    pairs = MATCHER.knnMatch(des1, des2, k=2)
    # len(p) == 2 porque se o outro frame so tiver 1 descritor o knnMatch so devolve 1 vizinho
    return sum(1 for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance)


def shot_features(frames, shots, n=1):
    # as features dos keyframes de todos os shots de um video
    # fica uma lista com uma entrada por shot, e cada entrada e a lista dos (hist, des) dos seus n keyframes
    return [[keyframe_features(frames[t]) for t in keyframes(shot, n)] for shot in shots]


def similarity_matrix(feats, kind="color", combine="max"):
    # matriz shot x shot: sim[a, b] e a semelhanca entre os shots a e b
    # kind diz se comparamos os keyframes pela cor ou pelo SIFT
    # com varios keyframes ha varios pares (keyframe de a, keyframe de b), e o combine diz como os juntamos:
    #   "max"  fica o melhor par, basta um angulo parecido nos dois shots para os ligar
    #   "mean" a media de todos os pares, os dois shots tem de ser parecidos no geral
    # com 1 keyframe por shot so ha um par e as duas opcoes dao o mesmo
    n = len(feats)
    sim = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            values = []
            for hist_a, des_a in feats[a]:
                for hist_b, des_b in feats[b]:
                    if kind == "color":
                        values.append(color_similarity(hist_a, hist_b))
                    else:
                        values.append(sift_similarity(des_a, des_b))
            if combine == "max":
                sim[a, b] = max(values)
            else:
                sim[a, b] = np.mean(values)
            sim[b, a] = sim[a, b]   # a matriz e simetrica, o grouping so olha para a < b mas assim fica certa para os graficos
    # a diagonal (um shot com ele proprio) fica a 0, o grouping nunca olha para ela
    return sim


def scene_labels(sim, shots, threshold):
    # agrupa os shots com o group_shots que nos deram e poe o resultado no formato do campo "scenes" do json
    labels = grouping.group_shots(sim, threshold)
    return {str(shot["id"]): label for shot, label in zip(shots, labels)}


def score_scenes(sims, gts, threshold):
    # F1 dos pares de shots em varios videos de uma vez, igual ao score_cuts da parte 1:
    # somamos os TP, FP e FN de todos os videos e so no fim calculamos P, R e F1
    # sims e gts sao dicionarios com o nome do video como chave
    tp = fp = fn = 0
    for name in gts:
        shots = gts[name]["shots"]
        gt_scenes = {str(shot["id"]): shot["scene"] for shot in shots}
        v_tp, v_fp, v_fn = metrics.pairwise_scene_counts(gt_scenes, scene_labels(sims[name], shots, threshold))
        tp += v_tp
        fp += v_fp
        fn += v_fn
    return metrics.precision_recall_f1(tp, fp, fn)


def sweep_scenes(sims, gts, thresholds):
    # experimenta varios thresholds e devolve (T, P, R, F1) para cada um, como o signals.sweep
    # o signals.best_f1 serve para escolher o melhor destes resultados
    results = []
    for T in thresholds:
        p, r, f1 = score_scenes(sims, gts, T)
        results.append((T, p, r, f1))
    return results
