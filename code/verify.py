# Parte 2 do projeto: verificar os cortes candidatos com features locais (SIFT).
#
# A ideia: o detetor da parte 1 da candidatos a corte. Para cada candidato t comparamos um frame antes (t - GAP)
# com um frame depois (t + GAP). Se os dois tem muitos matches SIFT entao mostram o mesmo conteudo e era um falso
# alarme (flash, camara a mexer, alguem a passar) e o candidato e rejeitado. Se tem poucos e um corte a serio.

import cv2
import numpy as np

import signals

# distancia ao candidato dos dois frames comparados (um flash dura ate 3 frames, assim ficamos fora dele)
GAP = 4
# Lowe's ratio test: um match e bom se o melhor vizinho for claramente melhor do que o segundo
RATIO = 0.75
# abaixo deste numero de matches o candidato e mantido como corte (escolhido no notebook, seccao 3.4)
MIN_MATCHES = 10
# contar so os inliers RANSAC de uma homografia em vez de todos os matches (seccao 3.3)
USE_RANSAC = True


def sift_matches(f1, f2, ratio=RATIO):
    """Matches SIFT entre dois frames BGR que passam o ratio test de Lowe.

    Devolve (kp1, kp2, good): os keypoints de cada frame e a lista dos matches bons (cv2.DMatch).
    Se um frame nao tiver keypoints (des = None) devolve zero matches, e os descritores com so 1 vizinho sao saltados.
    """
    gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []
    
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)

    best_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            best_matches.append(m)

    return kp1, kp2, best_matches


def count_matches(f1, f2, ratio=RATIO, use_ransac=USE_RANSAC):
    """Numero de matches bons entre dois frames.

    Com use_ransac=True conta so os inliers de uma homografia (cv2.findHomography com cv2.RANSAC),
    o que precisa de pelo menos 4 matches.
    """
    kp1, kp2, good = sift_matches(f1, f2, ratio)
    if not use_ransac:
        return len(good)

    if len(good) < 4:
        return 0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return 0
    return int(np.sum(mask))


def candidate_matches(frames, candidates, gap=GAP, use_ransac=USE_RANSAC):
    """Para cada candidato t, o numero de matches entre frames[t - gap] e frames[t + gap].

    Devolve um dict {t: numero de matches}. E a parte lenta (o SIFT), por isso calcula-se uma vez
    e depois testam-se varios min_matches so com estes numeros.
    """
    matches = {}
    for t in candidates:
        if t - gap < 0 or t + gap >= len(frames):
            continue  # nao da para verificar: fica fora do dict e o verify_cuts mantem-no como corte
        f1 = frames[t - gap]
        f2 = frames[t + gap]
        matches[t] = count_matches(f1, f2, use_ransac=use_ransac)
    return matches

def verify_cuts(candidates, matches, min_matches):
    """Os candidatos que sobrevivem a verificacao: os que tem menos de min_matches matches."""
    return [t for t in candidates if matches.get(t, 0) < min_matches]


def sweep_min_matches(candidates_per_video, matches_per_video, gts, values):
    """Como o signals.sweep, mas para o min_matches. Devolve (min_matches, P, R, F1) para cada valor.

    candidates_per_video e matches_per_video sao dicts com o nome do video como chave.
    """
    results = []
    for min_matches in values:
        preds = {name: verify_cuts(candidates_per_video[name], matches_per_video[name], min_matches) for name in gts}
        p, r, f1 = signals.score_cuts(preds, gts)
        results.append((min_matches, p, r, f1))
    return results
