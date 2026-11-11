# Parte 1 do projeto: os tres sinais de diferenca entre frames e a detecao de cortes.
#
# A ideia de tudo isto e simples. Para cada frame t calculamos um numero d[t] que diz o quao diferente
# o frame t e do frame t-1. Dentro de um shot esse numero é pequeno e num corte da um pico.
# O d[0] fica sempre a 0 porque o primeiro frame nao tem nenhum antes dele.

import cv2
import numpy as np

import metrics

# bins do histograma de cor em HSV, por ordem H, S, V
# o V so tem 1 bin, o que na pratica quer dizer que ignoramos o brilho (explicado no color_histogram)
# estes valores foram escolhidos nos videos dev, a comparacao esta no notebook
HIST_BINS = (16, 2, 1)

# parametros do ECR: os dois thresholds do canny e o tamanho do quadrado da dilatacao
CANNY_LOW = 100
CANNY_HIGH = 200
DILATE_SIZE = 5


# ---------------------------------------------------------------------------
# sinal 1: diferenca de pixeis (a baseline)
# ---------------------------------------------------------------------------

def to_gray(frame):
    # passamos para float porque em uint8 as subtracoes dao a volta (10 - 20 dava 246 em vez de -10)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)


def pixel_difference(f1, f2):
    # media da diferenca absoluta entre os cinzentos dos dois frames, da um valor entre 0 e 255
    # e a diferenca de frames do lab 2 mas reduzida a um numero so
    return np.mean(np.abs(f1 - f2))


def pixel_signal(frames):
    d = np.zeros(len(frames))
    prev = to_gray(frames[0])
    for t in range(1, len(frames)):
        cur = to_gray(frames[t])
        d[t] = pixel_difference(prev, cur)
        prev = cur   # guardamos o cinzento para nao o calcular outra vez na proxima volta
    return d


# ---------------------------------------------------------------------------
# sinal 2: histograma de cor comparado com a distancia chi quadrado
# ---------------------------------------------------------------------------

def color_histogram(frame, space="hsv", bins=HIST_BINS):
    # o histograma diz que percentagem da imagem tem cada cor, mas nao diz onde estao as cores
    # por isso se a camara mexer um bocado o histograma quase nao muda, mas num corte muda bastante
    #
    # em HSV a cor (H e S) fica separada do brilho (V). com 1 bin no V o brilho nao conta para nada,
    # e isso ajuda muito com os flashes porque um flash muda o brilho e nao a cor das coisas
    # a opcao rgb so existe para comparar no notebook
    if space == "hsv":
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ranges = [(0, 180), (0, 256), (0, 256)]   # no opencv o H vai de 0 a 179
    else:
        img = frame
        ranges = [(0, 256), (0, 256), (0, 256)]

    pixels = img.reshape(-1, 3)   # uma linha por pixel com os 3 canais
    hist, _ = np.histogramdd(pixels, bins=bins, range=ranges)
    hist = hist.ravel()           # passa de 3D para uma lista de bins

    # dividimos pelo total para ficar em percentagens (soma 1), assim nao depende do tamanho da imagem
    return hist / hist.sum()


def histogram_distance(h1, h2):
    # distancia chi quadrado feita por nos
    #   chi2 = 1/2 * soma de (h1 - h2)^2 / (h1 + h2)
    # dividir por h1 + h2 faz com que uma cor que aparece do nada pese mais do que uma cor
    # que ja ocupava a imagem toda e so variou um bocado
    # o 1/2 deixa o resultado entre 0 (iguais) e 1 (nenhuma cor em comum)
    soma = h1 + h2
    usados = soma > 0   # bins vazios nos dois iam dar 0/0, e nao sao diferenca nenhuma, entao saltamos
    return 0.5 * np.sum((h1[usados] - h2[usados]) ** 2 / soma[usados])


def histogram_signal(frames, space="hsv", bins=HIST_BINS):
    d = np.zeros(len(frames))
    prev = color_histogram(frames[0], space, bins)
    for t in range(1, len(frames)):
        cur = color_histogram(frames[t], space, bins)
        d[t] = histogram_distance(prev, cur)
        prev = cur
    return d


# ---------------------------------------------------------------------------
# sinal 3: edge change ratio (ECR)
# ---------------------------------------------------------------------------

def edge_map(frame):
    # arestas com o canny (o canny so funciona num canal, por isso passamos a cinzento primeiro)
    # o > 0 transforma o resultado em True/False, que e o que precisamos para as contas do ECR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH) > 0


def dilate_edges(edges):
    # engorda as arestas para um quadrado de DILATE_SIZE x DILATE_SIZE
    # assim uma aresta que se mexeu 1 ou 2 pixeis entre frames continua a cair dentro da antiga
    # o dilate nao aceita booleanos, por isso convertemos para uint8 e depois voltamos
    kernel = np.ones((DILATE_SIZE, DILATE_SIZE), np.uint8)
    return cv2.dilate(edges.astype(np.uint8), kernel) > 0


def edge_change_ratio(e_prev, e_t, d_prev, d_t):
    # e_prev e e_t sao as arestas dos frames t-1 e t, d_prev e d_t as mesmas mas dilatadas
    n_prev = np.sum(e_prev)
    n_t = np.sum(e_t)

    # caso especial que o enunciado pede para decidirmos: frames sem arestas nenhumas (tipo ecra preto)
    # se os dois nao tem arestas nao mudou nada, se so um tem entao mudou tudo
    # isto tambem evita dividir por zero mais abaixo
    if n_prev == 0 and n_t == 0:
        return 0.0
    if n_prev == 0 or n_t == 0:
        return 1.0

    # arestas que entram: estao no frame t mas longe de qualquer aresta do frame t-1
    x_in = np.sum(e_t & ~d_prev)
    # arestas que saem: estavam no frame t-1 e ficaram longe de qualquer aresta do frame t
    x_out = np.sum(e_prev & ~d_t)

    # cada uma dividida pelas arestas do seu frame, e ficamos com a maior
    return max(x_in / n_t, x_out / n_prev)


def ecr_signal(frames):
    d = np.zeros(len(frames))
    prev_edges = edge_map(frames[0])
    prev_dil = dilate_edges(prev_edges)
    for t in range(1, len(frames)):
        edges = edge_map(frames[t])
        dil = dilate_edges(edges)
        d[t] = edge_change_ratio(prev_edges, edges, prev_dil, dil)
        # as arestas deste frame passam a ser as do anterior na proxima volta,
        # assim cada frame so passa pelo canny e pela dilatacao uma vez
        prev_edges, prev_dil = edges, dil
    return d


def compute_signals(frames):
    # os tres sinais de uma vez, usado no notebook para comparar
    return {"pixel": pixel_signal(frames),
            "histogram": histogram_signal(frames),
            "ecr": ecr_signal(frames)}


# ---------------------------------------------------------------------------
# de um sinal para cortes, e como escolhemos os thresholds
# ---------------------------------------------------------------------------

def detect_cuts(d, threshold):
    # ha corte no frame t quando d[t] passa o threshold
    # o int e para os numeros irem bem para o json
    return [int(t) for t in np.flatnonzero(d > threshold)]


def detect_cuts_two(d1, d2, t1, t2):
    # combinacao de dois sinais: so ha corte quando os dois passam o seu threshold ao mesmo tempo
    # na pratica usamos histograma + ECR, o histograma encontra os cortes e o ECR serve de confirmacao
    return [int(t) for t in np.flatnonzero((d1 > t1) & (d2 > t2))]


# meia largura da janela da non-maximum suppression (o shot mais curto dos videos dev tem 29 frames)
NMS_WINDOW = 5


def local_maxima(d, cuts, window=NMS_WINDOW):
    # non-maximum suppression: com thresholds baixos os frames vizinhos de um corte tambem passam o threshold,
    # e cada um contava como um corte a mais. ficamos so com os cortes onde d[t] e o maximo de d entre
    # t - window e t + window, assim cada pico do sinal da um corte so
    # o max(0, ...) e porque em python um indice negativo nao da erro, contava a partir do fim do array
    return [t for t in cuts if d[t] == d[max(0, t - window):t + window + 1].max()]


def score_cuts(preds, gts):
    # avalia os cortes previstos em varios videos de uma vez
    # preds e gts sao dicionarios com o nome do video como chave
    # os totais se fazem somando os TP, FP e FN de todos os videos
    # e so depois calcular P, R e F1 (nada de fazer a media dos F1)
    tp = fp = fn = 0
    for name in gts:
        gt_cuts = [c["start"] for c in gts[name]["transitions"]]
        v_tp, v_fp, v_fn = metrics.match_cuts(gt_cuts, preds[name])
        tp += v_tp
        fp += v_fp
        fn += v_fn
    return metrics.precision_recall_f1(tp, fp, fn)


def sweep(d_per_video, gts, thresholds):
    # experimenta varios thresholds num sinal e devolve (T, P, R, F1) para cada um
    # cada um destes e um ponto da curva precision-recall
    results = []
    for T in thresholds:
        preds = {name: detect_cuts(d_per_video[name], T) for name in gts}
        p, r, f1 = score_cuts(preds, gts)
        results.append((T, p, r, f1))
    return results


def sweep_two(d1_per_video, d2_per_video, gts, thresholds1, thresholds2):
    # igual ao sweep mas para dois sinais combinados, testa todos os pares de thresholds
    # devolve (T1, T2, P, R, F1) para cada par
    results = []
    for t1 in thresholds1:
        for t2 in thresholds2:
            preds = {name: detect_cuts_two(d1_per_video[name], d2_per_video[name], t1, t2) for name in gts}
            p, r, f1 = score_cuts(preds, gts)
            results.append((t1, t2, p, r, f1))
    return results


def best_f1(results):
    # o resultado com o maior F1, que e sempre o ultimo valor de cada tuplo
    best = results[0]
    for res in results:
        if res[-1] > best[-1]:
            best = res
    return best
