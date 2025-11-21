# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
#
# GPL v3 license
#

import embeddings
from cupy_utils import *

import argparse
import collections
import numpy as np
import sys
import os
import glob
import csv


BATCH_SIZE = 500


def topk_mean(m, k, inplace=False):  # Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k


def run_eval(src_embeddings, trg_embeddings, args):
    # Choose dtype
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read embeddings
    srcfile = open(src_embeddings, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(trg_embeddings, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)

    # Normalize embeddings (for cosine)
    if not args.dot:
        embeddings.length_normalize(x)
        embeddings.length_normalize(z)

    # Build word->index maps
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # Read dictionary and compute coverage
    f = open(args.dictionary, encoding=args.encoding, errors='surrogateescape')
    src2trg = collections.defaultdict(set)
    oov = set()
    vocab = set()
    for line in f:
        src, trg = line.split()
        try:
            src_ind = src_word2ind[src]
            trg_ind = trg_word2ind[trg]
            src2trg[src_ind].add(trg_ind)
            vocab.add(src)
        except KeyError:
            oov.add(src)
    src = list(src2trg.keys())
    oov -= vocab
    coverage = len(src2trg) / (len(src2trg) + len(oov)) if (len(src2trg) + len(oov)) > 0 else 0.0

    # Translation candidates
    max_k = max(args.topk_list)
    translation = {}
    if args.retrieval == 'nn':
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = x[src[i:j]].dot(z.T)
            topk = (-similarities).argsort(axis=1)[:, :max_k]
            for k in range(j - i):
                translation[src[i + k]] = topk[k].tolist()

    elif args.retrieval == 'csls':
        knn_sim_bwd = xp.zeros(z.shape[0])
        for i in range(0, z.shape[0], BATCH_SIZE):
            j = min(i + BATCH_SIZE, z.shape[0])
            knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
        for i in range(0, len(src), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(src))
            similarities = 2 * x[src[i:j]].dot(z.T) - knn_sim_bwd
            topk = (-similarities).argsort(axis=1)[:, :max_k]
            for k in range(j - i):
                translation[src[i + k]] = topk[k].tolist()

    # Compute accuracy for each k
    acc_dict = {}
    for k in args.topk_list:
        correct = 0
        for i in src:
            predicted = translation[i][:k]
            gold = src2trg[i]
            if any(t in gold for t in predicted):
                correct += 1
        accuracy = correct / len(src) if len(src) > 0 else 0.0
        acc_dict[k] = accuracy

    # ---- Compute average cosine similarity for dictionary pairs ----
    cos_sims = []
    for src_ind, trg_inds in src2trg.items():
        for trg_ind in trg_inds:
            sim = float(x[src_ind].dot(z[trg_ind]))
            cos_sims.append(sim)
    avg_cos = float(np.mean(cos_sims)) if cos_sims else 0.0

    return coverage, acc_dict, avg_cos


def auto_eval(folder, args):
    results = []

    # 遍历 folder 下所有子文件夹
    for subdir, dirs, files in os.walk(folder):
        # 只识别 .vec 文件
        vec_files = [os.path.join(subdir, f) for f in files if f.endswith(".vec")]

        # 如果没有 vec 文件，跳过
        if len(vec_files) == 0:
            continue

        # 如果不等于 2，认为错误
        if len(vec_files) != 2:
            print(f"[WARN] 子文件夹中 .vec 文件不是 2 个: {subdir} (找到 {len(vec_files)} 个)")
            continue

        # 取两个 vec 文件
        file1, file2 = sorted(vec_files)

        print(f"\n=== Evaluating in folder: {subdir} ===")
        print(f"  File 1: {file1}")
        print(f"  File 2: {file2}")

        coverage, acc_dict, avg_cos = run_eval(file1, file2, args)

        row = {
            "folder": subdir,
            "file1": file1,
            "file2": file2,
            "coverage": coverage,
            "avg_cos": avg_cos,
        }
        for k, acc in acc_dict.items():
            row[f"top{k}_acc"] = acc

        results.append(row)

    # 输出结果
    dict_base = os.path.splitext(os.path.basename(args.dictionary))[0]
    out_csv = os.path.join(folder, f"evaluation_results_{dict_base}.csv")

    fieldnames = ["folder", "file1", "file2", "coverage"] + \
                 [f"top{k}_acc" for k in args.topk_list] + ["avg_cos"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n[INFO] 结果已保存到 {out_csv}")




def main():
    parser = argparse.ArgumentParser(description='Evaluate embeddings in a folder (USA* vs China*)')
    parser.add_argument('folder', help='folder containing embeddings')
    parser.add_argument('-d', '--dictionary', required=True, help='the test dictionary file')
    parser.add_argument('--retrieval', default='csls', choices=['nn', 'invnn', 'invsoftmax', 'csls'], help='retrieval method')
    parser.add_argument('--inv_temperature', default=1, type=float, help='inverse temperature (for invsoftmax)')
    parser.add_argument('--inv_sample', default=None, type=int, help='random subset size for inverse computations (invsoftmax)')
    parser.add_argument('-k', '--neighborhood', default=10, type=int, help='neighborhood size (for csls)')
    parser.add_argument('--dot', action='store_true', help='use dot product instead of cosine')
    parser.add_argument('--encoding', default='utf-8', help='character encoding for input/output')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='floating-point precision')
    parser.add_argument('--topk_list', type=int, nargs='+', default=[1, 5, 10], help='compute Top-k accuracy for given k values (default=[1,5,10])')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    args = parser.parse_args()

    auto_eval(args.folder, args)


if __name__ == '__main__':
    main()
