"""species_800: dictionary lookup vs. logistic-regression BIO tagger with ontology features."""
import io, json, os, re, sys, time
import numpy as np, pandas as pd, requests, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

SEED = 42
np.random.seed(SEED)
# The canonical "species_800" id now lives under a user namespace on the Hub; try several ids.
REPOS = ["spyysalo/species_800", "species_800", "bigbio/species_800"]
HEADERS = {}
if os.environ.get("HF_TOKEN"):
    HEADERS["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]


def get(url, tries=6):
    """HTTP GET with timeout and exponential backoff on 429/5xx; fail fast on other 4xx."""
    for k in range(tries):
        try:
            r = requests.get(url, timeout=60, headers=HEADERS)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(str(r.status_code))
            if 400 <= r.status_code < 500:
                print("client error", r.status_code, url)
                break
            return r
        except requests.RequestException as e:
            print("retry", k, e)
            time.sleep(min(2 ** k, 20))
    raise RuntimeError("download failed: " + url)


def find_meta():
    """Return (repo_id, {config: {split: [parquet urls]}}) from the first repo that works."""
    for repo in REPOS:
        try:
            meta = get(f"https://huggingface.co/api/datasets/{repo}/parquet", tries=2).json()
            if meta:
                return repo, meta
        except Exception as e:
            print("API failed for", repo, e)
        # direct auto-converted parquet files
        for cfg in ("default", "species_800"):
            meta = {cfg: {}}
            try:
                for sp in ("train", "test"):
                    u = f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{cfg}/{sp}/0000.parquet"
                    get(u, tries=1)
                    meta[cfg][sp] = [u]
                return repo, meta
            except Exception as e:
                print("direct parquet failed for", repo, cfg, e)
    raise RuntimeError("could not locate species_800 parquet files")


def load_split(meta, split):
    urls = []
    for cfg, splits in meta.items():  # {config: {split: [urls]}}
        urls += splits.get(split, [])
    return pd.concat([pd.read_parquet(io.BytesIO(get(u).content)) for u in urls], ignore_index=True)


def label_names(repo):
    try:
        info = get(f"https://datasets-server.huggingface.co/info?dataset={repo}", tries=2).json()
        cfg = list(info["dataset_info"].values())[0] if "config_name" not in info["dataset_info"] else info["dataset_info"]
        return [n[0].upper() for n in cfg["features"]["ner_tags"]["feature"]["names"]]
    except Exception as e:
        print("label names fallback", e)
        return ["O", "B", "I"]


def to_lists(df, names):
    toks = [list(t) for t in df["tokens"]]
    tags = [[names[x] if not isinstance(x, str) else x[0].upper() for x in t] for t in df["ner_tags"]]
    return toks, tags


def spans(tags):
    """Set of (start, end) entity spans from BIO tags (stray I starts a span)."""
    out, s = set(), None
    for i, t in enumerate(list(tags) + ["O"]):
        if t in ("O", "B") or s is None:
            if s is not None:
                out.add((s, i))
                s = None
            if t in ("B", "I"):
                s = i
    return out


def prf(gold, pred):
    """Hand-written exact-match entity-level precision/recall/F1 (micro)."""
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        G = spans(g)
        P = spans(p)
        tp += len(G & P); fp += len(P - G); fn += len(G - P)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def make_dict(toks, tags):
    """Ontology list = lower-cased training entity strings."""
    d = {tuple(w.lower() for w in t[s:e]) for t, g in zip(toks, tags) for s, e in spans(g)}
    return d, max(len(x) for x in d), {w for x in d for w in x}


def lookup(t, D):
    """Longest-match left-to-right dictionary lookup -> BIO tags."""
    d, ml, _ = D
    low = [w.lower() for w in t]
    out, i = ["O"] * len(t), 0
    while i < len(t):
        for L in range(min(ml, len(t) - i), 0, -1):
            if tuple(low[i:i + L]) in d:
                out[i] = "B"; out[i + 1:i + L] = ["I"] * (L - 1); i += L
                break
        else:
            i += 1
    return out


def shape(w):
    return re.sub(r"(.)\1+", r"\1", re.sub(r"\d", "0", re.sub(r"[a-z]", "a", re.sub(r"[A-Z]", "A", w))))


def feats(t, dt, vocab):
    """Per-token feature dicts: word, affixes, shape, context, ontology match."""
    low = [w.lower() for w in t]
    sh = [shape(w) for w in t]
    n, rows = len(t), []
    for i, w in enumerate(t):
        f = {"w": low[i], "p3": low[i][:3], "s2": low[i][-2:], "s3": low[i][-3:], "s4": low[i][-4:],
             "sh": sh[i], "title": w.istitle(), "digit": any(c.isdigit() for c in w),
             "dt": dt[i], "inv": low[i] in vocab, "bias": 1}
        for o in (-2, -1, 1, 2):
            j = i + o
            f[f"w{o}"] = low[j] if 0 <= j < n else "<pad>"
            f[f"sh{o}"] = sh[j] if 0 <= j < n else "<pad>"
            f[f"dt{o}"] = dt[j] if 0 <= j < n else "<pad>"
            f[f"inv{o}"] = (low[j] in vocab) if 0 <= j < n else False
        rows.append(f)
    return rows


def oof_dtags(toks, tags):
    """Dictionary tags for training sentences from dictionaries built on the other folds
    (avoids the tagger over-trusting a dictionary that contains its own answers)."""
    dt = [None] * len(toks)
    for a, b in KFold(5, shuffle=True, random_state=SEED).split(toks):
        D = make_dict([toks[i] for i in a], [tags[i] for i in a])
        for i in b:
            dt[i] = (lookup(toks[i], D), D[2])
    return dt


def tagger_predict(trt, trg, evt, C):
    """Fit the logistic-regression tagger on (trt, trg), predict BIO tags for evt."""
    D = make_dict(trt, trg)
    X = []
    for t, (dt, v) in zip(trt, oof_dtags(trt, trg)):
        X += feats(t, dt, v)
    y = [x for g in trg for x in g]
    vec = DictVectorizer()
    clf = LogisticRegression(C=C, max_iter=300, random_state=SEED).fit(vec.fit_transform(X), y)
    out = []
    for t in evt:
        out.append(list(clf.predict(vec.transform(feats(t, lookup(t, D), D[2])))))
    return out


def main():
    repo, meta = find_meta()
    print("using dataset repo:", repo)
    names = label_names(repo)
    tr, te = load_split(meta, "train"), load_split(meta, "test")
    Xtr, Ytr = to_lists(tr, names)
    Xte, Yte = to_lists(te, names)
    print("sentences train/test:", len(Xtr), len(Xte), "label names:", names)
    if not Xtr or not Xte or sum(len(spans(g)) for g in Ytr) == 0:
        sys.exit("no usable data")
    # 10% of official train held out for tuning
    perm = np.random.RandomState(SEED).permutation(len(Xtr))
    cut = int(0.9 * len(Xtr))
    a, b = perm[:cut], perm[cut:]
    sub = lambda L, idx: [L[i] for i in idx]
    # tune the regularisation strength on the held-out part of train
    dev, dev_base = {}, None
    for C in (0.1, 1.0, 10.0):
        p = tagger_predict(sub(Xtr, a), sub(Ytr, a), sub(Xtr, b), C)
        dev[C] = prf(sub(Ytr, b), p)[2]
        print("C", C, "dev F1", round(dev[C], 4))
    bestC = max(dev, key=dev.get)
    Dsub = make_dict(sub(Xtr, a), sub(Ytr, a))
    dev_base = prf(sub(Ytr, b), [lookup(t, Dsub) for t in sub(Xtr, b)])[2]
    # final: both models use the full train set, scored once on test
    D = make_dict(Xtr, Ytr)
    bp, br, bf = prf(Yte, [lookup(t, D) for t in Xte])
    print("baseline test F1", round(bf, 4))
    ip, ir, i_f = prf(Yte, tagger_predict(Xtr, Ytr, Xte, bestC))
    print("improved test F1", round(i_f, 4))
    if not np.isfinite([bf, i_f]).all():
        sys.exit("invalid metrics")
    os.makedirs("figures", exist_ok=True)
    plt.bar(["dictionary lookup", "LR tagger"], [bf, i_f], color=["gray", "tab:blue"])
    plt.ylabel("entity-level exact-match F1 (test)"); plt.tight_layout()
    plt.savefig("figures/f1.png", dpi=120)
    res = {
        "question": "Can a lightweight context-aware token classifier with ontology-term features extract organism entities from scientific text better than a plain ontology-term lookup?",
        "method": "species_800 official train/test; 10% of train held out to tune C; dictionary lookup vs logistic-regression BIO tagger; exact-match entity-level F1.",
        "dataset": {"name": "species_800", "source_url": "https://huggingface.co/datasets/" + repo,
                    "n_rows": len(Xtr) + len(Xte), "n_train": len(Xtr), "n_test": len(Xte),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "entity-level exact-match F1", "higher_is_better": True,
        "baseline": "Longest-match dictionary lookup using training entity strings as the ontology list.",
        "improved": "Logistic-regression BIO tagger with word, suffix, shape, context and ontology-match features; C tuned on held-out 10% of train.",
        "metrics": {"baseline_score": bf, "improved_score": i_f, "improvement": i_f - bf,
                    "baseline_f1": bf, "improved_f1": i_f, "baseline_precision": bp, "baseline_recall": br,
                    "improved_precision": ip, "improved_recall": ir, "best_C": bestC,
                    "dev_f1_baseline": dev_base, "dev_f1_improved": dev[bestC]},
        "notes": "Deterministic models (single run, seed 42). The LLM of the paper is not re-implemented; species_800 (biomedical abstracts) is a stand-in for agricultural text. The ontology list comes from training annotations, so lookup recall is bounded by training vocabulary. Tagger ontology features use out-of-fold dictionaries on training data. Data loading fixed: the Hub id 'species_800' returned 401, so namespaced ids are tried.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res["metrics"], indent=2))


if __name__ == "__main__":
    main()
