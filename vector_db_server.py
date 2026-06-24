from __future__ import annotations

import math
import heapq
import random
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests
from flask import Flask, request, jsonify, send_file

# =====================================================================
#  CONSTANTS
# =====================================================================

DIMS = 16  # Demo vector dimensionality

# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: list[float]


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: list[float]


DistFn = Callable[[list[float], list[float]], float]

# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean

# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort()
        return results[:k]

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]

# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item  = item
        self.left: Optional[KDNode]  = None
        self.right: Optional[KDNode] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def insert(self, v: VectorItem) -> None:
        self.root = self._ins(self.root, v, 0)

    def _ins(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        axis = depth % self.dims
        if v.emb[axis] < node.item.emb[axis]:
            node.left  = self._ins(node.left,  v, depth + 1)
        else:
            node.right = self._ins(node.right, v, depth + 1)
        return node

    def knn(self, q: list[float], k: int, dist: DistFn) -> list[tuple[float, int]]:
        # max-heap stored as negative distances
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        result = [(-d, idx) for d, idx in heap]
        result.sort()
        return result

    def _knn(self, node: Optional[KDNode], q: list[float], k: int,
             depth: int, dist: DistFn, heap: list) -> None:
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer  = node.left  if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left
        self._knn(closer,  q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def rebuild(self, items: list[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)

# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

@dataclass
class HNSWNode:
    item: VectorItem
    max_lyr: int
    nbrs: list[list[int]] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M         = m
        self.M0        = 2 * m
        self.ef_build  = ef_build
        self.mL        = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt  = -1
        self.G: dict[int, HNSWNode] = {}
        self._rng = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: list[float], ep: int, ef: int,
                      lyr: int, dist: DistFn) -> list[tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.G[ep].item.emb)
        cands = [(d0, ep)]
        found = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, idx) for d, idx in found]
        result.sort()
        return result

    def _select_nbrs(self, cands: list[tuple[float, int]], max_m: int) -> list[int]:
        return [idx for _, idx in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        id_  = item.id
        lvl  = self._rand_level()
        node = HNSWNode(item=item, max_lyr=lvl, nbrs=[[] for _ in range(lvl + 1)])
        self.G[id_] = node

        if self.entry_pt == -1:
            self.entry_pt  = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if ep in self.G and lc < len(self.G[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w    = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel  = self._select_nbrs(w, max_m)
            self.G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                n = self.G[nid]
                if lc >= len(n.nbrs):
                    n.nbrs.extend([] for _ in range(lc + 1 - len(n.nbrs)))
                n.nbrs[lc].append(id_)
                if len(n.nbrs[lc]) > max_m:
                    ds = sorted(
                        (dist(n.item.emb, self.G[c].item.emb), c)
                        for c in n.nbrs[lc] if c in self.G
                    )
                    n.nbrs[lc] = [c for _, c in ds[:max_m]]
            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt  = id_

    def knn(self, q: list[float], k: int, ef: int,
            dist: DistFn) -> list[tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if ep in self.G and lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, id_: int) -> None:
        if id_ not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = next((nid for nid in self.G if nid != id_), -1)
        del self.G[id_]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for id_, nd in self.G.items():
            nodes.append({
                "id": id_,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr":   nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer":      self.top_layer,
            "nodeCount":     len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes":         nodes,
            "edges":         edges,
        }

    def __len__(self) -> int:
        return len(self.G)

# =====================================================================
#  VECTOR DATABASE
# =====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims   = dims
        self._store: dict[int, VectorItem] = {}
        self._bf    = BruteForce()
        self._kdt   = KDTree(dims)
        self._hnsw  = HNSW(16, 200)
        self._lock  = threading.Lock()
        self._next  = 1

    def insert(self, meta: str, cat: str, emb: list[float], dist: DistFn) -> int:
        with self._lock:
            v = VectorItem(id=self._next, metadata=meta, category=cat, emb=emb)
            self._next += 1
            self._store[v.id] = v
            self._bf.insert(v)
            self._kdt.insert(v)
            self._hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._bf.remove(id_)
            self._hnsw.remove(id_)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: list[float], k: int,
               metric: str, algo: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0  = time.perf_counter()
            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                if id_ in self._store:
                    v = self._store[id_]
                    hits.append({
                        "id":        v.id,
                        "metadata":  v.metadata,
                        "category":  v.category,
                        "distance":  d,
                        "embedding": v.emb,
                    })
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: list[float], k: int, metric: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)

            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bruteforceUs": timed(lambda: self._bf.knn(q, k, dfn)),
                "kdtreeUs":     timed(lambda: self._kdt.knn(q, k, dfn)),
                "hnswUs":       timed(lambda: self._hnsw.knn(q, k, 50, dfn)),
                "itemCount":    len(self._store),
            }

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

class DocumentDB:
    def __init__(self):
        self._store: dict[int, DocItem] = {}
        self._hnsw  = HNSW(16, 200)
        self._bf    = BruteForce()
        self._lock  = threading.Lock()
        self._next  = 1
        self._dims  = 0

    def insert(self, title: str, text: str, emb: list[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(id=self._next, title=title, text=text, emb=emb)
            self._next += 1
            self._store[item.id] = item
            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: list[float], k: int,
               max_dist: float = 0.7) -> list[tuple[float, DocItem]]:
        with self._lock:
            if not self._store:
                return []
            raw = (self._bf.knn(q, k, cosine)
                   if len(self._store) < 10
                   else self._hnsw.knn(q, k, 50, cosine))
            return [(d, self._store[id_])
                    for d, id_ in raw
                    if id_ in self._store and d <= max_dist]

    def remove(self, id_: int) -> bool:
        with self._lock:
            if id_ not in self._store:
                return False
            del self._store[id_]
            self._hnsw.remove(id_)
            self._bf.remove(id_)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def get_dims(self) -> int:
        return self._dims

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250,
               overlap_words: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]
    step   = chunk_words - overlap_words
    chunks = []
    i      = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks

# =====================================================================
#  OLLAMA CLIENT
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base     = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model   = "llama3.2:1b"  # Fixed tag here

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        try:
            r = requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"

# =====================================================================
#  DEMO DATA
# =====================================================================

DEMO_DATA = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]


def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_DATA:
        db.insert(meta, cat, emb, dist)

# =====================================================================
#  FLASK REST API
# =====================================================================

def create_app() -> Flask:
    app    = Flask(__name__)
    db     = VectorDB(DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient()

    load_demo(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{len(db)} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    def parse_vec(s: str) -> list[float]:
        try:
            return [float(x) for x in s.split(",") if x.strip()]
        except Exception:
            return []

    @app.route("/search")
    def search():
        q = parse_vec(request.args.get("v", ""))
        if len(q) != DIMS:
            return jsonify({"error": f"need {DIMS}D vector"}), 400
        k      = int(request.args.get("k", 5))
        metric = request.args.get("metric", "cosine")
        algo   = request.args.get("algo", "hnsw")
        return jsonify(db.search(q, k, metric, algo))

    @app.route("/insert", methods=["POST"])
    def insert():
        data = request.get_json(force=True, silent=True) or {}
        meta = data.get("metadata", "")
        cat  = data.get("category", "")
        emb  = data.get("embedding", [])
        if not meta or len(emb) != DIMS:
            return jsonify({"error": "invalid body"}), 400
        id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
        return jsonify({"id": id_})

    @app.route("/delete/<int:id_>", methods=["DELETE"])
    def delete(id_):
        return jsonify({"ok": db.remove(id_)})

    @app.route("/items")
    def items():
        return jsonify([
            {"id": v.id, "metadata": v.metadata,
             "category": v.category, "embedding": v.emb}
            for v in db.all()
        ])

    @app.route("/benchmark")
    def benchmark():
        q = parse_vec(request.args.get("v", ""))
        if len(q) != DIMS:
            return jsonify({"error": f"need {DIMS}D vector"}), 400
        k      = int(request.args.get("k", 5))
        metric = request.args.get("metric", "cosine")
        return jsonify(db.benchmark(q, k, metric))

    @app.route("/hnsw-info")
    def hnsw_info():
        return jsonify(db.hnsw_info())

    @app.route("/stats")
    def stats():
        return jsonify({
            "count":      len(db),
            "dims":       DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics":    ["euclidean", "cosine", "manhattan"],
        })

    @app.route("/doc/insert", methods=["POST"])
    def doc_insert():
        data  = request.get_json(force=True, silent=True) or {}
        title = data.get("title", "")
        text  = data.get("text", "")
        if not title or not text:
            return jsonify({"error": "need title and text"}), 400

        chunks = chunk_text(text, 250, 30)
        ids    = []
        for i, chunk in enumerate(chunks):
            emb = ollama.embed(chunk)
            if not emb:
                return jsonify({
                    "error": (
                        "Ollama unavailable. Install from https://ollama.com then run: "
                        "ollama pull nomic-embed-text && ollama pull llama3.2:1b"
                    )
                }), 503
            chunk_title = (
                f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
            )
            ids.append(doc_db.insert(chunk_title, chunk, emb))

        return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})

    @app.route("/doc/delete/<int:id_>", methods=["DELETE"])
    def doc_delete(id_):
        return jsonify({"ok": doc_db.remove(id_)})

    @app.route("/doc/list")
    def doc_list():
        docs = doc_db.all()
        result = []
        for d in docs:
            preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
            result.append({
                "id":      d.id,
                "title":   d.title,
                "preview": preview,
                "words":   len(d.text.split()),
            })
        return jsonify(result)

    @app.route("/doc/search", methods=["POST"])
    def doc_search():
        data     = request.get_json(force=True, silent=True) or {}
        question = data.get("question", "")
        k        = int(data.get("k", 3))
        if not question:
            return jsonify({"error": "need question"}), 400

        q_emb = ollama.embed(question)
        if not q_emb:
            return jsonify({"error": "Ollama unavailable"}), 503

        hits = doc_db.search(q_emb, k)
        return jsonify({
            "contexts": [
                {"id": d.id, "title": d.title, "distance": round(dist, 4)}
                for dist, d in hits
            ]
        })

    @app.route("/doc/ask", methods=["POST"])
    def doc_ask():
        data     = request.get_json(force=True, silent=True) or {}
        question = data.get("question", "")
        k        = int(data.get("k", 3))
        if not question:
            return jsonify({"error": "need question"}), 400

        q_emb = ollama.embed(question)
        if not q_emb:
            return jsonify({"error": "Ollama unavailable"}), 503

        hits = doc_db.search(q_emb, k)

        ctx = "".join(
            f"[{i+1}] {d.title}:\n{d.text}\n\n"
            for i, (_, d) in enumerate(hits)
        )
        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things "
            "like 'the context doesn't mention'. Just answer the question naturally.\n\n"
            f"Context:\n{ctx}"
            f"Question: {question}\n\nAnswer:"
        )

        answer = ollama.generate(prompt)

        return jsonify({
            "answer":   answer,
            "model":    ollama.gen_model,
            "contexts": [
                {"id": d.id, "title": d.title,
                 "text": d.text, "distance": round(dist, 4)}
                for dist, d in hits
            ],
            "docCount": len(doc_db),
        })

    @app.route("/status")
    def status():
        up = ollama.is_available()
        return jsonify({
            "ollamaAvailable": up,
            "embedModel":      ollama.embed_model,
            "genModel":        ollama.gen_model,
            "docCount":        len(doc_db),
            "docDims":         doc_db.get_dims(),
            "demoDims":        DIMS,
            "demoCount":       len(db),
        })

    @app.route("/")
    def index():
        try:
            return send_file("index.html")
        except Exception:
            return "<h1>VectorDB Engine running</h1>", 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, threaded=True)