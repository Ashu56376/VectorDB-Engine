from __future__ import annotations

import heapq
import io
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Optional

import requests
from flask import Flask, jsonify, request, render_template_string
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vectordb")

DIMS = 16
MAX_UPLOAD_MB = 25
ALLOWED_EXTENSIONS = {"txt", "md", "csv", "json", "pdf"}

# =========================================================
# VECTOR DATA STRUCTURES
# =========================================================

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
    group: str = ""


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0 or nb == 0:
        return 1.0

    return max(0.0, min(2.0, 1.0 - dot / (na * nb)))


METRICS = {
    "cosine": cosine,
    "euclidean": euclidean,
    "manhattan": manhattan,
}


def get_metric(name):
    name = (name or "cosine").lower()
    if name not in METRICS:
        raise ValueError(f"Unknown metric. Choose: {list(METRICS)}")
    return METRICS[name]


def validate_vector(vector, expected_dims=None):
    if not isinstance(vector, list) or not vector:
        raise ValueError("Embedding must be a non-empty list.")

    try:
        vector = [float(x) for x in vector]
    except (TypeError, ValueError):
        raise ValueError("Embedding values must be numeric.")

    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains NaN or infinity.")

    if expected_dims is not None and len(vector) != expected_dims:
        raise ValueError(
            f"Expected {expected_dims} dimensions, received {len(vector)}."
        )

    return vector


# =========================================================
# BRUTE-FORCE SEARCH
# =========================================================

class BruteForce:
    def __init__(self):
        self.items = {}

    def insert(self, item):
        self.items[item.id] = item

    def remove(self, item_id):
        self.items.pop(item_id, None)

    def knn(self, query, k, metric):
        fn = get_metric(metric)
        results = [
            (fn(query, item.emb), item.id)
            for item in self.items.values()
        ]
        return sorted(results)[:k]


# =========================================================
# KD-TREE
# Exact fallback search is used for reliable results.
# =========================================================

class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.items = {}

    def insert(self, item):
        if len(item.emb) != self.dims:
            raise ValueError("KD-tree vector dimensions do not match.")
        self.items[item.id] = item

    def remove(self, item_id):
        self.items.pop(item_id, None)

    def knn(self, query, k, metric, prune=True):
        # Exact exhaustive ranking avoids incorrect pruning when cosine
        # distance is selected. This is a correctness-first implementation.
        fn = get_metric(metric)
        results = [
            (fn(query, item.emb), item.id)
            for item in self.items.values()
        ]
        return sorted(results)[:k]


# =========================================================
# HNSW APPROXIMATE NEAREST-NEIGHBOR INDEX
# =========================================================

class HNSW:
    def __init__(self, m=16, ef_build=200):
        self.M = max(2, int(m))
        self.ef_build = max(10, int(ef_build))
        self.items = {}
        self.neighbors = {}
        self.lock = threading.RLock()

    def insert(self, item, metric=cosine):
        with self.lock:
            self.items[item.id] = item
            self.neighbors.setdefault(item.id, set())

            if len(item.emb) == 0:
                return

            others = []
            for other_id, other in self.items.items():
                if other_id == item.id:
                    continue
                distance = metric(item.emb, other.emb)
                others.append((distance, other_id))

            others.sort()
            for _, other_id in others[:self.M]:
                self.neighbors[item.id].add(other_id)
                self.neighbors.setdefault(other_id, set()).add(item.id)

                # Keep each node's adjacency list bounded.
                if len(self.neighbors[other_id]) > self.M * 2:
                    node = self.items[other_id]
                    ranked = sorted(
                        (
                            metric(node.emb, self.items[n].emb), n
                        )
                        for n in self.neighbors[other_id]
                        if n in self.items
                    )
                    self.neighbors[other_id] = {
                        n for _, n in ranked[:self.M * 2]
                    }

    def remove(self, item_id):
        with self.lock:
            self.items.pop(item_id, None)
            self.neighbors.pop(item_id, None)
            for neighbors in self.neighbors.values():
                neighbors.discard(item_id)

    def knn(self, query, k, ef=50, metric=cosine):
        # Search all indexed vectors for a reliable answer. The graph is
        # maintained for inspection and can later be replaced with a fully
        # navigable HNSW implementation without changing the API.
        with self.lock:
            results = [
                (metric(query, item.emb), item.id)
                for item in self.items.values()
            ]
        return sorted(results)[:k]

    def get_info(self):
        with self.lock:
            return {
                "count": len(self.items),
                "M": self.M,
                "efBuild": self.ef_build,
                "edges": sum(len(x) for x in self.neighbors.values()),
            }


# =========================================================
# VECTOR DATABASE
# =========================================================

class VectorDB:
    ALGOS = ("brute", "kdtree", "hnsw")

    def __init__(self, dims=DIMS):
        self.dims = dims
        self.items = {}
        self.next_id = 1
        self.lock = threading.RLock()

        self.brute = BruteForce()
        self.kdtree = KDTree(dims)
        self.hnsw = HNSW()

    def insert(self, metadata, category, embedding):
        embedding = validate_vector(embedding, self.dims)

        with self.lock:
            item = VectorItem(
                self.next_id, metadata, category, embedding
            )

            self.items[item.id] = item
            self.brute.insert(item)
            self.kdtree.insert(item)
            self.hnsw.insert(item, cosine)

            self.next_id += 1
            return item.id

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.items:
                return False

            del self.items[item_id]
            self.brute.remove(item_id)
            self.kdtree.remove(item_id)
            self.hnsw.remove(item_id)
            return True

    def search(self, query, k=5, metric="cosine", algo="hnsw"):
        query = validate_vector(query, self.dims)
        metric_fn = get_metric(metric)
        algo = (algo or "hnsw").lower()

        if algo == "brute":
            result = self.brute.knn(query, k, metric)
        elif algo == "kdtree":
            result = self.kdtree.knn(query, k, metric)
        elif algo == "hnsw":
            result = self.hnsw.knn(query, k, 50, metric_fn)
        else:
            raise ValueError(f"Unknown algorithm. Choose: {self.ALGOS}")

        output = []
        for distance, item_id in result:
            item = self.items.get(item_id)
            if item:
                output.append({
                    "id": item.id,
                    "metadata": item.metadata,
                    "category": item.category,
                    "distance": round(distance, 6),
                })
        return output

    def all(self):
        with self.lock:
            return list(self.items.values())

    def __len__(self):
        return len(self.items)


# =========================================================
# OLLAMA CLIENT
# =========================================================

class OllamaUnavailable(Exception):
    pass


class OllamaClient:
    def __init__(self):
        self.base = os.environ.get(
            "OLLAMA_BASE_URL",
            "http://127.0.0.1:11434"
        ).rstrip("/")

        self.embed_model = os.environ.get(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text"
        )
        self.gen_model = os.environ.get(
            "OLLAMA_GEN_MODEL", "llama3.2:1b"
        )

    def is_available(self):
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text):
        if not text or not text.strip():
            raise ValueError("Cannot create an embedding for empty text.")

        try:
            response = requests.post(
                f"{self.base}/api/embed",
                json={
                    "model": self.embed_model,
                    "input": text,
                },
                timeout=120,
            )

            # Compatibility with older Ollama versions.
            if response.status_code in (404, 405):
                response = requests.post(
                    f"{self.base}/api/embeddings",
                    json={
                        "model": self.embed_model,
                        "prompt": text,
                    },
                    timeout=120,
                )

            response.raise_for_status()
            data = response.json()

            if data.get("embeddings"):
                embedding = data["embeddings"][0]
            else:
                embedding = data.get("embedding", [])

            return validate_vector(embedding)

        except (requests.RequestException, ValueError, KeyError) as exc:
            log.exception("Ollama embedding request failed")
            raise OllamaUnavailable(
                f"Embedding failed. Check Ollama and model "
                f"'{self.embed_model}'. Details: {exc}"
            ) from exc

    def generate(self, prompt):
        try:
            response = requests.post(
                f"{self.base}/api/generate",
                json={
                    "model": self.gen_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=300,
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()

            if not answer:
                raise OllamaUnavailable("Ollama returned an empty answer.")

            return answer

        except requests.RequestException as exc:
            log.exception("Ollama generation request failed")
            raise OllamaUnavailable(
                f"Generation failed. Check model '{self.gen_model}'."
            ) from exc


# =========================================================
# DOCUMENT DATABASE
# Dynamically detects embedding dimensions from first document.
# =========================================================

class DocumentDB:
    def __init__(self):
        self.items = {}
        self.next_id = 1
        self.dims = None
        self.lock = threading.RLock()

        self.brute = BruteForce()
        self.hnsw = HNSW()

    def insert(self, title, text, embedding, group=""):
        embedding = validate_vector(embedding)

        with self.lock:
            if self.dims is None:
                self.dims = len(embedding)
            elif len(embedding) != self.dims:
                raise ValueError(
                    f"Embedding dimension mismatch: document index expects "
                    f"{self.dims}, but model returned {len(embedding)}. "
                    "Use the same embedding model for all documents."
                )

            item = DocItem(
                self.next_id,
                title,
                text,
                embedding,
                group,
            )
            vector = VectorItem(
                item.id,
                title,
                "document",
                embedding,
            )

            self.items[item.id] = item
            self.brute.insert(vector)
            self.hnsw.insert(vector, cosine)
            self.next_id += 1
            return item.id

    def search(self, query, k=3, max_distance=1.0):
        query = validate_vector(query)

        with self.lock:
            if not self.items:
                return []

            if len(query) != self.dims:
                raise ValueError(
                    f"Question embedding has {len(query)} dimensions, "
                    f"but documents have {self.dims}. "
                    "Check that both use the same embedding model."
                )

            raw = self.brute.knn(query, k, "cosine")
            return [
                (distance, self.items[item_id])
                for distance, item_id in raw
                if item_id in self.items and distance <= max_distance
            ]

    def remove(self, item_id):
        with self.lock:
            if item_id not in self.items:
                return False

            del self.items[item_id]
            self.brute.remove(item_id)
            self.hnsw.remove(item_id)

            if not self.items:
                self.dims = None

            return True

    def all(self):
        with self.lock:
            return list(self.items.values())

    def __len__(self):
        return len(self.items)


# =========================================================
# DOCUMENT EXTRACTION AND CHUNKING
# =========================================================

def extract_text(filename, raw):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. "
            f"Supported types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError(
                "PDF support needs pypdf. Run: pip install pypdf"
            ) from exc

        try:
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc

        if not text.strip():
            raise ValueError(
                "No selectable text was found in the PDF. "
                "Scanned PDFs need OCR before upload."
            )
        return text

    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")


def chunk_text(text, chunk_words=220, overlap_words=40):
    words = re.findall(r"\S+", text)
    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = max(start + 1, end - overlap_words)

    return chunks


# =========================================================
# RAG PROMPT
# =========================================================

def build_prompt(question, hits):
    if not hits:
        return (
            "Answer the following question clearly and honestly. "
            "If you are uncertain, say so.\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    contexts = []
    for index, (distance, document) in enumerate(hits, start=1):
        contexts.append(
            f"[Source {index}: {document.title}]\n{document.text}"
        )

    context_text = "\n\n".join(contexts)

    return f"""
You are a document question-answering assistant.

Answer the user's question using the supplied document excerpts.
Prioritize the excerpts over general knowledge.
Do not invent facts, quotations, numbers, or references.
If the documents do not contain the answer, clearly say that the
provided documents do not contain enough information.
Use a concise but complete answer.

DOCUMENT EXCERPTS:
{context_text}

USER QUESTION:
{question}

ANSWER:
""".strip()


# =========================================================
# SIMPLE FRONTEND
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VectorDB Engine — Document Q&A</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#10141e;color:#edf2ff;font-family:Arial,sans-serif}
main{max-width:950px;margin:30px auto;padding:20px}
h1{font-size:28px}
.card{background:#1b2230;border:1px solid #30394a;border-radius:14px;
padding:20px;margin:18px 0}
button{background:#6278ff;color:white;border:0;padding:11px 16px;
border-radius:8px;cursor:pointer;font-weight:bold}
input,textarea,select{background:#111722;color:white;border:1px solid #394457;
border-radius:8px;padding:11px;width:100%;margin:8px 0}
textarea{min-height:100px;resize:vertical}
.small{color:#a8b4c9;font-size:13px}
#answer{white-space:pre-wrap;line-height:1.6}
#docs div{padding:10px;border-bottom:1px solid #30394a}
.error{color:#ff8d8d}
.success{color:#8be9a7}
</style>
</head>
<body>
<main>
<h1>VectorDB Engine</h1>
<p class="small">Document upload · Embeddings · Retrieval-Augmented Generation</p>

<section class="card">
<h2>1. Server status</h2>
<button onclick="checkStatus()">Check status</button>
<pre id="status">Checking...</pre>
</section>

<section class="card">
<h2>2. Upload a document</h2>
<p class="small">Supported: PDF, TXT, MD, CSV, JSON (25 MB maximum).</p>
<input id="file" type="file" accept=".pdf,.txt,.md,.csv,.json">
<input id="title" placeholder="Optional document title">
<button onclick="uploadDoc()">Upload document</button>
<p id="uploadResult"></p>
</section>

<section class="card">
<h2>3. Ask a question</h2>
<textarea id="question" placeholder="Ask something from your uploaded document..."></textarea>
<label>Number of contexts</label>
<select id="k">
<option value="3">3 contexts</option>
<option value="5">5 contexts</option>
<option value="8">8 contexts</option>
</select>
<button onclick="ask()">Generate answer</button>
<p id="askStatus" class="small"></p>
<h3>Answer</h3>
<div id="answer"></div>
<h3>Sources used</h3>
<div id="sources"></div>
</section>

<section class="card">
<h2>4. Uploaded document chunks</h2>
<button onclick="loadDocs()">Refresh documents</button>
<div id="docs"></div>
</section>
</main>

<script>
async function api(url, options={}) {
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); }
  catch { throw new Error("Server returned an invalid response."); }
  if (!response.ok) throw new Error(data.error || "Request failed.");
  return data;
}

async function checkStatus() {
  const el = document.getElementById("status");
  try {
    const d = await api("/status");
    el.textContent = JSON.stringify(d, null, 2);
  } catch (e) {
    el.textContent = e.message;
  }
}

async function uploadDoc() {
  const file = document.getElementById("file").files[0];
  const result = document.getElementById("uploadResult");

  if (!file) {
    result.className = "error";
    result.textContent = "Please choose a document first.";
    return;
  }

  const form = new FormData();
  form.append("file", file);
  form.append("title", document.getElementById("title").value.trim());

  result.textContent = "Uploading and creating embeddings...";
  result.className = "";

  try {
    const d = await api("/doc/upload", {method:"POST", body:form});
    result.className = "success";
    result.textContent =
      "Upload successful. Chunks: " + d.chunks +
      ", embedding dimensions: " + d.dims;
    await loadDocs();
    await checkStatus();
  } catch(e) {
    result.className = "error";
    result.textContent = e.message;
  }
}

async function ask() {
  const question = document.getElementById("question").value.trim();
  const answer = document.getElementById("answer");
  const sources = document.getElementById("sources");
  const status = document.getElementById("askStatus");

  if (!question) {
    status.textContent = "Please enter a question.";
    return;
  }

  answer.textContent = "";
  sources.textContent = "";
  status.textContent = "Searching documents and generating answer...";

  try {
    const d = await api("/doc/ask", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({question:question, k:Number(document.getElementById("k").value)})
    });

    answer.textContent = d.answer;
    status.textContent = d.usedDocuments
      ? "Answer generated using uploaded documents."
      : "No relevant document context found; model answered without document context.";

    for (const source of (d.contexts || [])) {
      const div = document.createElement("div");
      div.className = "card";
      const heading = document.createElement("strong");
      heading.textContent = source.title + " (distance: " + source.distance + ")";
      const body = document.createElement("p");
      body.textContent = source.text;
      div.appendChild(heading);
      div.appendChild(body);
      sources.appendChild(div);
    }
  } catch(e) {
    status.className = "error";
    status.textContent = e.message;
  }
}

async function loadDocs() {
  const box = document.getElementById("docs");
  try {
    const docs = await api("/doc/list");
    box.replaceChildren();

    if (!docs.length) {
      box.textContent = "No documents uploaded yet.";
      return;
    }

    for (const d of docs) {
      const row = document.createElement("div");
      const label = document.createElement("span");
      label.textContent = "#" + d.id + " — " + d.title + " (" + d.words + " words) ";
      const button = document.createElement("button");
      button.textContent = "Delete chunk";
      button.onclick = async () => {
        try {
          await api("/doc/delete/" + d.id, {method:"DELETE"});
          await loadDocs();
          await checkStatus();
        } catch(e) {
          alert(e.message);
        }
      };
      row.appendChild(label);
      row.appendChild(button);
      box.appendChild(row);
    }
  } catch(e) {
    box.textContent = e.message;
  }
}

checkStatus();
loadDocs();
</script>
</body>
</html>
"""


# =========================================================
# FLASK APP
# =========================================================

def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    vector_db = VectorDB(DIMS)
    document_db = DocumentDB()
    ollama = OllamaClient()

    def parse_k(data, default=3):
        try:
            k = int(data.get("k", default))
        except (TypeError, ValueError):
            raise ValueError("k must be an integer.")

        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20.")

        return k

    def ingest_document(title, text):
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("No readable text found in the document.")

        inserted_ids = []

        try:
            for index, chunk in enumerate(chunks, start=1):
                embedding = ollama.embed(chunk)

                chunk_title = (
                    f"{title} [{index}/{len(chunks)}]"
                    if len(chunks) > 1 else title
                )

                doc_id = document_db.insert(
                    chunk_title,
                    chunk,
                    embedding,
                    group=title,
                )
                inserted_ids.append(doc_id)

        except Exception:
            for doc_id in inserted_ids:
                document_db.remove(doc_id)
            raise

        return inserted_ids

    @app.errorhandler(OllamaUnavailable)
    def ollama_error(exc):
        return jsonify({
            "error": str(exc),
            "hint": (
                "Ensure Ollama is running and both models are installed: "
                "nomic-embed-text and llama3.2:1b."
            )
        }), 503

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({
            "error": f"File too large. Maximum upload size is {MAX_UPLOAD_MB} MB."
        }), 413

    @app.get("/")
    def index():
        return render_template_string(HTML)

    @app.get("/status")
    def status():
        return jsonify({
            "ollamaAvailable": ollama.is_available(),
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": len(document_db),
            "docDims": document_db.dims,
            "demoDims": DIMS,
            "demoCount": len(vector_db),
        })

    @app.post("/doc/upload")
    def doc_upload():
        uploaded = request.files.get("file")

        if not uploaded or not uploaded.filename:
            return jsonify({"error": "Select a file to upload."}), 400

        filename = secure_filename(uploaded.filename)
        if not filename:
            return jsonify({"error": "Invalid filename."}), 400

        try:
            raw = uploaded.read()
            if not raw:
                return jsonify({"error": "The uploaded file is empty."}), 400

            text = extract_text(filename, raw).strip()
            title = (
                request.form.get("title", "").strip()
                or filename
            )

            ids = ingest_document(title, text)

            return jsonify({
                "ok": True,
                "title": title,
                "ids": ids,
                "chunks": len(ids),
                "dims": document_db.dims,
            })

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/doc/insert")
    def doc_insert():
        data = request.get_json(silent=True) or {}
        title = str(data.get("title", "")).strip()
        text = str(data.get("text", "")).strip()

        if not title or not text:
            return jsonify({"error": "Both title and text are required."}), 400

        ids = ingest_document(title, text)

        return jsonify({
            "ok": True,
            "ids": ids,
            "chunks": len(ids),
            "dims": document_db.dims,
        })

    @app.get("/doc/list")
    def doc_list():
        return jsonify([
            {
                "id": d.id,
                "title": d.title,
                "preview": d.text[:180],
                "words": len(d.text.split()),
            }
            for d in document_db.all()
        ])

    @app.delete("/doc/delete/<int:doc_id>")
    def doc_delete(doc_id):
        return jsonify({"ok": document_db.remove(doc_id)})

    def retrieve(question, k, max_distance):
        hits = []

        if len(document_db):
            query_embedding = ollama.embed(question)
            hits = document_db.search(
                query_embedding,
                k=k,
                max_distance=max_distance,
            )

        return hits

    @app.post("/doc/search")
    def doc_search():
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()

        if not question:
            return jsonify({"error": "Question is required."}), 400

        try:
            k = parse_k(data)
            max_distance = float(data.get("max_dist", 1.0))

            if not math.isfinite(max_distance) or max_distance < 0:
                raise ValueError("max_dist must be a finite non-negative number.")

            hits = retrieve(question, k, max_distance)

            return jsonify({
                "contexts": [
                    {
                        "id": d.id,
                        "title": d.title,
                        "distance": round(distance, 4),
                        "text": d.text,
                    }
                    for distance, d in hits
                ]
            })

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.post("/doc/ask")
    def doc_ask():
        data = request.get_json(silent=True) or {}
        question = str(data.get("question", "")).strip()

        if not question:
            return jsonify({"error": "Question is required."}), 400

        try:
            k = parse_k(data)
            max_distance = float(data.get("max_dist", 1.0))

            if not math.isfinite(max_distance) or max_distance < 0:
                raise ValueError("max_dist must be a finite non-negative number.")

            hits = retrieve(question, k, max_distance)
            prompt = build_prompt(question, hits)
            answer = ollama.generate(prompt)

            return jsonify({
                "answer": answer,
                "model": ollama.gen_model,
                "usedDocuments": bool(hits),
                "source": "documents" if hits else "general_knowledge",
                "contexts": [
                    {
                        "id": d.id,
                        "title": d.title,
                        "text": d.text,
                        "distance": round(distance, 4),
                    }
                    for distance, d in hits
                ],
                "docCount": len(document_db),
            })

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    # Demo vector API
    @app.post("/insert")
    def vector_insert():
        data = request.get_json(silent=True) or {}

        try:
            metadata = str(data.get("metadata", "")).strip()
            category = str(data.get("category", "general"))
            embedding = validate_vector(data.get("embedding"), DIMS)

            if not metadata:
                raise ValueError("metadata is required.")

            item_id = vector_db.insert(metadata, category, embedding)
            return jsonify({"id": item_id})

        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/search")
    def vector_search():
        try:
            raw = request.args.get("v", "")
            query = validate_vector(
                [float(x) for x in raw.split(",") if x.strip()],
                DIMS,
            )
            k = int(request.args.get("k", 5))
            if not 1 <= k <= 100:
                raise ValueError("k must be between 1 and 100.")

            metric = request.args.get("metric", "cosine")
            algo = request.args.get("algo", "hnsw")

            return jsonify(vector_db.search(query, k, metric, algo))

        except (ValueError, TypeError) as exc:
            return jsonify({"error": str(exc)}), 400

    @app.delete("/delete/<int:item_id>")
    def vector_delete(item_id):
        return jsonify({"ok": vector_db.remove(item_id)})

    @app.get("/items")
    def vector_items():
        return jsonify([
            {
                "id": item.id,
                "metadata": item.metadata,
                "category": item.category,
                "embedding": item.emb,
            }
            for item in vector_db.all()
        ])

    @app.get("/stats")
    def stats():
        return jsonify({
            "count": len(vector_db),
            "dims": DIMS,
            "algorithms": list(VectorDB.ALGOS),
            "metrics": list(METRICS),
        })

    @app.get("/hnsw-info")
    def hnsw_info():
        return jsonify(vector_db.hnsw.get_info())

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    app = create_app()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
    )
