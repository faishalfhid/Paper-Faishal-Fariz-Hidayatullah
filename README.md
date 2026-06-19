# Automatic Bloom's Taxonomy Question Generation Using RAG and ModernBERT Classification with Regenerative Feedback

Implementasi sistem AI untuk pembangkitan soal ujian secara otomatis berdasarkan tingkat kognitif Taksonomi Bloom (C1–C6), menggunakan pipeline **RAG (Retrieval-Augmented Generation) + LLM + ModernBERT Classifier + Feedback Loop**.

> Paper: *"Automatic Bloom's Taxonomy Question Generation Using RAG and ModernBERT Classification with Regenerative Feedback"*
> Penulis: Faishal Fariz Hidayatullah — Program Studi S2, Metodologi Penelitian

---

## Struktur Folder

```
.
├── DATASET/
│   ├── openstax_materials_v2_cleaned.csv   # Knowledge base (350 materi pembelajaran)
│   └── blooms_taxonomy_dataset.csv         # Dataset evaluasi klasifikasi (8.767 soal berlabel)
│
├── SOURCE CODE/
│   ├── llama_RAG_Bloom_Question_Generator.ipynb    # Pipeline dengan LLM Llama 3.1 (8B)
│   ├── Mixtral_RAG_Bloom_Question_Generator.ipynb  # Pipeline dengan LLM Mistral 7B
│   ├── Gemma_RAG_Bloom_Question_Generator.ipynb    # Pipeline dengan LLM Gemma3 4B
│   └── Qwen_RAG_Bloom_Question_Generator.ipynb     # Pipeline dengan LLM Qwen3 8B
│
├── Automatic Bloom's Taxonomy Question Generation ... .pdf   # Paper lengkap
├── Hasil iThenticate.pdf                                     # Hasil cek plagiasi
└── README.md
```

---

## Gambaran Umum Pipeline

Keempat notebook mengimplementasikan pipeline yang **identik**, hanya berbeda pada LLM yang digunakan di Phase 2. Pipeline terdiri dari 3 fase utama:

```
[CSV Materi]
     │
     ▼
PHASE 1A — Knowledge Base
  Load CSV → Preprocessing → Chunking (500 char) → Embedding (MiniLM-L6-v2) → Qdrant (in-memory)
     │
     ▼
PHASE 1B — Bloom Classifier
  ModernBERT fine-tuned (manzarimalik/ModernBERT-Bloom-Taxonomy)
  Input: soal teks → Output: level Bloom (C1–C6) + confidence score
     │
     ▼
PHASE 2 — RAG + LLM Generation
  Topic + Target Level → Retrieve top-3 chunks → Prompt LLM → Parse soal
     │
     ▼
PHASE 3 — Evaluation + Feedback Loop
  Klasifikasi soal hasil → Bandingkan dengan target:
    • Ideal (distance=0)         → Diterima
    • Slightly Ideal (distance=1)→ Diterima
    • Non-Ideal (distance≥2)    → Regenerasi (maks. 3x)
```

---

## Dataset

### `openstax_materials_v2_cleaned.csv` — Knowledge Base

| Kolom | Deskripsi |
|-------|-----------|
| `content` | Teks konten materi pembelajaran |
| `topic` | Nama sub-topik (contoh: "5.3 The Calvin Cycle") |
| `subject` | Mata pelajaran (contoh: Biology) |
| `keywords` | Kata kunci yang dipisah koma |

- **Jumlah baris:** 350 entri materi
- **Sumber:** OpenStax (Biology)
- **Hasil setelah chunking:** ~3.666 chunk (ukuran 500 karakter, overlap 50)

### `blooms_taxonomy_dataset.csv` — Dataset Evaluasi Klasifikasi

| Kolom | Deskripsi |
|-------|-----------|
| `Questions` | Teks soal |
| `Category` | Label level Bloom: `BT1` s.d. `BT6` |

- **Jumlah baris:** 8.767 soal berlabel
- **Distribusi label:** Remember (2.582), Understand (1.801), Apply (1.508), Analyze (1.293), Create (800), Evaluate (783)

---

## Model yang Digunakan

| Komponen | Model | Keterangan |
|----------|-------|------------|
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | 384 dimensi, cosine similarity |
| Classifier | `manzarimalik/ModernBERT-Bloom-Taxonomy` | Fine-tuned untuk klasifikasi C1–C6 |
| Tokenizer classifier | `answerdotai/ModernBERT-base` | Diload terpisah dari bobot classifier |
| Vector DB | Qdrant (in-memory) | Tidak perlu server eksternal |
| LLM | Lihat tabel notebook di bawah | Via Ollama (lokal) |

### LLM per Notebook

| Notebook | LLM | Ollama Model |
|----------|-----|--------------|
| `llama_RAG_Bloom_Question_Generator.ipynb` | Llama 3.1 8B | `llama3.1:8b` |
| `Mixtral_RAG_Bloom_Question_Generator.ipynb` | Mistral 7B | `mistral:7b` |
| `Gemma_RAG_Bloom_Question_Generator.ipynb` | Gemma3 4B | `gemma3:4b` |
| `Qwen_RAG_Bloom_Question_Generator.ipynb` | Qwen3 8B | `qwen3:8b` |

---

## Persyaratan

### Python Libraries

Install semua dependensi dengan menjalankan cell pertama pada notebook:

```bash
pip install -q -U \
    langchain \
    langchain-core \
    langchain-community \
    langchain-groq \
    langchain-huggingface \
    langchain-qdrant \
    langchain-text-splitters \
    qdrant-client \
    sentence-transformers \
    transformers \
    accelerate \
    datasets \
    langchain-ollama \
    scikit-learn \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    torch
```

### Dependensi Eksternal

- **Ollama** — untuk menjalankan LLM secara lokal. Install dari [ollama.com](https://ollama.com), lalu pull model yang dibutuhkan:

  ```bash
  ollama pull llama3.1:8b    # untuk notebook Llama
  ollama pull mistral:7b     # untuk notebook Mixtral
  ollama pull gemma3:4b      # untuk notebook Gemma
  ollama pull qwen3:8b       # untuk notebook Qwen
  ```

- **GPU (opsional tapi direkomendasikan)** — Classifier ModernBERT berjalan jauh lebih cepat di GPU. Notebook diuji dengan NVIDIA GeForce RTX 3060.

---

## Cara Menjalankan

### Opsi A: Lokal (dengan Ollama)

1. Pastikan Ollama sudah berjalan di background dan model sudah ter-pull.
2. Letakkan kedua file CSV dari folder `DATASET/` ke direktori yang sama dengan notebook (atau sesuaikan path di cell loading data).
3. Buka notebook di Jupyter Lab / VS Code, lalu jalankan cell dari atas ke bawah secara berurutan.
4. Cell install dependency (#1) hanya perlu dijalankan sekali.

### Opsi B: Google Colab

1. Upload notebook `.ipynb` ke Google Colab (`File → Upload notebook`).
2. Ubah runtime ke GPU: `Runtime → Change runtime type → T4 GPU`.
3. Upload kedua file CSV ke Colab (`Files panel → Upload`).
4. Pada cell konfigurasi LLM, ganti `ChatOllama` dengan `ChatGroq` dan isi API key Groq.
5. Jalankan cell dari atas ke bawah. Jika diminta restart runtime setelah install, restart lalu lanjutkan dari cell import.

---

## Alur Eksekusi Notebook (Urutan Cell)

| Nomor | Konten |
|-------|--------|
| 1 | Install dependencies |
| 2 | Import library |
| 3 | Konfigurasi konstanta (model, chunk size, TOP_K, dsb.) |
| 4 | Phase 1A: Load CSV materi → preprocessing → chunking |
| 5 | Phase 1A: Embedding + build Qdrant vector store |
| 6 | Phase 1B: Load ModernBERT classifier |
| 7 | Phase 1B: Helper `classify_bloom()` + smoke test |
| 8 | Phase 2: Setup retriever |
| 9 | Phase 2: Setup LLM (`ChatOllama`) |
| 10 | Phase 2: Prompt template + generation chain (`generate_questions()`) |
| 11 | Phase 2: Quick test generasi soal |
| 12 | Phase 3: `evaluate_question()` + `generate_with_feedback()` |
| 13 | Phase 3: Test feedback loop single topic |
| 14 | End-to-end demo: `batch_generate()` untuk satu topik |
| 15 | Evaluasi klasifikasi ModernBERT pada `blooms_taxonomy_dataset.csv` |
| 16 | Confusion matrix klasifikasi |
| 17 | Evaluasi paper-style (Ideal/Slightly Ideal/Non-Ideal) — `run_full_evaluation_resumable()` |
| 18 | Hitung TP/FP/FN keseluruhan + Precision/Recall/F1/Accuracy |
| 19 | Confusion matrix generasi (target vs predicted) |
| 20 | Metrik per level Bloom |
| 21 | Bar chart metrik per level |
| 22 | Stacked bar chart distribusi status |
| 23 | Export hasil ke CSV |

---

## Fitur Resumable Evaluation

Evaluasi penuh (seluruh topik x 6 level Bloom = ±1.914 kombinasi) membutuhkan waktu lama. Setiap notebook menyimpan progres ke file checkpoint `.jsonl` sehingga bisa dilanjutkan jika terputus:

| Notebook | Checkpoint File |
|----------|----------------|
| Llama | `eval_checkpoint.jsonl` |
| Mixtral | `mixtral_eval_checkpoint.jsonl` |
| Gemma | `gemma_eval_checkpoint.jsonl` |
| Qwen | `qwen_eval_checkpoint.jsonl` |

Cukup jalankan ulang cell evaluasi — baris yang sudah selesai akan dilewati otomatis. Hapus file `.jsonl` untuk memulai evaluasi dari awal.

---

## Output Evaluasi

### 1. Evaluasi Klasifikasi ModernBERT (pada `blooms_taxonomy_dataset.csv`)

Hasil yang diperoleh (sama di semua notebook karena classifier-nya identik):

| Metrik | Nilai |
|--------|-------|
| Accuracy | 0.8636 |
| Precision (weighted) | 0.8677 |
| Recall (weighted) | 0.8636 |
| F1-score (weighted) | 0.8629 |

### 2. Evaluasi Kualitas Generasi Soal (Paper-Style)

Definisi metrik: **TP** = Ideal (distance=0), **FP** = Slightly Ideal (distance=1), **FN** = Non-Ideal (distance≥2)

| LLM | Precision | Recall | F1-score | Accuracy |
|-----|-----------|--------|----------|----------|
| Llama 3.1 8B | 0.8152 | 0.9599 | 0.8817 | 0.7884 |
| Gemma3 4B | 0.8522 | 0.9188 | 0.8843 | 0.7926 |
| Mistral 7B | 0.7838 | 0.8894 | 0.8333 | 0.7142 |
| Qwen3 8B | 0.8603 | 0.7919 | 0.8247 | 0.7017 |

### 3. File Hasil yang Dihasilkan

Setelah evaluasi selesai, notebook menyimpan:
- `evaluation_results.csv` (Llama)
- `MIXTRAL_evaluation_results.csv`
- `GEMMA_evaluation_results.csv`
- `qwen_evaluation_results.csv` (nama variabel di notebook menunjuk ke ini)

---

## Konfigurasi Parameter Utama

Semua parameter dapat diubah di cell konfigurasi (cell #3) masing-masing notebook:

```python
EMBEDDING_MODEL           = "sentence-transformers/all-MiniLM-L6-v2"
CLASSIFIER_MODEL          = "manzarimalik/ModernBERT-Bloom-Taxonomy"
CLASSIFIER_BASE_TOKENIZER = "answerdotai/ModernBERT-base"
COLLECTION_NAME           = "learning_materials"
CHUNK_SIZE                = 500    # ukuran chunk teks (karakter)
CHUNK_OVERLAP             = 50     # overlap antar chunk
TOP_K                     = 3      # jumlah chunk yang diambil per query
MAX_REGENERATION_ATTEMPTS = 3      # maks. percobaan regenerasi
```

---

## Mapping Taksonomi Bloom

| Kode | Nama | Action Verbs |
|------|------|-------------|
| C1 (BT1) | Remember | define, list, identify, name, recall, state |
| C2 (BT2) | Understand | describe, explain, summarize, classify, compare |
| C3 (BT3) | Apply | apply, demonstrate, solve, use, compute, illustrate |
| C4 (BT4) | Analyze | analyze, contrast, examine, differentiate, infer |
| C5 (BT5) | Evaluate | assess, critique, judge, defend, justify, evaluate |
| C6 (BT6) | Create | design, construct, develop, formulate, propose, plan |
