[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/iA_HXS2O)
# Task

The task is described at [https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/](https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/)

The competition is hosted at [https://www.kaggle.com/competitions/ling-539-competition-2026](https://www.kaggle.com/competitions/ling-539-competition-2026)

**To join the competition, you must accept it at the following URL**: [https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b](https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b)

# Notes
- This project involves a **performance evaluation** as well as your **graded assessment**. It's important to keep these two things separate in your mind.
  - The rubric which will be used to assess your submission *for a grade* (ie, not to evaluate the performance of your model) is in the D2L assignment item
  - You are permitted to propose more than one classification model or approach. However, as described on the assessment rubric, **at least one of your submitted models must use one or more of the classification algorithms covered in this course.** (For more details related to assessment, be sure you understand the details of that rubric)
  - The performance of your model will be evaluated by Kaggle, and your model's performance will be ranked against other class submissions. The performance of your model is **one**, but not the only, factor by which your model will be assessed for a grade
- You are encouraged, but not obligated, to use Python
- You may delete or alter any files in this repository
- You are free to add dependencies, **however**, ensure that your code can be installed/used on another machine running Linux or MacOS (consider containerizing your project with Docker or an equivalent technology)

---

# Approach

3-class text classification:
- **Label 0**: Amazon product reviews
- **Label 1**: Positive IMDb movie reviews
- **Label 2**: Negative IMDb movie reviews

Key insight: ~2,400 training texts are in Chinese and belong exclusively to label 0 -- classified by rule with 100% accuracy before any ML model runs.

## Models

| Version | Script | Method | Val Acc | Submission |
|---|---|---|---|---|
| v1 | `solutions/solution.py` | TF-IDF + Logistic Regression (baseline) | 0.8879 | `submission_logreg.csv` |
| v1b | `solutions/solution.py` | TF-IDF + LinearSVC | 0.7983 | `submission_linear_svc.csv` |
| v2 | `solutions/solution_v2.py` | TF-IDF (char on raw text) + LogReg + Chinese rule | 0.8970 | `submission_logreg_v2.csv` |
| v3 | `solutions/solution_v3.py` | Sentence embeddings (all-MiniLM-L6-v2) + LogReg | 0.8798 | `submission_logreg_v3.csv` |
| v4 | `solutions/solution_v4.py` | TF-IDF + Sentence embeddings + LogReg + Chinese rule | 0.9066 | `submission_logreg_v4.csv` |
| v5 | `solutions/solution_v5.py` | DistilBERT (finetuned-sst-2) + Chinese rule -- **best** | **0.9439** | `submission_distilbert_sst2.csv` |

## Running with Docker (recommended)

```bash
# Build
docker build -t ling539 .

# Run Jupyter notebook (opens at http://localhost:9999)
docker run -p 9999:9999 -v $(pwd)/data:/app/data ling539

# Run any solution script
docker run -v $(pwd)/data:/app/data ling539 python solutions/solution_v4.py
docker run -v $(pwd)/data:/app/data ling539 python solutions/solution_v5.py
```

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download competition data first (requires Kaggle API token)
kaggle competitions download ling-539-competition-2026 -p data/
unzip data/ling-539-competition-2026.zip -d data/

# Run in order (v4 reuses v2/v3 artifacts; v5 is independent)
python solutions/solution_v2.py   # TF-IDF vectorizers
python solutions/solution_v3.py   # sentence embeddings
python solutions/solution_v4.py   # combined -- best classical model
python solutions/solution_v5.py   # DistilBERT -- best overall (GPU recommended)

# Submit best result to Kaggle
kaggle competitions submit ling-539-competition-2026 \
    -f submissions/submission_distilbert_sst2.csv \
    -m "v5: DistilBERT-SST2 fine-tuned, val acc 0.9439"
```

## Project structure

```
.
├── data/                        # Competition CSVs (not committed)
├── models/                      # Saved vectorizers and model weights
├── submissions/
│   ├── submission_logreg.csv         # v1 LogReg
│   ├── submission_linear_svc.csv     # v1 LinearSVC
│   ├── submission_logreg_v2.csv      # v2
│   ├── submission_logreg_v3.csv      # v3
│   ├── submission_logreg_v4.csv      # v4
│   └── submission_distilbert_sst2.csv  # v5 -- best
├── notebooks/
│   └── tutorial.ipynb           # EDA and approach walkthrough
├── solutions/
│   ├── solution.py              # v1: baseline TF-IDF + LogReg + LinearSVC
│   ├── solution_v2.py           # v2: char TF-IDF on raw text + Chinese rule
│   ├── solution_v3.py           # v3: sentence embeddings only
│   ├── solution_v4.py           # v4: TF-IDF + embeddings combined
│   ├── solution_v5.py           # v5: DistilBERT fine-tuned (best)
│   └── predict_v5.py            # inference only from saved checkpoint
├── requirements.txt
└── Dockerfile
```
