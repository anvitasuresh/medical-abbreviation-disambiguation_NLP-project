# Medical Abbreviation Disambiguation

Disambiguating medical abbreviations as a multi-class classification problem, comparing a Multinomial Naive Bayes classifier built from scratch against a neural network using pre-trained BioWordVec embeddings. Both models are evaluated across three types of data — template synthetic, NB-generated synthetic, and real medical abstracts — to understand how natural linguistic structure affects performance.

---

## The Problem

Medical abbreviations are highly ambiguous. The same two letters can refer to completely different clinical concepts depending on context:

| Abbreviation | Possible meanings |
|---|---|
| **CC** | colorectal cancer · cell culture · cervical cancer |
| **CP** | chronic pain · chest pain · cerebral palsy |
| **SA** | surface area · sleep apnea · substance abuse |

Misreading an abbreviation in a clinical note can have serious consequences. This project treats disambiguation as a 9-class classification problem — predicting the correct expansion of CC, CP, or SA from its surrounding context.

---

## Data

Three datasets are used, each representing a different level of linguistic complexity:

| Dataset | Source | Description |
|---|---|---|
| **Template synthetic** | `generate_synthetic.py` | Sentences built from hand-crafted templates with discriminative keywords (e.g., "CPAP", "colon", "addiction") inserted around the abbreviation |
| **NB-generated synthetic** | `generate_NB_synthetic.py` | Sentences sampled from probability distributions learned by a Naive Bayes model — more varied than templates but still artificial |
| **Real** | [MeDAL dataset](https://github.com/BaderLab/MeDAL) | Actual medical abstracts from PubMed, filtered to sentences containing CC, CP, or SA |

The gap in performance between synthetic and real data reveals what the models learned vs. what they memorized.

---

## Models

### Multinomial Naive Bayes (from scratch)
Located in `final_project/bayes_evaluation/`

- Built from scratch using NumPy — no sklearn classifiers
- Uses bag-of-words features with Laplace smoothing (α=1.0)
- Also evaluated with TF-IDF weighting
- Compared against a most-frequent-class baseline

### Mean Pooling Neural Network with BioWordVec
Located in `final_project/nn_evaluation/nn.py`

- Context words around each abbreviation (window size = 5) are looked up in pre-trained **BioWordVec** embeddings (200-dim, trained on PubMed + clinical notes)
- Embeddings are **mean-pooled** into a single context vector
- Fed into a two-hidden-layer feedforward network: `200 → 512 → 256 → 9`
- ReLU activations, softmax output, cross-entropy loss
- Mini-batch gradient descent with learning rate decay and early stopping on validation accuracy
- All implemented from scratch in NumPy

### Attention Neural Network
Located in `final_project/nn_evaluation/nn_attn.py`

- Same BioWordVec embeddings but uses **attention weighting** instead of mean pooling — learns which context words matter most for each abbreviation

---

## Project Structure

```
final_project/
├── preprocessing/
│   ├── filter_data.py           # filters MeDAL dataset to CC/CP/SA sentences
│   ├── generate_synthetic.py    # template-based synthetic data generation
│   ├── generate_NB_synthetic.py # NB-sampled synthetic data generation
│   └── testNB_NBsyn.py          # validates NB synthetic data quality
├── bayes_evaluation/
│   ├── main.py                  # runs NB experiments across all datasets
│   ├── models.py                # MostFrequentBaseline + MultinomialNB from scratch
│   ├── data_loader.py           # loads synthetic and real datasets
│   ├── feature_extraction.py    # bag-of-words feature extraction
│   ├── tfidf.py                 # TF-IDF weighting
│   ├── evaluation.py            # accuracy, precision, recall, F1
│   └── analysis.py              # error analysis and per-class breakdown
├── nn_evaluation/
│   ├── nn.py                    # mean pooling NN with BioWordVec (full pipeline)
│   └── nn_attn.py               # attention-based variant
├── data/                        # generated datasets (not tracked)
├── report/                      # written analysis
└── NLP_Final_Project_Report.pdf # final report
```

---

## Key Findings

- **Real medical text is harder than synthetic** — both models drop substantially in accuracy when moving from template-generated to real MeDAL abstracts
- **Natural linguistic structure is exploitable** — predictable word sequences, grammatical constraints, and semantic coherence in real text give both models signal beyond keyword matching
- **NB is competitive on synthetic data** but struggles more on real data, where word independence assumptions break down
- **BioWordVec NN generalizes better** on real data due to semantic embeddings trained on medical literature — words not seen in training still carry useful signal via similar embeddings
- **NB-generated synthetic data bridges the gap** somewhat between template and real data, but does not fully capture the complexity of natural text

---

## Tech Stack

Python · NumPy · BioWordVec · MeDAL dataset · scikit-learn (evaluation only)
