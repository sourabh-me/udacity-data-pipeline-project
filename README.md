# Fashion Forward Forecasting — StyleSense Review Recommendation Model

StyleSense, a fast-growing online women's clothing retailer, has a backlog of
product reviews where customers wrote text feedback but did not indicate whether
they recommend the product. This project builds a **machine learning pipeline**
that predicts the missing `Recommended IND` label (1 = recommended,
0 = not recommended) from the review text, the customer's age, the product
category, and other review metadata.

The entire workflow — numerical scaling, categorical encoding, spaCy-based NLP
feature extraction, and the classifier — lives inside a single scikit-learn
`Pipeline`, so the exact same object is used for training and for inference on
new reviews.

## Repository Contents

```
├── README.md                  <- This file
├── requirements.txt           <- Python dependencies
└── starter/
    ├── starter.ipynb          <- Main notebook: EDA, pipeline, tuning, evaluation
    └── data/
        └── reviews.csv        <- Anonymized customer reviews dataset (18,442 rows)
```

## Getting Started

### Dependencies

- Python 3.9+
- scikit-learn, pandas, numpy, spacy, matplotlib, notebook

### Installation

```bash
# Clone the repository
git clone <this-repo-url>
cd dsnd-pipelines-project

# Install requirements
python -m pip install -r requirements.txt

# Download the spaCy English model used for lemmatization
python -m spacy download en_core_web_sm

# Launch the notebook
jupyter notebook starter/starter.ipynb
```

Note (Apple Silicon Macs): you may need `python -m pip install 'spacy[apple]'`
before installing the rest of the requirements — see https://spacy.io/usage.

## Approach

**Data.** 18,442 complete reviews with 8 features: `Clothing ID`, `Age`, `Title`,
`Review Text`, `Positive Feedback Count`, `Division Name`, `Department Name`,
`Class Name`. The target is imbalanced (~82% recommended), so evaluation relies
on precision/recall/F1 rather than accuracy alone. A 90/10 train/test split is
made up front; all exploration and tuning use only the training split.

**Pipeline.** A `ColumnTransformer` with three branches:

| Branch | Columns | Processing |
|---|---|---|
| Numerical | `Age`, `Positive Feedback Count` | `StandardScaler` |
| Categorical | `Division Name`, `Department Name`, `Class Name` | `OneHotEncoder(handle_unknown='ignore')` |
| Text | `Title` + `Review Text` | spaCy tokenization, stop-word removal & lemmatization → TF-IDF (1–2 grams), plus engineered features (character count, word count, exclamation count) |

`Clothing ID` is excluded: it is a high-cardinality product identifier that would
encourage memorization rather than generalization.

**Model & tuning.** Logistic regression, fine-tuned with `GridSearchCV`
(3-fold cross-validation, F1 scoring) over regularization strength `C` and
`class_weight` to address the class imbalance.

**Evaluation.** The best pipeline is evaluated once on the held-out test set
with accuracy, precision, recall, F1, ROC AUC, a classification report, and a
confusion matrix. The notebook also inspects the learned coefficients to show
which review terms drive recommendations vs. non-recommendations.

## Built With

* [scikit-learn](https://scikit-learn.org) - Pipeline, preprocessing, model & tuning
* [spaCy](https://spacy.io) - Tokenization, stop-word removal, lemmatization
* [pandas](https://pandas.pydata.org) / [NumPy](https://numpy.org) - Data handling
* [Matplotlib](https://matplotlib.org) - Visualizations

## License

[License](LICENSE.txt)
