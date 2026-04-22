# Neural Tokenizers vs BPE and WordPiece on AG News

A case study on implementing a **neural network-based tokenizer** and comparing it with traditional subword tokenization methods: **Byte-Pair Encoding (BPE)** and **WordPiece**.

## Project Goal

The goal of this case study is to implement a **neural tokenizer** and compare its behavior and usefulness against standard tokenization approaches used in NLP.

The project includes:

- training **BPE** and **WordPiece** tokenizers on the same corpus
- building a **character-level BiLSTM boundary predictor** as a neural tokenizer
- generating pseudo-labels for token boundaries with **SentencePiece Unigram + sampling**
- evaluating all tokenizers with:
  - intrinsic tokenization metrics
  - qualitative examples
  - a downstream **AG News text classification** task

---

## Main Idea

Traditional tokenizers such as BPE and WordPiece split text using frequency-based subword rules. In contrast, the neural tokenizer in this project is formulated as a **sequence labeling problem**.

For each word, the model predicts whether there should be a token boundary after each character.

Example:

```text
tokenizator -> [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
````

This means there is a token boundary after the 5th character:

```text
token | izator
```

So the neural tokenizer does not merge or split text using handcrafted rules. Instead, it learns boundary placement from pseudo-supervision.

---

## Dataset

This project uses the **AG News** dataset, a standard text classification benchmark with four classes:

* World
* Sports
* Business
* Sci/Tech

AG News was chosen because:

* it is large enough for tokenizer training and comparison
* it contains real-world short news texts
* it supports a meaningful downstream evaluation task

---

## Methods

### 1. BPE

BPE learns subword units by repeatedly merging frequent symbol pairs. It is simple, efficient, and widely used in NLP systems.

### 2. WordPiece

WordPiece is another subword tokenization method that constructs a vocabulary of useful units and is commonly used in transformer-based models such as BERT.

### 3. Neural tokenizer

The neural tokenizer is a **character-level BiLSTM** that predicts token boundaries inside words.

Pipeline:

1. convert a word into a sequence of character IDs
2. embed the characters
3. process them with a **bidirectional LSTM**
4. predict a boundary probability for each character
5. convert probabilities into binary boundaries

This makes the tokenizer context-sensitive at the character level.

---

## Pseudo-label generation

The neural tokenizer requires training labels. Since manual boundary annotation is expensive, this project generates pseudo-labels with **SentencePiece Unigram + sampling**.

For each word:

1. sample multiple possible Unigram segmentations
2. convert each segmentation into a boundary vector
3. average sampled boundaries
4. threshold the averaged result to obtain final binary labels

This approach gives weak supervision for the neural model and makes the project feasible in a single notebook.

---

## Evaluation Strategy

The tokenizers are compared in two ways.

### Intrinsic evaluation

This measures tokenizer behavior directly:

* vocabulary size on corpus
* average number of tokens per text
* average number of characters per token
* tokenization time
* reconstruction accuracy

### Extrinsic evaluation

This measures usefulness in a real NLP task:

* AG News text classification
* same simple classifier architecture for all tokenizers
* comparison by:

  * test loss
  * test accuracy
  * test macro F1

---

## Results

### Intrinsic and downstream results

| Tokenizer | Vocab size on corpus | Avg tokens per text | Avg chars per token | Tokenization time (sec) | Reconstruction accuracy | Tokenizer train time (sec) | Classifier train time (sec) | Test loss | Test accuracy | Test macro F1 |
| --------- | -------------------: | ------------------: | ------------------: | ----------------------: | ----------------------: | -------------------------: | --------------------------: | --------: | ------------: | ------------: |
| BPE       |                 3884 |             60.2314 |              3.3064 |                  0.6634 |                  1.0000 |                     1.6394 |                      8.2737 |    0.3676 |        0.8788 |        0.8779 |
| WordPiece |                 3516 |             62.3696 |              3.1931 |                  0.6539 |                  1.0000 |                     1.5475 |                      8.0428 |    0.3578 |        0.8838 |        0.8828 |
| Neural    |                10718 |             64.4780 |              3.0887 |                217.7505 |                  0.9988 |                    19.8633 |                      9.1365 |    0.3474 |        0.8840 |        0.8832 |

---

## Discussion of Results

### 1. Best downstream performance

The **neural tokenizer** achieved the best downstream metrics:

* **lowest test loss**: `0.3474`
* **highest test accuracy**: `0.8840`
* **highest test macro F1**: `0.8832`

However, the margin over WordPiece is very small.

### 2. Best trade-off

**WordPiece** appears to offer the best overall trade-off:

* almost the same downstream performance as the neural tokenizer
* smaller vocabulary than BPE and much smaller than the neural tokenizer
* very fast tokenization
* low training cost

### 3. Efficiency cost of the neural method

The neural tokenizer is much more expensive:

* tokenization time is **217.75 sec**, compared to about **0.65 sec** for BPE and WordPiece
* tokenizer training time is also much higher
* vocabulary on corpus is much larger

This shows that although the neural model slightly improved downstream accuracy, it did so at a very high computational cost.

### 4. Segmentation behavior

The neural tokenizer produced:

* the highest average number of tokens per text
* the shortest average token length
* slightly imperfect reconstruction accuracy (`0.9988` instead of `1.0000`)

This suggests that the model tends to segment text more aggressively than BPE or WordPiece.

### 5. General conclusion from the experiment

The results suggest that a neural boundary-based tokenizer **can be competitive** with traditional subword methods on a downstream task, but in this case the gain is extremely small relative to the additional complexity and runtime cost.

So the final practical conclusion is:

* **WordPiece** is the strongest practical baseline
* **Neural tokenization** is interesting and competitive, but not efficient enough here to clearly outperform traditional approaches in practice

---

## Key Takeaways

* A neural tokenizer can be implemented as a **character-level boundary prediction model**
* SentencePiece Unigram sampling can provide useful pseudo-labels for training
* The neural tokenizer achieved the **best classification metrics**, but only by a very small margin
* BPE and WordPiece remain much more efficient
* Among the traditional methods, **WordPiece** gave the strongest overall results in this study

---

## Repository Structure

Example project structure:

```text
.
├── README.md
├── notebooks/
│   └── neural_tokenizer.ipynb
└── poster.pdf
```

---

## How to Run

### 1. Install dependencies

```bash
pip install datasets tokenizers sentencepiece scikit-learn matplotlib pandas tqdm torch
```

### 2. Open Jupyter Notebook

```bash
jupyter notebook
```

### 3. Run the notebook

Open:

```text
notebooks/neural_tokenizer.ipynb
```

and run all cells from top to bottom.

---

## Notebook Contents

The notebook contains the following sections:

1. Setup
2. Dataset
3. Baseline tokenizers: BPE and WordPiece
4. Neural tokenizer design with BiLSTM
5. Pseudo-label generation using SentencePiece Unigram + sampling
6. Training the neural tokenizer
7. Tokenization examples
8. Quantitative comparison
9. Downstream experiment: AG News classification
10. Final summary table

---

## Limitations

This case study has several limitations:

* the neural tokenizer is trained on **pseudo-labels**, not human-annotated segmentation
* the downstream model is intentionally simple
* the neural tokenizer is slower because tokenization is performed through a character-level model
* the experiment is conducted only on **AG News**, so conclusions may not generalize to all NLP tasks

---

## Possible Future Improvements

Several directions could improve the project:

* use a stronger downstream classifier
* optimize neural tokenization speed with batching and caching
* improve pseudo-label quality
* test on additional datasets and languages
* compare against SentencePiece Unigram directly as another baseline
* replace BiLSTM with a lightweight Transformer or CNN-based segmenter

---

## Conclusion

This case study demonstrates that neural tokenization can be framed as a practical learning problem rather than a fixed rule-based process.

The proposed BiLSTM tokenizer was able to achieve slightly better downstream classification results than BPE and WordPiece. However, the improvement was very small, while the computational cost was dramatically higher.

As a result, the experiment shows two important points:

1. neural tokenization is feasible and competitive
2. traditional methods such as WordPiece remain stronger in terms of practical efficiency

In this project, **WordPiece** gave the best balance between quality and efficiency, while the **neural tokenizer** served as an interesting proof of concept for learned token boundary prediction.

