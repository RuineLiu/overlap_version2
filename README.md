# overlap_version2

Sample selection for StruGene, a deep learning-based syntactic disambiguation model.

**Abstract**
This project studies syntactic ambiguity caused by overlapping phrases in Chinese and investigates how sample selection strategies affect the generalization of a deep learning model. A structured dataset is built from CCG-style phrase structures, and a Transformer-based classifier (StruGeneTransformer) is trained to predict disambiguation preferences: left-prefer (Lp), right-prefer (Rp), or no preference (Noth). The repository includes (1) a feature-based Transformer over engineered structural features, (2) a token-based Transformer baseline over linearized structure tokens, and (3) scripts to compare sampling strategies such as label balance and structural diversity.

**Keywords**: syntactic ambiguity, overlapping phrases, Transformer, StruGene, sample selection, CCG

## Research Motivation
Overlapping phrase structures are a major source of syntactic ambiguity. Traditional rule-based or linear statistical methods struggle to capture long-distance structural dependencies. StruGene provides a structure-aware disambiguation framework, but its performance is highly sensitive to the coverage and balance of training samples. This work treats sample selection as a first-class research variable and evaluates its impact on model performance.

## Contributions
- Build a structured dataset from CCG-based phrase overlap contexts.
- Design a feature extraction pipeline that captures phrase structure, context extensions, and overlap type.
- Propose StruGeneTransformer: a compact Transformer encoder over structural features.
- Compare sample selection strategies (original distribution, label-balanced, structural diversity).
- Provide a token-based Transformer baseline for sequence-style modeling of structure.

## Method Overview
**Data representation**
- Each instance is a 5-tuple of structure context: `LeftExtend`, `LeftOver`, `RightOver`, `RightExtend`, `OverType`.
- Labels are derived from `clauTagPrior`: `Lp`, `Rp`, `Noth`.

**Model variants**
1) **Feature-based StruGeneTransformer**  
   Uses engineered numeric features (hashes of structural tuples, one-hot overlap type, hit counts, context statistics).
2) **Token-based Transformer baseline**  
   Linearizes structure into tokens such as `[LEFT_EXT] ... [LEFT_OVER] ... [RIGHT_OVER] ... [RIGHT_EXT] ... [OT]`.

**Sample selection strategies**
- Original distribution (baseline).
- Label-balanced sampling.
- Structural diversity sampling (balanced by overlap type).

## Dataset
The main dataset is `data/stru_gene_202501.json`, containing **22,586** labeled samples.

**Key fields**
- `leftOver`, `rightOver`: structural tuples (category, rule, phrase type, span).
- `leftExtend`, `rightExtend`: context lists of structural tuples.
- `overType`: overlap type (0..4).
- `clauTagPrior`: disambiguation preference label(s).

**Label mapping**
- `Lp`: left preference  
- `Rp`: right preference  
- `Noth`: no preference

## Repository Layout
- `data/` raw dataset and evaluation CSVs
- `processed/` processed feature and token datasets
- `preprocess/` feature extraction and text input preparation
- `models/` Transformer architectures
- `tokenizer/` vocabulary builder and tokenizer
- `train/` training scripts and saved outputs
- `conparison_test/` sample selection experiments and metrics
- `predict.py` evaluation and prediction utility

## Quickstart
All scripts currently use **Windows-style absolute paths**. Update the paths to your local environment before running.

### 1) Feature-based pipeline
Build numeric features:
```bash
python preprocess/preprocess_2.py
```

Train StruGeneTransformer:
```bash
python train/train.py
```

Evaluate:
```bash
python predict.py
```

### 2) Token-based Transformer baseline
Build vocabulary:
```bash
python tokenizer/build_vocab.py
```

Prepare text inputs:
```bash
python preprocess/prepare_transformer_input.py
```

Train baseline model:
```bash
python train/train_classifier.py
```

### 3) Sample selection experiments
Generate sampling variants:
```bash
python conparison_test/sample_strategy_generator.py
```

Run comparison training:
```bash
python conparison_test/run_experiments.py
```

## Results (from the accompanying thesis)
Experiments show strong dependence on sampling strategy:
- **Label-balanced sampling** yields recall/accuracy/F1 above 99%.
- **Structural diversity sampling** performs poorly (often below 60%), indicating instability.

These findings support the hypothesis that balanced label coverage is crucial for StruGene-style disambiguation.

## Example (Bilingual)
**Chinese example**: “七二届学生代表”  
Ambiguity: “七二届” may modify “学生” or “学生代表”.  
模型目标：判断结构优先组合方向（Lp / Rp / Noth）。

**Tokenized input format**
```
[LEFT_EXT] ... [LEFT_OVER] ... [RIGHT_OVER] ... [RIGHT_EXT] ... [OT] 2
```

## Environment
- Python 3.8+
- PyTorch
- pandas, numpy, scikit-learn, matplotlib

## License
See `LICENSE`.
