"""
We will implement a CRF Pos tagging and compare it to a simple majority-count baseline.

1. Loads all `*_train.parquet` and `*_test.parquet` pairs under `./data/`.
2. Trains a linear-chain CRF for POS tagging for three datasets.
3. Trains a majority-count baseline (word -> most frequent POS in training).
4. Evaluates and prints token-level accuracy for both methods.

"""

### Some package that we will use 

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

try:
    import sklearn_crfsuite
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency `sklearn-crfsuite`. "
        "Install it with: pip install sklearn-crfsuite"
    ) from exc

# define some type aliases 
Sentence = List[str]
TagSeq = List[str]
Dataset = List[Tuple[Sentence, TagSeq]] # a list of (sentence, tag sequence) pairs 

# Some candidate column names for parse the parquet files. 
# after use help.py to peek the data, we find the correct column names 
# are [id, tokens, pos_tags]
TOKEN_COL_CANDIDATES = [
    "tokens",
]

TAG_COL_CANDIDATES = [
    "pos_tags",
]

GROUP_COL_CANDIDATES = [
    "id",
]


@dataclass
class EvalResult:
    dataset_name: str
    train_sents: int
    test_sents: int
    baseline_acc: float # baseline acc
    crf_acc: float # crf acc 

def convert_to_str(x: List):
    return [str(e) for e in x]


def _extract_sentence_level(df: pd.DataFrame, token_col: str, tag_col: str) -> Dataset:
    data: Dataset = []
    for _, row in df.iterrows():
        tokens = row[token_col]
        tags = row[tag_col]
        
        # convert the tokens and tags to string if they are not 
        if not isinstance(tokens[0], str):
            tokens = convert_to_str(tokens)
        if not isinstance(tags[0], str):
            tags = convert_to_str(tags)

        data.append((tokens, tags))
    return data


def parse_parquet_dataset(path: Path) -> Dataset:
    df = pd.read_parquet(path)
    if df.empty:
        return []

    # df.columns is like Index(['id', 'tokens', 'pos_tags'], dtype='object')
    # token_col = _choose_col(df.columns, TOKEN_COL_CANDIDATES)
    # tag_col = _choose_col(df.columns, TAG_COL_CANDIDATES)
    
    # directly use the correct column names 
    token_col = 'tokens'
    tag_col = 'pos_tags'

    # just read the data in sentence-level
    return _extract_sentence_level(df, token_col, tag_col)

# Define the state features for one word in the sentence.
# trans features are automatically handled by sklearn-crfsuite.
def word2features(sent: Sentence, i: int) -> Dict[str, object]:

    # build the feature dict for the i-th word in the sentence.

    word = sent[i]
    features: Dict[str, object] = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[:2]": word[:2],
        "word[:3]": word[:3],
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.has_hyphen": "-" in word,
    }

    if i == 0:
        features["BOS"] = True
    else:
        prev = sent[i - 1]
        features.update(
            {
                "-1:word.lower": prev.lower(),
                "-1:word.istitle": prev.istitle(),
                "-1:word.isupper": prev.isupper(),
            }
        )

    if i == len(sent) - 1:
        features["EOS"] = True
    else:
        nxt = sent[i + 1]
        features.update(
            {
                "+1:word.lower": nxt.lower(),
                "+1:word.istitle": nxt.istitle(),
                "+1:word.isupper": nxt.isupper(),
            }
        )

    return features


def sent2features(sent: Sentence) -> List[Dict[str, object]]:
    return [word2features(sent, i) for i in range(len(sent))]


def token_accuracy(y_true: Iterable[TagSeq], y_pred: Iterable[TagSeq]) -> float:
    correct = 0
    total = 0
    for gold_seq, pred_seq in zip(y_true, y_pred):
        n = min(len(gold_seq), len(pred_seq))
        for i in range(n):
            if gold_seq[i] == pred_seq[i]:
                correct += 1
        total += len(gold_seq)
    return (correct / total) if total > 0 else 0.0


class MajorityTagger:
    def __init__(self) -> None:
        self.word_to_tag: Dict[str, str] = {}
        self.default_tag = "0"

    def fit(self, train_data: Dataset) -> None:
        counts = defaultdict(Counter)
        global_counts = Counter()

        for sent, tags in train_data:
            for w, t in zip(sent, tags):
                counts[w.lower()][t] += 1
                global_counts[t] += 1

        self.word_to_tag = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        if global_counts:
            self.default_tag = global_counts.most_common(1)[0][0]

    def predict(self, sents: Iterable[Sentence]) -> List[TagSeq]:
        out: List[TagSeq] = []
        for sent in sents:
            out.append([self.word_to_tag.get(w.lower(), self.default_tag) for w in sent])
        return out


def train_crf(train_data: Dataset):
    X_train = [sent2features(sent) for sent, _ in train_data]
    y_train = [tags for _, tags in train_data]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(X_train, y_train)
    return crf


def evaluate_dataset(dataset_name: str, train_path: Path, test_path: Path) -> EvalResult:
    train_data = parse_parquet_dataset(train_path)
    test_data = parse_parquet_dataset(test_path)

    if not train_data or not test_data:
        raise ValueError(f"Dataset {dataset_name} has empty train/test after parsing.")

    test_sents = [sent for sent, _ in test_data]
    y_test = [tags for _, tags in test_data]

    baseline = MajorityTagger()
    baseline.fit(train_data)
    y_pred_baseline = baseline.predict(test_sents)
    baseline_acc = token_accuracy(y_test, y_pred_baseline)

    crf = train_crf(train_data)
    X_test = [sent2features(sent) for sent in test_sents]
    y_pred_crf = crf.predict(X_test)
    crf_acc = token_accuracy(y_test, y_pred_crf)

    return EvalResult(
        dataset_name=dataset_name,
        train_sents=len(train_data),
        test_sents=len(test_data),
        baseline_acc=baseline_acc,
        crf_acc=crf_acc,
    )


def discover_dataset_pairs(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs = []
    for train_file in sorted(data_dir.glob("*_train.parquet")):
        prefix = train_file.name[: -len("_train.parquet")]
        test_file = data_dir / f"{prefix}_test.parquet"
        if test_file.exists():
            pairs.append((prefix, train_file, test_file))
    return pairs


def print_results_table(results: List[EvalResult]) -> None:
    header = (
        f"{'Dataset':<24} {'TrainSents':>10} {'TestSents':>10} "
        f"{'MajorityAcc':>12} {'CRFAcc':>10} {'Delta':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        delta = r.crf_acc - r.baseline_acc
        print(
            f"{r.dataset_name:<24} {r.train_sents:>10d} {r.test_sents:>10d} "
            f"{r.baseline_acc:>12.4f} {r.crf_acc:>10.4f} {delta:>10.4f}"
        )


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs = discover_dataset_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError("No *_train.parquet and *_test.parquet pairs found in ./data")

    results: List[EvalResult] = []
    for name, train_path, test_path in pairs:
        print(f"\n[INFO] Running dataset: {name}")
        result = evaluate_dataset(name, train_path, test_path)
        results.append(result)

    print("\n=== POS Tagging Results (Token Accuracy) ===\n")
    print_results_table(results)


if __name__ == "__main__":
    main()

