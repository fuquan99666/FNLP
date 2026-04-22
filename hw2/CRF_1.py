"""
From-scratch POS tagging with CRF implement 
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn

# Define some type aliases 
Sentence = List[str]
RawTagSeq = List[int]
RawDataset = List[Tuple[Sentence, RawTagSeq]]
EncodedSentenceFeatures = List[List[int]] # for each token, we use a list of int to represent its features
EncodedDataset = List[Tuple[EncodedSentenceFeatures, List[int]]] 
EncodedFeatureDataset = List[EncodedSentenceFeatures]

# same 
@dataclass
class EvalResult:
    dataset_name: str
    train_sents: int
    test_sents: int
    baseline_acc: float
    crf_acc: float

# set random seed for reproducibility 
def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# same 
def parse_parquet_dataset(path: Path) -> RawDataset:
    df = pd.read_parquet(path)
    if df.empty:
        return []

    data: RawDataset = []
    for _, row in df.iterrows():
        tokens = row["tokens"]
        tags = row["pos_tags"]
        if not isinstance(tokens, list) or not isinstance(tags, list):
            continue
        if len(tokens) == 0 or len(tokens) != len(tags):
            continue
        
        data.append(([str(t) for t in tokens], [int(y) for y in tags]))
    return data

# same 
def discover_dataset_pairs(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    pairs = []
    for train_file in sorted(data_dir.glob("*_train.parquet")):
        prefix = train_file.name[: -len("_train.parquet")]
        test_file = data_dir / f"{prefix}_test.parquet"
        if test_file.exists():
            pairs.append((prefix, train_file, test_file))
    return pairs

# same 
def token_accuracy(y_true: Iterable[RawTagSeq], y_pred: Iterable[RawTagSeq]) -> float:
    correct = 0
    total = 0
    for gold_seq, pred_seq in zip(y_true, y_pred):
        for g, p in zip(gold_seq, pred_seq):
            if g == p:
                correct += 1
            total += 1
    return (correct / total) if total > 0 else 0.0

# same 
class MajorityTagger:
    def __init__(self) -> None:
        self.word_to_tag: Dict[str, int] = {}
        self.default_tag = 0

    def fit(self, train_data: RawDataset) -> None:
        counts = defaultdict(Counter)
        global_counts = Counter()
        for sent, tags in train_data:
            for w, t in zip(sent, tags):
                lw = w.lower()
                counts[lw][t] += 1
                global_counts[t] += 1

        self.word_to_tag = {w: c.most_common(1)[0][0] for w, c in counts.items()}
        if global_counts:
            self.default_tag = global_counts.most_common(1)[0][0]

    def predict(self, sents: Iterable[Sentence]) -> List[RawTagSeq]:
        output: List[RawTagSeq] = []
        for sent in sents:
            output.append([self.word_to_tag.get(w.lower(), self.default_tag) for w in sent])
        return output

# same 
def word2features(sent: Sentence, i: int) -> List[str]:
    word = sent[i]
    feats = [
        "bias",
        f"w.lower={word.lower()}",
        f"w[-3:]={word[-3:]}",
        f"w[-2:]={word[-2:]}",
        f"w[:2]={word[:2]}",
        f"w[:3]={word[:3]}",
        f"w.isupper={word.isupper()}",
        f"w.istitle={word.istitle()}",
        f"w.isdigit={word.isdigit()}",
        f"w.has_hyphen={'-' in word}",
    ]

    if i == 0:
        feats.append("BOS")
    else:
        prev = sent[i - 1]
        feats.extend(
            [
                f"-1:w.lower={prev.lower()}",
                f"-1:w.istitle={prev.istitle()}",
                f"-1:w.isupper={prev.isupper()}",
            ]
        )

    if i == len(sent) - 1:
        feats.append("EOS")
    else:
        nxt = sent[i + 1]
        feats.extend(
            [
                f"+1:w.lower={nxt.lower()}",
                f"+1:w.istitle={nxt.istitle()}",
                f"+1:w.isupper={nxt.isupper()}",
            ]
        )

    return feats

# input : traindata 
# {} is a set 
# output : tag vocab 
def build_tag_vocab(train_data: RawDataset) -> Tuple[Dict[int, int], Dict[int, int]]:
    # get all unique tags in the training data and sort them 
    tags = sorted({t for _, ys in train_data for t in ys})
    # build the mapping from tag to index based on the sorted turn 
    raw_to_ix = {t: i for i, t in enumerate(tags)}
    # build the reverse mapping from index to tag 
    ix_to_raw = {i: t for t, i in raw_to_ix.items()}
    return raw_to_ix, ix_to_raw

# input : tranindata 
# output : feature vocab 
def build_feature_vocab(train_data: RawDataset, min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()

    # first count the nums of each word's features in the training data 
    for sent, _ in train_data:
        for i in range(len(sent)):
            # update can update all the elements in the list 
            counter.update(word2features(sent, i))

    
    feat_to_ix: Dict[str, int] = {}
    for feat, freq in counter.items():
        if freq >= min_freq:
            feat_to_ix[feat] = len(feat_to_ix)
    return feat_to_ix


def encode_dataset(
    data: RawDataset,
    raw_tag_to_ix: Dict[int, int],
    feat_to_ix: Dict[str, int],
) -> EncodedDataset:
    encoded: EncodedDataset = []
    for sent, raw_tags in data:
        sent_feat_ix: EncodedSentenceFeatures = []

        # loop over each token in the sentence 
        for i in range(len(sent)):
            # convert each token's each feature to int based on the prepared feat_to_ix
            feature_ids = [feat_to_ix[f] for f in word2features(sent, i) if f in feat_to_ix]
            if not feature_ids:
                # Keep at least one feature id list per token; empty list is still acceptable.
                feature_ids = []
            sent_feat_ix.append(feature_ids)

        tag_ix_seq = [raw_tag_to_ix[t] for t in raw_tags]
        encoded.append((sent_feat_ix, tag_ix_seq))
    return encoded


def encode_features_only(data: RawDataset, feat_to_ix: Dict[str, int]) -> EncodedFeatureDataset:
    encoded: EncodedFeatureDataset = []
    for sent, _ in data:
        sent_feat_ix: EncodedSentenceFeatures = []
        for i in range(len(sent)):
            feature_ids = [feat_to_ix[f] for f in word2features(sent, i) if f in feat_to_ix]
            sent_feat_ix.append(feature_ids)
        encoded.append(sent_feat_ix)
    return encoded

# The most important part : CRF model 
class LinearChainCrf(nn.Module):
    def __init__(self, num_tags: int, num_features: int):
        super().__init__()
        self.num_tags = num_tags
        self.num_features = num_features

        # state_weights : the score of each feature for each tag 
        # For example, NN : w.lower=dog = 0.5 , NN : w.lower=cat = 0.3
        self.state_weights = nn.Parameter(torch.zeros(num_tags, num_features))

        # the score between two tags (transition score)
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

        # the start score for each tag and the end score for each tag 
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self._reset_parameters()

    # Magic method to init the parameters of the model 
    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.state_weights, -0.01, 0.01)
        nn.init.uniform_(self.transitions, -0.01, 0.01)
        nn.init.uniform_(self.start_transitions, -0.01, 0.01)
        nn.init.uniform_(self.end_transitions, -0.01, 0.01)


    def emissions_for_sentence(self, sent_feature_ids: EncodedSentenceFeatures) -> torch.Tensor:
        # Returns [T, C], where C is num_tags， T is the length of the sentence 
        emission_rows = []

        # loop over each token's feature ids 
        for feat_ids in sent_feature_ids:
            if feat_ids:
                # convert the feature ids to tensor 
                feat_tensor = torch.tensor(feat_ids, dtype=torch.long, device=self.state_weights.device)
                # find the tag scores for this token's features 
                # and sum the state_weights scores for all tags 
                scores = self.state_weights.index_select(1, feat_tensor).sum(dim=1)
            else:
                scores = torch.zeros(self.num_tags, device=self.state_weights.device)
            emission_rows.append(scores)

        return torch.stack(emission_rows, dim=0)

    # given a sentence's tag list and precompute its emissions, 
    # compute the score of this tag sequence 
    def gold_score(self, emissions: torch.Tensor, tags: List[int]) -> torch.Tensor:
        # emissions: [T, C]
        # tags: a list of tag for the sentence (maybe not the correct tag sequence)
        tag0 = tags[0] 
        score = self.start_transitions[tag0] + emissions[0, tag0] # the score of the first tag is the start score of the first tag + teh state score of the first token.
        for t in range(1, len(tags)):
            prev_tag = tags[t - 1]
            cur_tag = tags[t]
            score = score + self.transitions[prev_tag, cur_tag] + emissions[t, cur_tag]
        score = score + self.end_transitions[tags[-1]] 
        return score

    def log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        # Forward algorithm in log-space.
        # This part is a little difficult to understand, 
        # maybe you need to write down some simple examples to help understand .
        alpha = self.start_transitions + emissions[0]  # [C]
        for t in range(1, emissions.size(0)):
            scores = alpha.unsqueeze(1) + self.transitions  # [C, C]
            alpha = torch.logsumexp(scores, dim=0) + emissions[t]
        alpha = alpha + self.end_transitions
        return torch.logsumexp(alpha, dim=0)

    def neg_log_likelihood_one(self, sent_feature_ids: EncodedSentenceFeatures, tags: List[int]) -> torch.Tensor:
        emissions = self.emissions_for_sentence(sent_feature_ids)
        # based on our - log P(tags|sentence) = log Z - score(sentence, tags) , 
        # this is the loss function we want to minimize.
        # this tags is exactly the correct tag sequence for this sentence in the training dataset .
        return self.log_partition(emissions) - self.gold_score(emissions, tags)

    # This funtion is used for test dataset, we only have the sentence and its features,
    # to find the best tag sequence based on the best score .
    # This viterbi algorithm is between 贪心 and dp
    def viterbi_decode_one(self, sent_feature_ids: EncodedSentenceFeatures) -> List[int]:
        emissions = self.emissions_for_sentence(sent_feature_ids)
        seq_len = emissions.size(0)

        dp = self.start_transitions + emissions[0]  # [C]
        backpointers: List[torch.Tensor] = []

        for t in range(1, seq_len):
            scores = dp.unsqueeze(1) + self.transitions  # [C, C]
            best_prev_scores, best_prev_tags = torch.max(scores, dim=0)
            dp = best_prev_scores + emissions[t]
            backpointers.append(best_prev_tags)

        dp = dp + self.end_transitions
        best_last_tag = int(torch.argmax(dp).item())

        best_path = [best_last_tag]
        for bp in reversed(backpointers):
            best_last_tag = int(bp[best_last_tag].item())
            best_path.append(best_last_tag)
        best_path.reverse()
        return best_path

def train_crf_lbfgs(
    train_encoded: EncodedDataset,
    num_tags: int,
    num_features: int,
    device: torch.device,
    epochs: int = 20,
    l2: float = 1e-4,
) -> LinearChainCrf:
    model = LinearChainCrf(num_tags=num_tags, num_features=num_features).to(device)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    for epoch in range(1, epochs + 1):
        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            losses = []
            for sent_feature_ids, tag_ix_seq in train_encoded:
                losses.append(model.neg_log_likelihood_one(sent_feature_ids, tag_ix_seq))

            nll = torch.stack(losses).mean()

            # L2 regularization ?
            reg = 0.5 * l2 * (
                model.state_weights.pow(2).sum()
                + model.transitions.pow(2).sum()
                + model.start_transitions.pow(2).sum()
                + model.end_transitions.pow(2).sum()
            )
            loss = nll + reg
            loss.backward()
            return loss

        loss_val = optimizer.step(closure)
        print(f"  [Epoch {epoch:02d}] train_obj={float(loss_val):.4f}")

    return model

# based on the trained model , given some sentences' features , 
# use the viterbi to predict the best tag sequence for each sentence 
def predict_crf(
    model: LinearChainCrf,
    encoded_data: EncodedFeatureDataset,
    ix_to_raw_tag: Dict[int, int],
) -> List[RawTagSeq]:
    model.eval()
    output: List[RawTagSeq] = []
    with torch.no_grad():
        for sent_feature_ids in encoded_data:
            pred_ix = model.viterbi_decode_one(sent_feature_ids)
            output.append([ix_to_raw_tag[i] for i in pred_ix])
    return output


def evaluate_dataset(dataset_name: str, train_path: Path, test_path: Path, device: torch.device) -> EvalResult:
    
    train_data = parse_parquet_dataset(train_path)
    test_data = parse_parquet_dataset(test_path)
    
    if not train_data or not test_data:
        raise ValueError(f"Dataset {dataset_name} has empty train/test after parsing.")

    test_sents = [s for s, _ in test_data]
    y_test = [y for _, y in test_data]

    baseline = MajorityTagger()
    baseline.fit(train_data)
    y_pred_baseline = baseline.predict(test_sents)
    baseline_acc = token_accuracy(y_test, y_pred_baseline)

    # build tag vocab 
    raw_tag_to_ix, ix_to_raw_tag = build_tag_vocab(train_data)

    # build feature vocab , only keep the features that appear at least once in the training dataset 
    feat_to_ix = build_feature_vocab(train_data, min_freq=1)

    # we all know that training need compute , so we should convert all this data into int 
    # get encoded training data
    train_encoded = encode_dataset(train_data, raw_tag_to_ix, feat_to_ix)

    # get only feature encoded test data 
    test_encoded = encode_features_only(test_data, feat_to_ix)

    model = train_crf_lbfgs(
        train_encoded=train_encoded,
        num_tags=len(raw_tag_to_ix),
        num_features=len(feat_to_ix),
        device=device,
        epochs=20,
        l2=1e-4,
    )

    y_pred_crf = predict_crf(model, test_encoded, ix_to_raw_tag)
    crf_acc = token_accuracy(y_test, y_pred_crf)

    return EvalResult(
        dataset_name=dataset_name,
        train_sents=len(train_data),
        test_sents=len(test_data),
        baseline_acc=baseline_acc,
        crf_acc=crf_acc,
    )

# same
def print_results_table(results: Sequence[EvalResult]) -> None:
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

# same 
def main() -> None:
    set_seed(42)
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    pairs = discover_dataset_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError("No *_train.parquet and *_test.parquet pairs found in ./data")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    results: List[EvalResult] = []
    for name, train_path, test_path in pairs:
        print(f"\n[INFO] Running dataset: {name}")
        result = evaluate_dataset(name, train_path, test_path, device)
        results.append(result)

    print("\n=== POS Tagging Results (Token Accuracy) ===")
    print_results_table(results)


if __name__ == "__main__":
    main()