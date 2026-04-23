"""
From-scratch POS tagging with CRF implement 
"""

from __future__ import annotations
import math

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if len(tokens) == 0 or len(tokens) != len(tags):
            print(f"UUUUUU")
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

    
        encoded.append((sent_feat_ix, raw_tags))
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


def prepare_batched_training_tensors(
    train_encoded: EncodedDataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not train_encoded:
        raise ValueError("train_encoded is empty.")

    batch_size = len(train_encoded)
    lengths = [len(sent_feats) for sent_feats, _ in train_encoded]
    max_len = max(lengths)

    tags = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)

    all_feature_ids: List[int] = []
    offsets: List[int] = [0]

    for b, (sent_feats, tag_seq) in enumerate(train_encoded):
        if len(sent_feats) != len(tag_seq):
            raise ValueError("Feature length and tag length mismatch in encoded data.")

        seq_len = len(tag_seq)
        if seq_len == 0:
            raise ValueError("Found empty sentence in encoded training data.")

         # Fill the tags based on its tag sequence and set the mask to True .
        tags[b, :seq_len] = torch.tensor(tag_seq, dtype=torch.long, device=device)
        mask[b, :seq_len] = True

        for t in range(max_len):
            if t < seq_len:
                feats = sent_feats[t]
                all_feature_ids.extend(feats)
            offsets.append(len(all_feature_ids))

    # all_feature_ids is a long list of all the feature ids for all tokens in the batch.
    feature_ids = torch.tensor(all_feature_ids, dtype=torch.long, device=device)
    # offset is a list of the starting index of each token's feature ids in the all_feature_ids list.
    offsets_t = torch.tensor(offsets, dtype=torch.long, device=device)
    return feature_ids, offsets_t, tags, mask


def prepare_mini_batch_tensors(
    train_encoded: EncodedDataset,
    sample_indices: List[int],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Build one padded mini-batch (feature bags + tags + mask) for selected sentence indices.
    batch_data = [train_encoded[i] for i in sample_indices]
    return prepare_batched_training_tensors(batch_data, device)


def build_length_aware_batches(
    train_encoded: EncodedDataset,
    max_tokens_per_batch: int,
    shuffle: bool = True,
) -> List[List[int]]:
    if max_tokens_per_batch <= 0:
        raise ValueError("max_tokens_per_batch must be positive.")

    lengths = [len(feats) for feats, _ in train_encoded]
    sorted_indices = sorted(range(len(train_encoded)), key=lambda i: lengths[i])

    if shuffle:
        ## 先将chunk_size个长度相近的句子放在一起，然后打乱它们的顺序，
        ## 确保在尽量保证每个chunk内的句子长度相近，并且增加随机性。
        chunk_size = 256
        chunks = [sorted_indices[i : i + chunk_size] for i in range(0, len(sorted_indices), chunk_size)]
        perm = torch.randperm(len(chunks)).tolist()
        sorted_indices = [idx for p in perm for idx in chunks[p]]

    batches: List[List[int]] = []
    cur_batch: List[int] = []
    cur_max_len = 0

    for idx in sorted_indices:
        # current sentence's length
        sent_len = lengths[idx]
        # get the max length of current sentence and the previous sentences' max length
        next_max_len = max(cur_max_len, sent_len)
        # if we add this sentence to the current batch, how many sentences .
        next_batch_size = len(cur_batch) + 1
        # the total nums tokens 
        estimated_tokens = next_max_len * next_batch_size

        if cur_batch and estimated_tokens > max_tokens_per_batch:
            # we can't add this sentence to the current batch.
            batches.append(cur_batch)
            # reset the current batch , put idx in it .
            cur_batch = [idx]
            cur_max_len = sent_len
        else:
            cur_batch.append(idx)
            cur_max_len = next_max_len

    # The last some batches may not get max_tokens_per_batch, 
    # but we need to add this batch to the batches list .
    if cur_batch:
        batches.append(cur_batch)

    return batches

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

    def emissions_from_offsets(
        self,
        feature_ids: torch.Tensor,
        offsets: torch.Tensor,
        batch_size: int,
        max_len: int,
    ) -> torch.Tensor:
        # Embedding-bag sums feature weights for each token bag, output shape: [B, T, C].
        # A magic function.
        token_scores = F.embedding_bag(
            feature_ids,
            self.state_weights.transpose(0, 1).contiguous(), # [tags, features] -> [features, tags]
            offsets,
            mode="sum",
            include_last_offset=True,
        )
        return token_scores.view(batch_size, max_len, self.num_tags)

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

    def gold_score_batch(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: [B, T, C], tags: [B, T], mask: [B, T]
        first_tags = tags[:, 0]
        score = self.start_transitions[first_tags] + emissions[:, 0, :].gather(1, first_tags.unsqueeze(1)).squeeze(1)

        for t in range(1, emissions.size(1)):
            cur_tags = tags[:, t]
            prev_tags = tags[:, t - 1]
            trans_scores = self.transitions[prev_tags, cur_tags]
            emit_scores = emissions[:, t, :].gather(1, cur_tags.unsqueeze(1)).squeeze(1)
            score = score + (trans_scores + emit_scores) * mask[:, t].float()

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def log_partition_batch(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # emissions: [B, T, C], mask: [B, T]
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0, :]

        for t in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            # parallelly compute over batch dimension.
            next_alpha = torch.logsumexp(scores, dim=1) + emissions[:, t, :]
            # if mask[:, t] is False, that means this position is padding, we should not update alpha for this position, keep it the alpha.
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def neg_log_likelihood_batch(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        log_z = self.log_partition_batch(emissions, mask)
        gold = self.gold_score_batch(emissions, tags, mask)
        return log_z - gold

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
    feature_ids, offsets, tags, mask = prepare_batched_training_tensors(train_encoded, device)
    batch_size, max_len = tags.size(0), tags.size(1)

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
            emissions = model.emissions_from_offsets(feature_ids, offsets, batch_size, max_len)
            nll = model.neg_log_likelihood_batch(emissions, tags, mask).mean()

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

def train_crf_adam(
    train_encoded: EncodedDataset,
    num_tags: int,
    num_features: int,
    device: torch.device,
    epochs: int = 10,
    lr: float = 0.001,
    l2: float = 1e-5,
    max_tokens_per_batch: int = 12000,
) -> LinearChainCrf:
    model = LinearChainCrf(num_tags=num_tags, num_features=num_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    
    model.train()
    for epoch in range(1, epochs + 1):
        # based on the length of sentences, build batches that have similar sentence lengths,
        # and the total num of tokens is less than max_tokens_per_batch.
        batches = build_length_aware_batches(
            train_encoded=train_encoded,
            max_tokens_per_batch=max_tokens_per_batch,
            shuffle=True,
        )

        total_loss = 0.0
        total_nll = 0.0
        total_sents = 0

        for sample_indices in batches:
            # feature_ids: a long list of all the feature ids for all tokens in the batch.
            # offsets: a list of the starting index of each token's feature ids in the all feature_ids list.
            # tags: a [B, L] tensor of tag for the batch.
            # mask: a [B, L] boolean tensor indicating which positions are valid.
            feature_ids, offsets, tags, mask = prepare_mini_batch_tensors(train_encoded, sample_indices, device)
            batch_size = len(sample_indices)
            max_len = tags.size(1)

            optimizer.zero_grad()

            emissions = model.emissions_from_offsets(feature_ids, offsets, batch_size, max_len)
            nll = model.neg_log_likelihood_batch(emissions, tags, mask).mean()

            reg = 0.5 * l2 * (
                model.state_weights.pow(2).sum()
                + model.transitions.pow(2).sum()
                + model.start_transitions.pow(2).sum()
                + model.end_transitions.pow(2).sum()
            )
            loss = nll + reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_nll += nll.item() * batch_size
            total_sents += batch_size

        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        avg_loss = total_loss / max(total_sents, 1)
        avg_nll = total_nll / max(total_sents, 1)
        print(
            f"  ✅ Epoch {epoch:02d} 完成 | loss={avg_loss:.4f} | nll={avg_nll:.4f} | "
            f"batches={len(batches)} | lr={lr_now:.6f}"
        )
    
    return model


def save_crf_checkpoint(
    model: LinearChainCrf,
    path: Path,
    num_tags: int,
    num_features: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "num_tags": num_tags,
        "num_features": num_features,
    }
    torch.save(payload, path)


def load_crf_checkpoint(path: Path, device: torch.device) -> LinearChainCrf:
    payload = torch.load(path, map_location=device)
    num_tags = int(payload["num_tags"])
    num_features = int(payload["num_features"])

    model = LinearChainCrf(num_tags=num_tags, num_features=num_features).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

# based on the trained model , given some sentences' features , 
# use the viterbi to predict the best tag sequence for each sentence 
def predict_crf(
    model: LinearChainCrf,
    encoded_data: EncodedFeatureDataset,
) -> List[RawTagSeq]:
    model.eval()
    output: List[RawTagSeq] = []
    with torch.no_grad():
        for sent_feature_ids in encoded_data:
            pred_ix = model.viterbi_decode_one(sent_feature_ids)
            output.append(pred_ix)
    return output


def evaluate_dataset(
    dataset_name: str,
    train_path: Path,
    test_path: Path,
    device: torch.device,
    checkpoints_dir: Path,
    force_retrain: bool = False,
) -> EvalResult:
    
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
    max_tag = max(t for _, tag in train_data for t in tag)
    tag_num = max_tag + 1

    # build feature vocab , only keep the features that appear at least once in the training dataset 
    feat_to_ix = build_feature_vocab(train_data, min_freq=1)

    # we all know that training need compute , so we should convert all this data into int 
    # get encoded training data
    train_encoded = encode_dataset(train_data, feat_to_ix)

    # get only feature encoded test data 
    test_encoded = encode_features_only(test_data, feat_to_ix)

    ckpt_path = checkpoints_dir / f"{dataset_name}_crf.pt"

    model: LinearChainCrf
    if ckpt_path.exists() and not force_retrain:
        print(f"Load model from: {ckpt_path}")
        model = load_crf_checkpoint(ckpt_path, device)
    else:
        print(f"Start training !\n")
        model = train_crf_adam(
            train_encoded=train_encoded,
            num_tags=tag_num,
            num_features=len(feat_to_ix),
            device=device,
            epochs=30,
            l2=1e-4,
        )
        save_crf_checkpoint(
            model=model,
            path=ckpt_path,
            num_tags=tag_num,
            num_features=len(feat_to_ix),
        )
        print(f"Save model to: {ckpt_path}")

    y_pred_crf = predict_crf(model, test_encoded)
    crf_acc = token_accuracy(y_test, y_pred_crf)

    # Release cached GPU memory between datasets to reduce peak usage.
    if device.type == "cuda":
        torch.cuda.empty_cache()

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

    checkpoints_dir = project_root / "checkpoints"
    force_retrain = False

    results: List[EvalResult] = []
    for name, train_path, test_path in pairs:
        print(f"\n[INFO] Running dataset: {name}")

        result = evaluate_dataset(
            name,
            train_path,
            test_path,
            device,
            checkpoints_dir=checkpoints_dir,
            force_retrain=force_retrain,
        )
        results.append(result)

    print("\n=== POS Tagging Results (Token Accuracy) ===")
    print_results_table(results)


if __name__ == "__main__":
    main()



