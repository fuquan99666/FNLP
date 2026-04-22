# In this file, we implement the LSTM-CRF model for pos tagging.


from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset


Sentence = List[str]
TagSeq = List[int]
RawDataset = List[Tuple[Sentence, TagSeq]]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = -100


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

		sent = [str(t) for t in tokens]
		tag_seq = [int(t) for t in tags]
		data.append((sent, tag_seq))
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
def token_accuracy(y_true: Iterable[TagSeq], y_pred: Iterable[TagSeq]) -> float:
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

	def predict(self, sents: Iterable[Sentence]) -> List[TagSeq]:
		out: List[TagSeq] = []
		for sent in sents:
			out.append([self.word_to_tag.get(w.lower(), self.default_tag) for w in sent])
		return out

# compute the nums of the word and convert the word to index according to its nums .
def build_word_vocab(data: RawDataset, min_freq: int = 1) -> Dict[str, int]:
	counter = Counter()
	for sent, _ in data:
		counter.update(w.lower() for w in sent)

	vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
	for w, c in counter.items():
		if c >= min_freq:
			vocab[w] = len(vocab)
	return vocab


def build_tag_vocab(data: RawDataset) -> Tuple[Dict[int, int], Dict[int, int]]:
	all_tags = sorted({t for _, tags in data for t in tags})
	tag_to_ix = {tag: i for i, tag in enumerate(all_tags)}
	ix_to_tag = {i: tag for tag, i in tag_to_ix.items()}
	return tag_to_ix, ix_to_tag


class EncodedPosDataset(TorchDataset):
	def __init__(self, data: RawDataset, word_to_ix: Dict[str, int], tag_to_ix: Dict[int, int]):
		self.samples = []
		unk = word_to_ix[UNK_TOKEN]
		for sent, tags in data:
			x = [word_to_ix.get(w.lower(), unk) for w in sent]
			y = [tag_to_ix[t] for t in tags]
			self.samples.append((x, y))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		return self.samples[idx]


def collate_batch(batch: Sequence[Tuple[List[int], List[int]]]):
	lengths = [len(x) for x, _ in batch]
	max_len = max(lengths)
	bsz = len(batch)

	tokens = torch.zeros((bsz, max_len), dtype=torch.long)
	tags = torch.full((bsz, max_len), fill_value=PAD_TAG, dtype=torch.long)
	mask = torch.zeros((bsz, max_len), dtype=torch.bool)

	for i, (x, y) in enumerate(batch):
		n = len(x)
		tokens[i, :n] = torch.tensor(x, dtype=torch.long)
		tags[i, :n] = torch.tensor(y, dtype=torch.long)
		mask[i, :n] = True

	return tokens, tags, mask, lengths


def log_sum_exp(tensor: torch.Tensor, dim: int) -> torch.Tensor:
	max_score, _ = tensor.max(dim)
	return max_score + torch.log(torch.sum(torch.exp(tensor - max_score.unsqueeze(dim)), dim))


class CRFLayer(nn.Module):
	"""Linear-chain CRF with manual forward algorithm and Viterbi decoding."""

	def __init__(self, num_tags: int):
		super().__init__()
		self.num_tags = num_tags
		self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
		self.start_transitions = nn.Parameter(torch.empty(num_tags))
		self.end_transitions = nn.Parameter(torch.empty(num_tags))
		self.reset_parameters()

	def reset_parameters(self) -> None:
		nn.init.uniform_(self.transitions, -0.1, 0.1)
		nn.init.uniform_(self.start_transitions, -0.1, 0.1)
		nn.init.uniform_(self.end_transitions, -0.1, 0.1)

	def score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		# emissions: [B, T, C], tags: [B, T], mask: [B, T]
		bsz, seq_len, _ = emissions.shape
		score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0:1]).squeeze(1)

		for t in range(1, seq_len):
			valid = mask[:, t]
			prev_tag = tags[:, t - 1]
			cur_tag = tags[:, t]
			trans_score = self.transitions[prev_tag, cur_tag]
			emit_score = emissions[:, t, :].gather(1, cur_tag.unsqueeze(1)).squeeze(1)
			score = score + (trans_score + emit_score) * valid

		lengths = mask.long().sum(dim=1) - 1
		last_tags = tags.gather(1, lengths.unsqueeze(1)).squeeze(1)
		score = score + self.end_transitions[last_tags]
		return score

	def compute_log_partition(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		# alpha: [B, C]
		alpha = self.start_transitions + emissions[:, 0, :]
		seq_len = emissions.size(1)

		for t in range(1, seq_len):
			emit_t = emissions[:, t, :].unsqueeze(1)  # [B, 1, C]
			score_t = alpha.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t  # [B, C, C]
			next_alpha = log_sum_exp(score_t, dim=1)  # [B, C]
			mask_t = mask[:, t].unsqueeze(1)
			alpha = torch.where(mask_t, next_alpha, alpha)

		alpha = alpha + self.end_transitions
		return log_sum_exp(alpha, dim=1)

	def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		gold_score = self.score_sentence(emissions, tags, mask)
		log_z = self.compute_log_partition(emissions, mask)
		return (log_z - gold_score).mean()

	def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
		# Viterbi per instance for clarity and correctness.
		bsz, seq_len, num_tags = emissions.shape
		best_paths: List[List[int]] = []

		for i in range(bsz):
			length = int(mask[i].sum().item())
			emit = emissions[i, :length, :]  # [L, C]
			dp = self.start_transitions + emit[0]  # [C]
			backpointers = []

			for t in range(1, length):
				scores = dp.unsqueeze(1) + self.transitions  # [C, C]
				best_prev_scores, best_prev_tags = scores.max(dim=0)
				dp = best_prev_scores + emit[t]
				backpointers.append(best_prev_tags)

			dp = dp + self.end_transitions
			best_last_tag = int(torch.argmax(dp).item())

			best_path = [best_last_tag]
			for bp in reversed(backpointers):
				best_last_tag = int(bp[best_last_tag].item())
				best_path.append(best_last_tag)
			best_path.reverse()
			best_paths.append(best_path)

		return best_paths


class BiLstmCrf(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		num_tags: int,
		emb_dim: int = 128,
		hidden_dim: int = 256,
		dropout: float = 0.2,
	):
		super().__init__()
		self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
		self.lstm = nn.LSTM(
			emb_dim,
			hidden_dim // 2,
			num_layers=1,
			batch_first=True,
			bidirectional=True,
		)
		self.dropout = nn.Dropout(dropout)
		self.proj = nn.Linear(hidden_dim, num_tags)
		self.crf = CRFLayer(num_tags)

	def emissions(self, tokens: torch.Tensor) -> torch.Tensor:
		x = self.embedding(tokens)
		x, _ = self.lstm(x)
		x = self.dropout(x)
		return self.proj(x)

	def loss(self, tokens: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
		emit = self.emissions(tokens)
		return self.crf.neg_log_likelihood(emit, tags, mask)

	def predict(self, tokens: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
		emit = self.emissions(tokens)
		return self.crf.decode(emit, mask)


def predict_model(
	model: BiLstmCrf,
	data_loader: DataLoader,
	device: torch.device,
	ix_to_tag: Dict[int, int],
) -> List[TagSeq]:
	model.eval()
	y_pred: List[TagSeq] = []
	with torch.no_grad():
		for tokens, _, mask, _ in data_loader:
			tokens = tokens.to(device)
			mask = mask.to(device)
			batch_pred = model.predict(tokens, mask)
			for seq in batch_pred:
				y_pred.append([ix_to_tag[ix] for ix in seq])
	return y_pred


def train_model(
	model: BiLstmCrf,
	train_loader: DataLoader,
	device: torch.device,
	epochs: int = 5,
	lr: float = 1e-3,
) -> None:
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	model.to(device)

	for epoch in range(1, epochs + 1):
		model.train()
		total_loss = 0.0
		for tokens, tags, mask, _ in train_loader:
			tokens = tokens.to(device)
			tags = tags.to(device)
			mask = mask.to(device)

			loss = model.loss(tokens, tags, mask)
			optimizer.zero_grad()
			loss.backward()
			nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
			optimizer.step()
			total_loss += float(loss.item())

		avg_loss = total_loss / max(len(train_loader), 1)
		print(f"  [Epoch {epoch:02d}] train_loss={avg_loss:.4f}")


def evaluate_dataset(dataset_name: str, train_path: Path, test_path: Path, device: torch.device) -> EvalResult:
	train_data = parse_parquet_dataset(train_path)
	test_data = parse_parquet_dataset(test_path)
	if not train_data or not test_data:
		raise ValueError(f"Dataset {dataset_name} has empty train/test after parsing.")

	# Baseline
	y_test = [tags for _, tags in test_data]
	test_sents = [sent for sent, _ in test_data]
	baseline = MajorityTagger()
	baseline.fit(train_data)
	y_pred_baseline = baseline.predict(test_sents)
	baseline_acc = token_accuracy(y_test, y_pred_baseline)

	# CRF model
	word_to_ix = build_word_vocab(train_data)
	tag_to_ix, ix_to_tag = build_tag_vocab(train_data)

	train_ds = EncodedPosDataset(train_data, word_to_ix, tag_to_ix)
	test_ds = EncodedPosDataset(test_data, word_to_ix, tag_to_ix)

	train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_batch)
	test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_batch)

	model = BiLstmCrf(vocab_size=len(word_to_ix), num_tags=len(tag_to_ix))
	train_model(model, train_loader, device=device, epochs=5, lr=1e-3)
	y_pred_crf = predict_model(model, test_loader, device=device, ix_to_tag=ix_to_tag)
	crf_acc = token_accuracy(y_test, y_pred_crf)

	return EvalResult(
		dataset_name=dataset_name,
		train_sents=len(train_data),
		test_sents=len(test_data),
		baseline_acc=baseline_acc,
		crf_acc=crf_acc,
	)

# same 
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
		result = evaluate_dataset(name, train_path, test_path, device=device)
		results.append(result)

	print("\n=== POS Tagging Results (Token Accuracy) ===")
	print_results_table(results)


if __name__ == "__main__":
	main()
