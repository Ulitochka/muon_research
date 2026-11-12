"""Semantic-ID tokenizer adapter for TIGER on top of the MuonRec-E447 data.

The framework stores each item's semantic ID as a list of ``rq_n_codebooks``
string tokens, e.g. ``["<a_236>", "<b_231>", "<c_226>"]`` in ``*.index.json``.
The MiniOneRec (causal-LM) path adds these strings as extra tokens to a text
tokenizer. TIGER instead uses a *compact* T5 vocabulary consisting only of the
semantic-ID tokens plus a padding and an EOS token.

This adapter converts the framework's string SIDs into that compact integer
vocabulary using per-codebook offsets (the same scheme as the original TIGER
tokenizer), but **without the user-ID token** and **without the extra
conflict-resolution digit** (the framework already resolves collisions when it
builds ``*.index.json`` and evaluates at the concatenated-SID level).

Vocabulary layout (for ``codebook_size == 256`` and ``n_digit == 3``)::

    0            : padding
    1   - 256    : digit 1  (<a_k>  -> 1   + k)
    257 - 512    : digit 2  (<b_k>  -> 257 + k)
    513 - 768    : digit 3  (<c_k>  -> 513 + k)
    769          : eos
"""

import ast
import json
import re
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple


class TigerSidTokenizer:
    """Maps framework semantic-ID strings to a compact integer T5 vocabulary.

    Args:
        index_path: Path to the framework ``*.index.json`` file mapping
            ``item_id -> [<a_i>, <b_j>, <c_k>, ...]``.
        codebook_size: Size of each RQ codebook (default ``256``). Token indices
            are offset per digit by this value.
        max_item_seq_len: Maximum number of history items fed to the encoder.
    """

    # Prefix letters, one per codebook digit. Extend if you use more codebooks.
    DIGIT_PREFIXES: Tuple[str, ...] = ("a", "b", "c", "d", "e", "f", "g", "h")

    def __init__(self, index_path: str, codebook_size: int = 256, max_item_seq_len: int = 20):
        self.index_path = index_path
        self.codebook_size = int(codebook_size)
        self.max_item_seq_len = int(max_item_seq_len)

        self._token_re = re.compile(r"<([a-z])_(\d+)>")
        self.padding_token = 0

        # Determine n_digit from the first entry of the index file.
        raw = self._read_index(index_path)
        first_codes = next(iter(raw.values()))
        self.n_digit = len(first_codes)

        # Per-digit offsets: digit d occupies [1 + d*cb, 1 + (d+1)*cb).
        self.offsets: List[int] = [1 + d * self.codebook_size for d in range(self.n_digit)]
        self.eos_token = 1 + self.n_digit * self.codebook_size

        # Build item -> integer-token-tuple mapping and the set of valid items.
        self.item2tokens: Dict[str, Tuple[int, ...]] = {}
        self.tokens2sid_string: Dict[Tuple[int, ...], str] = {}
        valid = set()

        for item_id, codes in raw.items():
            toks = self.sid_tokens_to_ids(codes)
            self.item2tokens[str(item_id)] = toks
            valid.add(toks)
            self.tokens2sid_string[toks] = "".join(codes)
        self.valid_token_tuples = valid

        # Build a prefix trie for constrained decoding.
        self._trie = self._build_trie(valid)

    # ------------------------------------------------------------------ #
    # Static / helper                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read_index(index_path: str) -> Dict[str, List[str]]:
        with open(index_path, "r") as f:
            return json.load(f)

    def _digit_of(self, letter: str) -> int:
        return self.DIGIT_PREFIXES.index(letter)

    # ------------------------------------------------------------------ #
    # Encoding / decoding                                                #
    # ------------------------------------------------------------------ #
    def sid_tokens_to_ids(self, codes: Sequence[str]) -> Tuple[int, ...]:
        """Convert a list like ``["<a_1>", "<b_2>", "<c_3>"]`` to integer ids."""
        assert len(codes) == self.n_digit, f"expected {self.n_digit} codes, got {codes!r}"
        ids: List[int] = []

        for i, code in enumerate(codes):
            m = self._token_re.fullmatch(code.strip())
            assert m is not None, f"malformed sem-id token: {code!r}"
            letter, num = m.group(1), int(m.group(2))
            d = self._digit_of(letter)
            assert d == i, f"digit order mismatch in {codes!r} (token {code!r} at position {i})"
            assert 0 <= num < self.codebook_size, f"code index {num} >= codebook_size {self.codebook_size}"
            ids.append(self.offsets[i] + num)

        return tuple(ids)

    def sid_string_to_ids(self, sid_string: str) -> Tuple[int, ...]:
        """Convert a concatenated SID string like ``"<a_1><b_2><c_3>"`` to ids."""
        codes = ["<%s_%s>" % (m.group(1), m.group(2)) for m in self._token_re.finditer(sid_string)]

        return self.sid_tokens_to_ids(codes)

    def ids_to_sid_string(self, ids: Sequence[int]) -> str:
        """Inverse of :meth:`sid_string_to_ids`."""
        parts: List[str] = []
        for i, tid in enumerate(ids):
            num = int(tid) - self.offsets[i]
            parts.append(f"<{self.DIGIT_PREFIXES[i]}_{num}>")

        return "".join(parts)

    # ------------------------------------------------------------------ #
    # Constrained decoding                                               #
    # ------------------------------------------------------------------ #
    def _build_trie(self, valid_tuples) -> Dict[Tuple[int, ...], List[int]]:
        trie = defaultdict(set)
        for tup in valid_tuples:
            for depth in range(self.n_digit):
                trie[tup[:depth]].add(tup[depth])

        return {k: sorted(v) for k, v in trie.items()}

    def build_prefix_allowed_tokens_fn(self, decoder_start_token_id: int = 0):
        """Return a ``prefix_allowed_tokens_fn`` for constrained beam search.

        The decoder sequence always starts with ``decoder_start_token_id`` at
        position 0, followed by the generated semantic-ID tokens. Only token
        continuations that lead to a valid catalog item are permitted; after
        ``n_digit`` tokens have been produced, only EOS is allowed.
        """
        n_digit = self.n_digit
        eos = self.eos_token
        trie = self._trie

        def prefix_allowed_tokens_fn(batch_id, input_ids):
            ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
            generated = tuple(int(x) for x in ids[1:])  # drop decoder_start token
            if len(generated) >= n_digit:
                return [eos]
            return trie.get(generated, [eos])

        return prefix_allowed_tokens_fn

    # ------------------------------------------------------------------ #
    # Properties                                                         #
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        return self.eos_token + 1

    @property
    def max_token_seq_len(self) -> int:
        # history items * digits per item, plus 1 for the trailing EOS token.
        return self.max_item_seq_len * self.n_digit + 1

    # ------------------------------------------------------------------ #
    # Sequence tokenization                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def parse_sid_list(cell) -> List[str]:
        """Parse a CSV ``history_item_sid`` cell into a list of SID strings."""
        if isinstance(cell, list):
            return cell
        return ast.literal_eval(cell)

    def encode_history(self, history_sids: Sequence[str]) -> Tuple[List[int], List[int]]:
        """Encode a list of concatenated history SID strings for the encoder.

        Returns a fixed-length ``(input_ids, attention_mask)`` pair padded to
        :attr:`max_token_seq_len`.
        """
        history_sids = list(history_sids)[-self.max_item_seq_len:]
        input_ids: List[int] = []
        for sid in history_sids:
            input_ids.extend(self.sid_string_to_ids(sid))
        input_ids.append(self.eos_token)
        real_len = len(input_ids)

        pad_len = self.max_token_seq_len - real_len
        if pad_len < 0:  # safety: should not happen given the truncation above
            input_ids = input_ids[:self.max_token_seq_len]
            real_len = self.max_token_seq_len
            pad_len = 0
        input_ids = input_ids + [self.padding_token] * pad_len
        attention_mask = [1] * real_len + [0] * pad_len
        
        return input_ids, attention_mask

    def encode_target(self, target_sid_string: str) -> List[int]:
        """Encode the next-item SID string as decoder labels (tokens + EOS)."""
        return list(self.sid_string_to_ids(target_sid_string)) + [self.eos_token]
