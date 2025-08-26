import json
from .my_train_bpe import split_by_special, PAT
from typing import Iterator, Iterable
import json

def split_to_words(text):
    "Split text into words."
    return PAT.findall(text)

def word2bytes(word) ->tuple[bytes, ...]:
    "Convert word string to tuple of bytes"
    a = list(word.encode('utf-8'))
    return tuple(bytes([i]) for i in a)

def apply_merges(word_bytes, merges_set, vocab_to_id):
    word_bytes_list = list(word_bytes)
    while True:
        min_token_id = float('inf')        
        merged = None
        best_pair_idx = -1

        for i in range(len(word_bytes_list) - 1) :
            pair = (word_bytes_list[i], word_bytes_list[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined                
        
        if best_pair_idx == -1:
            break
        word_bytes_list = word_bytes_list[:best_pair_idx] + [merged] + word_bytes_list[best_pair_idx+2:]

    return tuple(word_bytes_list)

def encode_merged(text,merges,vocab_to_id) -> list[int]:
    word_list = split_to_words(text)
    tokens=[]
    for word in word_list:
        word_bytes=word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes,merges,vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens

class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens_bytes = [i.encode("utf-8") for i in self.special_tokens]

        self.vocab_to_id={v:k for k,v in vocab.items()}
        
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.vocab_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.vocab_to_id[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, 
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v) 
                     for k, v in vocab_data.items()}
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            lines =  mf.readlines()
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
                print(f"tokens: {tokens}")            
        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        "Decode a sequence of token IDs into text."
        return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')