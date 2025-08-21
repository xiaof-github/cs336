from collections import defaultdict
import copy
import regex as re
from tqdm.contrib.concurrent import process_map

def add_special_tokens(vocab, tokens):
    for tok in tokens:
        vocab[len(vocab)] = tok.encode("utf-8") 
def setBaseToken(vocab):
    for i in range(0, 256):
        vocab[i] = bytes([i])

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def get_max_pair(pair_cnt):
    max_count = -1
    max_pair = None
    for pair, count in pair_cnt.items():
        if count > max_count or (count == max_count and pair > max_pair):
            max_count = count
            max_pair = pair
    return max_pair

def pre_tokenization(corpus):
    frequency = {}  # 默认值为0
    for word in corpus:        
        frequency[word] = frequency.get(word, 0) + 1
    return frequency

def set_stats(corpus: list[list[str]]):
    frequency = {}
    for word in corpus:
        for i in range(len(word)):
            if len(word) == 1:
                pair = word[i]
            elif i < len(word) - 1:
                pair = (word[i], word[i + 1])
            else:
                break
            merged = ''.join(pair)
            frequency[merged] = frequency.get(merged, 0) + 1
            if len(word) == 1:
                frequency[merged] = 1
    return frequency

def count_word(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_cnt = {}
    for m in PAT.finditer(text):
        word = m.group(0)
        a = list(word.encode('utf-8'))
        word_bytes = tuple(bytes([i]) for i in a)         
        if len(word_bytes)>=2:
            word_cnt[word_bytes] = word_cnt.get(word_bytes, 0) + 1
    return word_cnt

def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_by_special(text, special_tokens, drop_special=True) -> list[str]:    
    if not special_tokens:
        return [text]

    # Sort by descending length to prioritize longer tokens (e.g., "<|endoftext|><|endoftext|>" before "<|endoftext|>")
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special: pattern = f"({pattern})"

    pattern = re.compile(pattern)
    chunks = pattern.split(text)
    return [c for c in chunks if c]

def train_bpe1(input_path,vocab_size,special_tokens):

    text = read_text(input_path)    
    chunks = split_by_special(text,special_tokens)

    vocab = {}
    setBaseToken(vocab)
    add_special_tokens(vocab, special_tokens)
    
    # word_dicts = process_map(count_word, chunks, chunksize=1)

    pre_bytes: list[list[bytes]] = []
    print(len(chunks))
    for doc in chunks:
        tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, doc)]
        for token in tokens:
            token_bytes = [bytes([b]) for b in token]
            pre_bytes.append(token_bytes)
    
    
    word_dicts = [count_word(chunk) for chunk in chunks]
    print(len(word_dicts))
    # print(word_dicts[:10])

    merged = {}
    for item in word_dicts:
        for k, v in item.items():
            merged[k] = merged.get(k, 0) + v
    i = 0
    for k, v in merged.items():
        print(k, v)
        i += 1
        if i > 10: break
    
    

    
    merges : list[tuple[bytes, bytes]] = []
    base_vocab_size = len(vocab)
    n_merges=vocab_size-base_vocab_size

    pre_tokens_bytes: list[list[bytes]] = [list(k) for k, _ in merged.items()]
    count_pair = defaultdict(int)
    print(f"pre_tokens_bytes: {pre_tokens_bytes[:3]}")
    for it in pre_tokens_bytes:
        # print(merged[tuple(it)])
        for i in range(len(it)):
            if i < len(it) - 1:
                count_pair[(it[i], it[i + 1])] += 1            

    print(f"count_pair: {len(count_pair)}")
    i = 0
    for pair, count in count_pair.items():
        print(pair, count)
        i += 1
        if i > 10: break

    max_pair = get_max_pair(count_pair)
    print(f"max_pair: {max_pair}, count: {count_pair[max_pair]}")        

    for i in range(n_merges):
        
        pass
    

    return vocab, merges


if __name__ == "__main__":
    vocab_size = 300

    print("train bpe")
    special_tokens = ["<|endoftext|>"]

    train_bpe1("tests/fixtures/corpus.en", vocab_size, special_tokens)
    print("done")


