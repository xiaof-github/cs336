from collections import defaultdict
import copy

class BPE_train:
    def __init__(self) -> None:
        self.vocab = {}

    def add_special_tokens(self, tokens):
        for tok in tokens:
            self.vocab[len(self.vocab)] = tok.encode("utf-8") 
    def setBytes(self):
        for i in range(len(self.vocab), 256):
            self.vocab[i] = bytes([i])
    
    def merge(self, corpus: list[list[str]], max_pair):
        if max_pair.encode("utf-8") not in self.vocab.values():    
            self.vocab[len(self.vocab)] = max_pair.encode("utf-8") 
        
        for word in corpus:
            i = 0
            while i < len(word):
    
                if len(word) == 1:
                    pair = word[i]
                    i += 1
                    continue

                if i < len(word) - 1:
                    pair = (word[i], word[i + 1])
                else:
                    break
                merged = ''.join(pair)
                if merged == max_pair and len(word) > 1:
                    word[i] = merged
                    word.pop(i + 1)
                i += 1                
                
        return

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

def get_max(frequency):    
    max_count = 0
    max_pair = None
    for pair, count in frequency.items():
        if count > max_count:
            max_count = count
            max_pair = pair
        elif count == max_count:
            if pair > max_pair:
                max_pair = pair
        
    return max_pair, max_count

def read_text(input_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def split_by_special(text, special_tokens, drop_special=True) -> list[str]:    
    return list(text)

def train_bpe(input_path,vocab_size,special_tokens):

    text = read_text(input_path)
    chunks = split_by_special(text,special_tokens)

    bpe = BPE_train()
    bpe.add_special_tokens(special_tokens)
    bpe.setBytes()
    
    frequency = pre_tokenization(chunks)
    # print(f"frequency: {frequency}")
    corpus = [list(word) for word in chunks]

    tmp_frequency = set_stats(corpus)
    max_pair, max_count = get_max(tmp_frequency)
    merges = []
    merges.append(max_pair)
    while max_count > 1 and len(bpe.vocab) < vocab_size:

        print(f"max pair: {max_pair}, count: {max_count}")
        bpe.merge(corpus, max_pair)
        print(f"after merge corpus: {corpus}")
        tmp_frequency = set_stats(corpus)
        max_pair, max_count = get_max(tmp_frequency)
        merges.append(max_pair)
    vocab = copy.deepcopy(bpe.vocab)
    return vocab, merges


if __name__ == "__main__":
    vocab_size = 300

    print("train bpe")
    special_tokens = ["<|endoftext|>"]

    bpe = BPE_train()
    bpe.add_special_tokens(special_tokens)
    bpe.setBytes()
    # print(f"vocab <|endoftext|> : {list(bpe.vocab[256])}")

    corpus = ["hello", "hello", "hero", "hi", "world", "so","some","some text that i'll pre-tokenize"]
    frequency = pre_tokenization(corpus)
    # print(f"frequency: {frequency}")
    corpus = [list(word) for word in corpus]
    # print(f"corpus: {corpus}")

    tmp_frequency = set_stats(corpus)
    max_pair, max_count = get_max(tmp_frequency)
    merges = []
    merges.append(max_pair)
    while max_count > 1 and len(bpe.vocab) < vocab_size:

        print(f"max pair: {max_pair}, count: {max_count}")
        bpe.merge(corpus, max_pair)
        print(f"after merge corpus: {corpus}")
        tmp_frequency = set_stats(corpus)
        max_pair, max_count = get_max(tmp_frequency)
        merges.append(max_pair)

    print(f"final vocab: {bpe.vocab}")
    # print(f"final corpus: {corpus}")

