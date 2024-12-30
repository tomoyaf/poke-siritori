import pickle
import os

# dataset_cache = "dataset_cache.pkl"
# if os.path.exists(dataset_cache):
#     print("Loading dataset from cache...")
#     with open(dataset_cache, "rb") as f:
#         dataset_texts = pickle.load(f)

# print(dataset_texts)

class SimpleTokenizer:
    """
    データセット内のすべての文字を集めて vocab を作る簡易トークナイザ
    """
    def __init__(self, tokenizer_path="./tokenizer.pkl"):
        self.char2id = {}
        self.id2char = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.user_token = "[USER]"
        self.gpt_token  = "[GPT]"
        self.tokenizer_path = tokenizer_path

    def build_vocab(self, text_list):
        chars = set()
        for text in text_list:
            for c in text:
                chars.add(c)
        unique_chars = [self.pad_token, self.unk_token, self.bos_token, self.eos_token] + sorted(list(chars))

        for i, c in enumerate(unique_chars):
            self.char2id[c] = i
            self.id2char[i] = c

    def encode(self, text, add_bos=True, add_eos=True):
        tokens = []
        if add_bos:
            tokens.append(self.char2id[self.bos_token])
        for c in text:
            if c in self.char2id:
                tokens.append(self.char2id[c])
            else:
                tokens.append(self.char2id[self.unk_token])
        if add_eos:
            tokens.append(self.char2id[self.eos_token])
        return tokens

    def decode(self, token_ids, remove_special=True):
        result = []
        for t in token_ids:
            c = self.id2char.get(t, self.unk_token)
            if remove_special and c in [self.pad_token, self.bos_token, self.eos_token, self.unk_token]:
                continue
            result.append(c)
        return "".join(result)

    def exists(self):
        return os.path.exists(self.tokenizer_path)

    def save(self):
        with open(self.tokenizer_path, "wb") as f:
            pickle.dump({
                "char2id": self.char2id,
                "id2char": self.id2char,
            }, f)

    def load(self):
        with open(self.tokenizer_path, "rb") as f:
            data = pickle.load(f)
            self.char2id = data["char2id"]
            self.id2char = data["id2char"]


tokenizer = SimpleTokenizer()

print(tokenizer.char2id)
print(tokenizer.id2char)

tokenizer.load()

print(tokenizer.char2id)
print(tokenizer.id2char)
