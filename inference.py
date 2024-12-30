#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import csv
import random
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
import jaconv
import unicodedata
from tqdm import tqdm
import math
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------------
# 1. PokeAPIからポケモン名を取得 → ひらがな変換 → CSV保存
# -------------------------------------------------------
def get_all_pokemon_names(language: str = "ja-Hrkt") -> list:
    """
    PokeAPIからポケモン名を取得し、指定した言語の名前を返す
    language='ja' で日本語名を取得
    ローカルにキャッシュしておいて、一度取得したらそれを使い回すようにする
    """
    cache_file = "pokemon_names_raw.csv"
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            all_pokemon_names = [row[0] for row in reader]
        return all_pokemon_names

    all_pokemon_names = []
    # PokeAPIのポケモン総数を取得
    # 例として151（初代）だけにするなら limit=151 などに変更
    url = "https://pokeapi.co/api/v2/pokemon?limit=1010"
    response = requests.get(url).json()
    results = response["results"]

    for result in tqdm(results, desc="Fetching Pokemon"):
        detail = requests.get(result["url"]).json()
        # detail["names"] から日本語名を検索
        names_info = detail.get("names", [])
        species_url = f"https://pokeapi.co/api/v2/pokemon-species/{detail['id']}"
        species_detail = requests.get(species_url).json()
        ja_name = None
        for item in species_detail.get("names", []):
            if item.get("language", {}).get("name", "") == language:
                ja_name = item.get("name")
                break
        if ja_name is None:
            pass
        all_pokemon_names.append(ja_name)

    with open(cache_file, "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name in all_pokemon_names:
            writer.writerow([name])

    return all_pokemon_names


def save_pokemon_names_to_csv(pokemon_names, csv_path="pokemon_names.csv"):
    """
    ポケモン名をCSVに保存
    1列目: ポケモン名(ひらがな)
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name_hiragana"])  # ヘッダ
        for name in pokemon_names:
            writer.writerow([name])

# -------------------------------------------------------
# 2. データセットを作成
#    （より現実的な会話のラリー形式で、しりとりを進める）
# -------------------------------------------------------

# ひらがな文字集合
# HIRAGANA_CHARS = [chr(i) for i in range(ord("ぁ"), ord("ゖ") + 1)]
# # 重複表現の記号などは適宜除外 (例: ゝ, ゞ など)
# HIRAGANA_CHARS = [c for c in HIRAGANA_CHARS if c not in ["ゝ","ゞ"]]
# カタカナ
KATAKANA_CHARS = [chr(i) for i in range(ord("ァ"), ord("ヺ") + 1)]
# 重複表現の記号などは適宜除外 (例: ゝ, ゞ など)
KATAKANA_CHARS = [c for c in KATAKANA_CHARS if c not in ["ゝ","ゞ", "ヮ", "ヵ", "ヸ", "ヹ", "ヺ", "ヿ", "ヾ", "ヰ"]]

def get_last_char(text):
    """
    しりとり回答の最後の文字を返す
    - 伸ばし棒は無視する
    - 小文字（ァィゥェォャュョッ）は大文字に変換する
    - ヮはワに変換する
    - ヵはカに変換する
    """
    try:
        last_char = text[-1]

        if last_char == "ー":
            if len(text) > 1:
                last_char = text[-2]
            else:
                return None

        if last_char in ["ァ", "ィ", "ゥ", "ェ", "ォ", "ャ", "ュ", "ョ", "ヮ", "ヵ", "ッ"]:
            last_char = chr(ord(last_char) - ord("ァ") + ord("ア"))

        return last_char
    except IndexError:
        return None

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        # x: (seq_len, batch_size, d_model)
        # Self-Attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)  # query,key,value = x
        x = x + attn_out
        x = self.layernorm1(x)

        # Feed Forward
        ff_out = self.feed_forward(x)
        x = x + ff_out
        x = self.layernorm2(x)
        return x

class CausalDecoderGPT(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4,
                 dim_feedforward=512, num_layers=4, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding   = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        # input_ids: (B, S)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embed
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        tok_emb = self.token_embedding(input_ids)  # (B, S, d_model)
        pos_emb = self.pos_embedding(positions)    # (1, S, d_model)
        x = tok_emb + pos_emb  # (B, S, d_model)

        # (S, B, d_model)に変形
        x = x.transpose(0, 1)

        # Causal mask: 下三角True → マスク
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )  # 上三角True
        # MultiheadAttentionの引数 attn_mask は “True=無視” なので上三角をTrueに

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        # (S, B, d_model) -> (B, S, d_model)
        x = x.transpose(0, 1)
        logits = self.fc_out(x)  # (B, S, vocab_size)

        return logits


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

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, tokenizer, max_seq_len=256):
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        text = self.text_list[idx]
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)

def collate_fn(batch):
    # バッチ内の系列長を最大長に合わせてパディング
    max_len = max([len(x) for x in batch])
    padded = []
    for x in batch:
        pad_len = max_len - len(x)
        padded.append(torch.cat([x, torch.zeros(pad_len, dtype=torch.long)]))
    return torch.stack(padded, dim=0)

# -------------------------------------------------------
# 6. Botとして推論 (生成)
# -------------------------------------------------------
def generate_response(model, tokenizer, prompt, max_len=100, device="cpu"):
    """
    簡易推論: prompt を入力に、その続きを自動生成。
    """
    model.eval()
    model.to(device)
    encoded = tokenizer.encode(prompt, add_eos=False)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]

            # top-1 (greedy)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # EOS なら break
            if next_token_id.item() == tokenizer.char2id[tokenizer.eos_token]:
                break

    generated = tokenizer.decode(input_ids[0].cpu().numpy())
    return generated

def generate_text(model, tokenizer, prompt: str, max_len=50, device="cpu"):
    """
    [USER], [GPT] などのロール表現も踏まえつつ、自己回帰的にトークンを生成
    Greedyサンプルの簡易実装 (top-1)
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        # プロンプトをトークナイズ（ここでは <BOS>, <EOS> を付けない）
        input_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

        for _ in range(max_len):
            # モデルで推論
            logits = model(input_ids)
            # logits.shape -> (B, seq_len, vocab_size)
            next_token_logits = logits[:, -1, :]  # 最後トークンのlogits (B, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1)  # greedy
            next_token_id = next_token_id.unsqueeze(0)  # (1, B)

            # 末尾に追加
            input_ids = torch.cat([input_ids, next_token_id.transpose(0,1)], dim=1)
            # EOSなら終了
            if next_token_id.item() == tokenizer.char2id[tokenizer.eos_token]:
                break

        generated_ids = input_ids[0].cpu().tolist()
        text = tokenizer.decode(generated_ids, remove_special=False)
        return text

# -------------------------------------------------------
# main
# -------------------------------------------------------
def main():
    # 0. もし既にCSVが無ければ取得＆保存
    csv_path = "pokemon_names.csv"
    if not os.path.exists(csv_path):
        print("Fetching Pokemon names from PokeAPI...")
        pokemon_names = get_all_pokemon_names(language="ja")
        save_pokemon_names_to_csv(pokemon_names, csv_path)
    else:
        pokemon_names = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # ヘッダスキップ
            for row in reader:
                pokemon_names.append(row[0])

    # トークナイザ作成
    tokenizer = SimpleTokenizer()
    tokenizer.load()

    # モデル
    vocab_size = len(tokenizer.char2id)
    model = CausalDecoderGPT(vocab_size=vocab_size,
                             d_model=1024,
                             nhead=16,
                             dim_feedforward=1024,
                             num_layers=8,
                             max_seq_len=256)

    # 学習
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    demo_prompt = ""
    while True:
        user_input = input("[USER] ")
        if user_input == "exit":
            break
        demo_prompt += f"\n[USER] {user_input}\n[GPT] "
        print(f"demo_prompt: {demo_prompt}")
        generated = generate_text(model, tokenizer, demo_prompt, max_len=80, device=device)
        # GPTが新しく生成した部分だけにするために、demo_promptの長さ文だけ先頭から削除する
        generated = generated[len(demo_prompt):]
        # 先頭から改行までを抜き出す
        # 空白と改行を消す
        new_gpt_text = generated.split("\n")[0].strip()
        demo_prompt += new_gpt_text
        print("new gpt text:")
        print(new_gpt_text)

    print("-------------------")

if __name__ == "__main__":
    main()
