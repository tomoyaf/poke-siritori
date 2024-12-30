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


def normalize_to_hiragana(text: str) -> str:
    """
    テキストをひらがなに変換し、半角・全角の違いなどを標準化
    jaconvを用いてカタカナ → ひらがなに変換
    """
    # カタカナ→ひらがな
    text = jaconv.kata2hira(text)
    # アルファベットや数字を無理やりカナに寄せる（今回は簡易的な例）
    text = jaconv.alphabet2kana(text)
    # 全角・半角の正規化
    text = unicodedata.normalize("NFKC", text)
    return text

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
            # hiragana_name = normalize_to_hiragana(name)
            # if hiragana_name:
            #     writer.writerow([hiragana_name])

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

def generate_random_user_word(max_length=5):
    """
    ユーザーの発話用にランダムひらがなを生成
    「ん」で終わる単語はNG
    """
    while True:
        length = random.randint(1, max_length)
        text = "".join(random.choice(KATAKANA_CHARS) for _ in range(length))

        # しりとりのルールにしたがって最後の文字を取得可能で、それが「ん」でないならそのまま返す
        last_char = get_last_char(text)
        if last_char is not None and last_char != "ン":
            return text

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


def build_realistic_conversation_dataset(
    pokemon_names,
    num_samples=5000,
    max_turns=6
):
    """
    しりとり会話データを大量に生成
    - ユーザーの発話: ランダムなひらがな単語 (末尾「ん」禁止)
    - GPTの発話: ポケモン名 (末尾「ん」禁止)
    - 一度出た単語は再利用禁止
    - 「最後の文字」→「頭文字」ルールが維持できない場合は会話終了
    - 会話は [USER] word\n[GPT] word\n[USER] word...\n のように進む
    - max_turns 回数程度、行ったら終了 (途中で候補がなければ終了)
    - 1サンプルにつき1つの会話ログ
    """
    # ポケモン名を「末尾がんじゃない」もののみ残す
    valid_pokemon_names = []
    for name in pokemon_names:
        if len(name) > 0 and name[-1] != "ン":
            valid_pokemon_names.append(name)

    # 頭文字別の辞書を作る
    pokemon_dict_by_initial = {}
    for name in valid_pokemon_names:
        init = name[0]
        if init not in pokemon_dict_by_initial:
            pokemon_dict_by_initial[init] = []
        pokemon_dict_by_initial[init].append(name)

    # 実際の会話データを作る
    data = []
    for _ in range(num_samples):
        used_words = set()
        conversation_lines = []

        # 開始はユーザーのランダムワード
        user_word = generate_random_user_word(max_length=5)
        used_words.add(user_word)
        conversation_lines.append(f"[USER] {user_word}")

        last_char = get_last_char(user_word)
        if last_char is None:
            break

        turn_count = 1  # ユーザー発話を1ターン目として数える

        while True:
            # GPTの発話
            candidate_list = pokemon_dict_by_initial.get(last_char, [])
            # まだ使われていない単語をフィルター
            candidate_list = [w for w in candidate_list if w not in used_words]
            if len(candidate_list) == 0:
                # 候補がないので会話終了
                break
            gpt_word = random.choice(candidate_list)
            used_words.add(gpt_word)
            conversation_lines.append(f"[GPT] {gpt_word}")

            turn_count += 1
            if turn_count >= max_turns:
                break

            # 次のユーザー発話
            last_char = get_last_char(gpt_word)
            if last_char is None:
                break
            candidate_user_word = None
            # ユーザーも同じルールでつなげる: 「最後の文字→頭文字」
            # が、ここでは(簡易的に) last_char で始まるユーザー語をランダム生成 してみる
            # ただし最後は「ん」にしない + 未使用の単語にする
            # (本当はガチしりとりなら「文字指定してランダム生成」する必要があるが、簡略化)
            # → ここでは「頭文字 = last_char」+ ランダムな長さ(1~5)で作る
            #    末尾に「ん」は付かないように再生成
            for _ in range(30):  # 最大30回試行
                length = random.randint(1, 5)
                # 強引に先頭文字を last_char に固定
                random_tail = "".join(random.choice(KATAKANA_CHARS) for _ in range(length - 1))
                user_candidate = last_char + random_tail
                if user_candidate[-1] != "ン" and user_candidate not in used_words:
                    candidate_user_word = user_candidate
                    break

            if candidate_user_word is None:
                # ユーザーが単語を作れなかったら会話終了
                break
            user_word = candidate_user_word
            used_words.add(user_word)
            conversation_lines.append(f"[USER] {user_word}")

            turn_count += 1
            if turn_count >= max_turns:
                break
            last_char = get_last_char(user_word)
            if last_char is None:
                break

        # 1つの会話ログとして結合
        # [USER] あいう
        # [GPT] うそ
        # [USER] そら
        # ...
        # のような形で一つのサンプル文章にする
        conversation_text = "\n".join(conversation_lines) + "\n"
        data.append(conversation_text)
    return data

# -------------------------------------------------------
# 3. ミニGPTモデル (超簡易版)
# -------------------------------------------------------
class ToyGPT(nn.Module):
    """
    簡易的な GPT 風モデル (Encoderベース)
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers=2, max_seq_len=256):
        super(ToyGPT, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=4, dim_feedforward=hidden_dim, activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)
        tok_emb = self.embedding(x)  # (B, S, d_model)
        pos_emb = self.pos_embedding(positions)    # (1, S, d_model)
        x = tok_emb + pos_emb
        x = x.transpose(0, 1)  # Convert to (S, B, E) format for transformer

        # Generate causal mask of appropriate size
        causal_mask = self.generate_causal_mask(seq_len, seq_len, device=x.device)

        out = self.transformer_decoder(x, memory=None, tgt_mask=causal_mask)  # (seq_len, batch_size, embed_dim)
        out = out.transpose(0, 1)     # (batch_size, seq_len, embed_dim)
        logits = self.fc_out(out)     # (batch_size, seq_len, vocab_size)
        return logits

    def generate_causal_mask(self, sz: int, sz2: int, device='cpu'):
        """
        下三角マスクを作り、未来のトークンを参照しないようにする
        shape: (sz, sz2)
        """
        # True=マスク なので、上三角を True に
        mask = torch.triu(torch.ones(sz, sz2, dtype=torch.bool, device=device), diagonal=1)
        return mask

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

# -------------------------------------------------------
# 4. トークナイザ & Dataset
# -------------------------------------------------------
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
# 5. 学習＆評価
# -------------------------------------------------------
def train_model(model, train_loader, val_loader, tokenizer, epochs=5, lr=1e-4, device="cpu"):
    writer = SummaryWriter(log_dir='runs/pokemon_siritori')
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char2id[tokenizer.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_val_loss = float("inf")
    steps = 0
    demo_interval = 500  # 500ステップごとにデモを実行

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            seq_len = targets.size(1)
            logits = logits[:, :seq_len, :].contiguous().view(-1, model.vocab_size)
            targets = targets.contiguous().view(-1)

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            # 定期的に推論デモを実行
            if steps % demo_interval == 0:
                # TensorBoardにステップごとの学習損失を記録
                writer.add_scalar('Loss/train_step', loss.item(), steps)
                print("\nDemo generation at step", steps)
                print("-" * 40)
                demo_prompts = ["[USER] ミズ\n[GPT] ", "[USER] カゼ\n[GPT] "]
                model.eval()
                with torch.no_grad():
                    for prompt in demo_prompts:
                        response = generate_text(model, tokenizer, prompt, device=device)
                        print(f"Prompt: {prompt}")
                        print(f"Response: {response}\n")
                model.train()
                print("-" * 40)

        train_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)

        # validation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                logits = model(inputs)
                seq_len = targets.size(1)
                logits = logits[:, :seq_len, :].contiguous().view(-1, model.vocab_size)
                targets = targets.contiguous().view(-1)

                loss = criterion(logits, targets)
                val_loss_sum += loss.item()

        val_loss = val_loss_sum / len(val_loader)
        writer.add_scalar('Loss/validation', val_loss, epoch)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 最良モデルの保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    writer.close()
    print("Training complete. Best val loss: ", best_val_loss)

def evaluate_model(model, test_loader, tokenizer, device="cpu"):
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char2id[tokenizer.pad_token])
    model.eval()
    model.to(device)

    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits = model(inputs)
            seq_len = targets.size(1)
            logits = logits[:, :seq_len, :].contiguous().view(-1, model.vocab_size)
            targets = targets.contiguous().view(-1)
            loss = criterion(logits, targets)
            total_loss += loss.item()
    test_loss = total_loss / len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

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

    # データセットを作る
    print("Building dataset (conversation style)...")
    dataset_cache = "dataset_cache.pkl"
    if os.path.exists(dataset_cache):
        print("Loading dataset from cache...")
        with open(dataset_cache, "rb") as f:
            dataset_texts = pickle.load(f)
    else:
        print("Building new dataset...")
        dataset_texts = build_realistic_conversation_dataset(
            pokemon_names,
            num_samples=300000,   # 必要に応じて増やす
            max_turns=10        # 1つの会話でのターン数 (USER->GPT->USER->GPT...)
        )
        print("Saving dataset to cache...")
        with open(dataset_cache, "wb") as f:
            pickle.dump(dataset_texts, f)

    # train, val, test に分割
    random.shuffle(dataset_texts)
    n = len(dataset_texts)
    train_texts = dataset_texts[: int(n*0.8)]
    val_texts   = dataset_texts[int(n*0.8) : int(n*0.9)]
    test_texts  = dataset_texts[int(n*0.9) : ]

    # トークナイザ作成
    tokenizer = SimpleTokenizer()
    if tokenizer.exists():
        tokenizer.load()
    else:
        tokenizer.build_vocab(train_texts + val_texts + test_texts)
        tokenizer.save()

    # Dataset
    train_ds = TextDataset(train_texts, tokenizer, max_seq_len=256)
    val_ds   = TextDataset(val_texts,   tokenizer, max_seq_len=256)
    test_ds  = TextDataset(test_texts,  tokenizer, max_seq_len=256)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn)

    # モデル
    vocab_size = len(tokenizer.char2id)
    # model = ToyGPT(vocab_size=vocab_size, embed_dim=1024, hidden_dim=1024, n_layers=12, max_seq_len=256)
    model = CausalDecoderGPT(vocab_size=vocab_size,
                             d_model=1024,
                             nhead=16,
                             dim_feedforward=1024,
                             num_layers=8,
                             max_seq_len=256)

    # 学習
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # train_model(model, train_loader, val_loader, tokenizer, epochs=3, lr=1e-4, device=device)

    # テスト評価
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    # evaluate_model(model, test_loader, tokenizer, device=device)

    # 推論デモ
    print("\n=== Demo ===")

    demo_prompt = ""
    while True:
        user_input = input("[USER] ")
        if user_input == "exit":
            break
        demo_prompt += f"\n[USER] {user_input}\n[GPT] "
        print(f"demo_prompt: {demo_prompt}")
        generated = generate_text(model, tokenizer, demo_prompt, max_len=80, device=device)
        print("generated1:")
        print(generated)
        # GPTが新しく生成した部分だけにするために、demo_promptの長さ文だけ先頭から削除する
        generated = generated[len(demo_prompt):]
        print("generated2:")
        print(generated)
        # 先頭から改行までを抜き出す
        # 空白と改行を消す
        new_gpt_text = generated.split("\n")[0].strip()
        demo_prompt += new_gpt_text
        print("new gpt text:")
        print(new_gpt_text)

    print("-------------------")

if __name__ == "__main__":
    main()
