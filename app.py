import torch
import gradio as gr
from main import CausalDecoderGPT, SimpleTokenizer, generate_text

# モデルとトークナイザのロード
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = SimpleTokenizer()
    tokenizer.load()
    
    model = CausalDecoderGPT(
        vocab_size=len(tokenizer.char2id),
        d_model=1024,
        nhead=16,
        dim_feedforward=1024,
        num_layers=8,
        max_seq_len=256
    )
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.to(device)
    return model, tokenizer, device

# Gradioインターフェース
def create_interface():
    model, tokenizer, device = load_model()
    
    def chat_interface(user_input, history):
        history = history or []
        prompt = "\n".join(history) + f"\n[USER] {user_input}\n[GPT] "
        response = generate_text(model, tokenizer, prompt, max_len=80, device=device)
        response = response[len(prompt):].split("\n")[0].strip()
        history.append(f"[USER] {user_input}")
        history.append(f"[GPT] {response}")
        return history, history
    
    with gr.Blocks() as demo:
        gr.Markdown("## ポケモンしりとりチャットボット")
        chatbot = gr.Chatbot()
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="ポケモン名を入力してください")
        with gr.Row():
            clear = gr.Button("クリア")
        
        txt.submit(chat_interface, [txt, chatbot], [chatbot, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return demo

# Huggingface Space用の設定
if __name__ == "__main__":
    interface = create_interface()
    interface.launch()
