import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

os.system("pip install bitsandbytes")

import torch
device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

# Download and cache the file from HF Hub
filepath = hf_hub_download(
    repo_id="AhmetSemih/news_merged_bpe_token_ids",
    filename="merged_bpe_tokens.safetensors",
    repo_type="dataset"
)

# Load tensors
tensors = load_file(filepath)
token_ids = tensors['a']


from huggingface_hub import hf_hub_download
from safetensors import torch as sftorch
current_dir = os.getcwd()
model_path = hf_hub_download(repo_id="aliarda/llama-50M-randParams", filename="llama-50M.safetensors", local_dir=current_dir)
state_dict = sftorch.load_file(model_path, device=device)

from llama_config import LlamaConfig
from llama_model import LlamaForCausalLM
llama_config = LlamaConfig(
    vocab_size=32768,
    emb_dim=256,
    context_length=256,
    n_heads=128,
    n_layers=20,
    n_kv_groups=64,
    hidden_dim=2048,
)

llama_model = LlamaForCausalLM(llama_config)
llama_model = llama_model.to(device)

llama_model.load_state_dict(state_dict)

from llama_text_dataset import TextDataset
from torch.utils.data import DataLoader

def create_dataloader(token_ids: list, context_len: int, stride: int, batch_size: int, shuffle: bool, device: str = "cpu"):
    dataset = TextDataset(token_ids, context_len, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(device='cpu')
    )
    return dataloader

train_dataloader = create_dataloader(token_ids.tolist(), 256, 256, 64, False,device)


import time
import torch
from bitsandbytes.optim import AdamW8bit
from huggingface_hub import upload_file
from tqdm import tqdm
from safetensors import torch as sftorch

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW8bit(llama_model.parameters(), lr=1e-3)

save_interval = 2*60*60 # 2 hours in seconds
last_save_time = time.time()

num_epochs = 2
checkpoint_num = 1

for epoch_idx in range(num_epochs):
    total_loss = 0
    last_loss = 0

    for X, Y in tqdm(train_dataloader):
        X, Y = X.to(device), Y.to(device)

        pred = llama_model(X)
        loss = loss_fn(pred.flatten(0, 1), Y.flatten())
        total_loss += loss.item()
        last_loss = loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del pred, loss, X, Y
        torch.cuda.empty_cache()

        # Push model to HF every save_interval seconds
        if time.time() - last_save_time >= save_interval:
            last_save_time = time.time()
            # Save and upload using sftorch + HF
            sftorch.save_file(llama_model.state_dict(), f"llama_model_{epoch_idx}_{checkpoint_num}.safetensors")
            upload_file(
                path_or_fileobj=f"llama_model_{epoch_idx}_{checkpoint_num}.safetensors",
                repo_id="AhmetSemih/llama-50m-pretrained-merged_dataset-bpe",
                path_in_repo="llama-50m-pretrained-merged_dataset-bpe.safetensors",
                commit_message=f"upload llama_model chunk: {checkpoint_num}, epoch: {epoch_idx}"
            )
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Uploaded checkpoint {checkpoint_num}")
            checkpoint_num += 1

    # Upload final model at the end of the epoch
    sftorch.save_file(llama_model.state_dict(), f"llama_model_{epoch_idx}_final.safetensors")
    upload_file(
        path_or_fileobj=f"llama_model_{epoch_idx}_final.safetensors",
        repo_id="AhmetSemih/llama-50m-pretrained-merged_dataset-bpe",
        path_in_repo="lllama-50m-pretrained-merged_dataset-bpe.safetensors",
        commit_message=f"Final upload epoch {epoch_idx}"
    )
    print(f"Epoch {epoch_idx} completed. Uploaded final checkpoint.")

