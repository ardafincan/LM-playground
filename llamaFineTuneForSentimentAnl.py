import torch
import sys

sys.path.append("./llama_architecture")
from safetensors import torch as sftorch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download, HfApi, upload_file
from config import LlamaConfig
from model_trnsfmrs import LlamaForCausalLM, LlamaForSentimentAnalysis
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from bitsandbytes.optim import AdamW32bit, AdamW8bit
from twilio.rest import Client
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/sentiment-finetuning")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
context_len = 128

tokenizer = AutoTokenizer.from_pretrained("aliarda/turkish-news-32k-tokenizer")
tensor_file = hf_hub_download(
    repo_id="aliarda/llama-50M-SentimentAnalysis", filename="model.safetensors"
)
tensors = sftorch.load_file(tensor_file, device=device)

llama_config = LlamaConfig(
    vocab_size=32768,
    emb_dim=256,
    context_length=context_len,
    n_heads=128,
    n_layers=20,
    n_kv_groups=64,
    hidden_dim=2048,
)

sentiment_model = LlamaForSentimentAnalysis(llama_config, tokenizer)
sentiment_model = sentiment_model.to(device)
sentiment_model.load_state_dict(tensors, strict=False)

for parameter in sentiment_model.parameters():
    parameter.requires_grad = True

for param in sentiment_model.model.embed_tokens.parameters():
    param.requires_grad = False

for idx, layer in enumerate(sentiment_model.model.layers):
    if idx < 11:
        for param in layer.parameters():
            param.requires_grad = False

# data processing
ds = load_dataset("aliarda/hepsiburadaForSentiment")

train_test_split = ds["train"].train_test_split(test_size=0.2, seed=42)
test_val_split = train_test_split["test"].train_test_split(test_size=0.5, seed=42)

trainData = train_test_split["train"]
testData = test_val_split["train"]
valData = test_val_split["test"]

pad_id = 1
eos_id = 3


class SentimentDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=None, pad_token=1):
        self.data = dataset
        self.pad_token = pad_token

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["text"]]

        if max_length is None:
            self.max_length = 256
        else:
            self.max_length = max_length

        self.encoded_texts = [text[: self.max_length] for text in self.encoded_texts]

        self.encoded_texts = [
            text + [self.pad_token] * (self.max_length - len(text))
            for text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = 1 if self.data["score"][index] == "positive" else 0
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data["score"])


def create_dataloader(dataset, batch_size: int, device: str = "cpu"):
    dataset = SentimentDataset(dataset, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device=device),
    )
    return dataloader

step_counter = 0
val_dataloader = create_dataloader(valData, 32)
val_iterator = iter(val_dataloader)

for i in range(1, 6):
    chunk = i
    train_dataloader = create_dataloader(trainData, 32)

    try:
        # twilio here
        pass
    except Exception as E:
        print(E)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = AdamW32bit(sentiment_model.parameters(), lr=1e-5)

    epoch = 2

    for epoch in range(epoch):
        total_loss = 0
        last_loss = 0
        for inx, (X, Y) in enumerate(tqdm(train_dataloader)):
            X, Y = X.to(device), Y.to(device)

            pred = sentiment_model(X)
            print(pred, pred.flatten(0, 1), Y, Y.flatten())
            loss = loss_fn(pred, Y.flatten())
            #writer.add_scalar("train/loss", loss, i)
            total_loss += loss.item()
            last_loss = loss.item()

            if inx % 8 == 0:
                try:
                    X_val, Y_val = next(val_iterator)
                except StopIteration:
                    val_iterator = iter(val_dataloader)
                    X_val, Y_val = next(val_iterator)
                
                X_val, Y_val = X_val.to(device), Y_val.to(device)

                sentiment_model.eval()
                with torch.no_grad():
                    pred_val = sentiment_model(X_val)
                    val_loss = loss_fn(pred_val, Y_val.flatten())
                sentiment_model.train()
                writer.add_scalars("Loss", {"train": loss, "validation": val_loss}, step_counter)

            step_counter += 1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            del pred, loss, X, Y
            torch.cuda.empty_cache()

        average_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} loss: {last_loss} average loss: {average_loss}")

        try:
            # twilio here
            pass
        except Exception as E:
            print(E)
        sftorch.save_file(
            sentiment_model.state_dict(),
            f"llama-50M-SentimentModel/llama_model_{epoch}_{chunk}.safetensors",
        )
        upload_file(
            path_or_fileobj=f"llama-50M-SentimentModel/llama_model_{epoch}_{chunk}.safetensors",
            repo_id="aliarda/Experimental-Sentiment_Llama-50M-Latest",
            path_in_repo="model.safetensors",
        )

writer.close()