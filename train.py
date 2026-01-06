from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Conv1D
from quantutils import SymQuantization, quantConfig, quantize_model, set_quant_precision
from lora import apply_lora, set_lora_precision

PRECISIONS = ['4bit', '8bit', '16bit']
PRECISION_CONFIGS = {
    "4bit": quantConfig(W_bit=4, A_bit=8, KV_bit=4),
    "8bit": quantConfig(W_bit=8, A_bit=8, KV_bit=4),
    "16bit": quantConfig(W_bit=16, A_bit=16, KV_bit=16),
}

LOSS_SCALE = {'4bit': 1.0, '8bit': 1.0, '16bit': 1.0}

def set_all_precisions(model, precision: str, precision_configs: dict):
    set_quant_precision(model, precision_configs[precision])
    set_lora_precision(precision)


def collate_fn(batch):
    """Custom collate for SQuAD dataset"""
    return {
        'context': [item['context'] for item in batch],
        'question': [item['question'] for item in batch],
        'answers': [item['answers']['text'][0] if item['answers']['text'] else "" for item in batch]
    }


def prepare_squad_batch(batch, tokenizer, max_length=128):
    """Prepare SQuAD batch for causal LM training"""
    texts = [f"Context: {c}\nQuestion: {q}\nAnswer: {a}" 
             for c, q, a in zip(batch['context'], batch['question'], batch['answers'])]
    
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    encodings['labels'] = encodings['input_ids'].clone()
    return encodings


def train_step(model, batch, optimizer, precisions, precision_configs, loss_scale):
    """
    Single training step that trains ALL precision LoRAs simultaneously.
    Accumulates gradients from all precisions, then does one optimizer step.
    """
    optimizer.zero_grad()
    losses = {}
    
    for precision in precisions:
        set_all_precisions(model, precision, precision_configs)
        
        # loss_fct = torch.nn.CrossEntropyLoss()
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )

        # loss = loss * loss_scale[precision]
        loss = outputs.loss * loss_scale[precision]
        loss.backward()  
        
        losses[precision] = outputs.loss.item()
        
        del outputs, loss
    
    optimizer.step()
    
    return losses



ds = load_dataset("rajpurkar/squad")
train_data = ds['train'].select(range(1000))  

train_loader = DataLoader(
    train_data, 
    batch_size=4, 
    shuffle=True,
    collate_fn=collate_fn
)

print(f"Training samples: {len(train_data)}")
print(f"Batches per epoch: {len(train_loader)}")


model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="sdpa")
model = quantize_model(model, quant_func=SymQuantization.apply)
model = apply_lora(model, PRECISIONS, r=16, alpha=2.0)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

tokenizer.pad_token = tokenizer.eos_token

lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
optimizer = AdamW(lora_params, lr=1e-3)

print(f"\nLoRA parameters per precision: {len(lora_params) // len(PRECISIONS)}")
print(f"Total LoRA parameters: {len(lora_params)}")

num_epochs = 1

model.train()
all_losses = {p: [] for p in PRECISIONS}

for epoch in range(num_epochs):
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        encoded = prepare_squad_batch(batch, tokenizer)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        
        losses = train_step(
            model, encoded, optimizer, 
            PRECISIONS, PRECISION_CONFIGS, LOSS_SCALE
        )
        
        for p, loss in losses.items():
            all_losses[p].append(loss)
        
        avg_losses = {p: sum(all_losses[p][-50:]) / len(all_losses[p][-50:]) for p in PRECISIONS}
        pbar.set_postfix({f"L_{k}": f"{v:.3f}" for k, v in avg_losses.items()})

set_lora_precision(None)

print("\n✅ Training complete!")
print("Final average losses:")
for p in PRECISIONS:
    print(f"  {p}: {sum(all_losses[p][-100:]) / len(all_losses[p][-100:]):.4f}")
