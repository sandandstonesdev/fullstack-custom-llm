"""
Main training script for Polish language model - Optimized for Polish quality.
"""
import os
import sys
import time
from tokenizers import Tokenizer
import torch
import torch.nn as nn

# Add src to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from model.model import GPT
from configs.config import GPTConfig
from data.dataloader import SmallLLMDataLoader

torch.set_float32_matmul_precision('high')

def train_small_model():
    """Train a small, affordable model on local laptop."""
    
    # Small model configuration
    config = GPTConfig(
        vocab_size=32768,
        context_length=128,     # Increased for better context understanding
        num_layers=6,           # Increased for better learning capacity
        num_heads=6,            # Increased for better attention
        embd_size=252           # Adjusted to be divisible by num_heads (6)
    )
    
    # Create small model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config=config).to(device)
    
    print(f"Starting small model training")
    print(f"Model size: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"Device: {device}")
    
    # Small training parameters
    batch_size = 8
    context_length = 128
    learning_rate = 5e-4
    steps = 10  # Increased training steps for better learning
    
    # Use small corpus
    train_loader = SmallLLMDataLoader(
        B=batch_size, 
        T=context_length, 
        process_rank=0, 
        num_processes=1, 
        split='train',
        small_corpus=True
    )
    
    val_loader = SmallLLMDataLoader(
        B=batch_size, 
        T=context_length, 
        process_rank=0, 
        num_processes=1, 
        split='val',
        small_corpus=True
    )
    
    # Simple optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Load the CORRECT tokenizer (full vocabulary) - UPDATED PATH
    tokenizer_path = os.path.join(root_path, '..', 'data', 'processed', 'models', 'tokenizers', 'polish_bpe_32k.json')

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"Using tokenizer from: {tokenizer_path}")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Training loop
    model.train()
    start_time = time.time()
    
    for step in range(steps):
        # Get batch
        x_batch, y_batch = train_loader[0]
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        # Forward pass
        try:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x_batch, y_batch)
        except:
            logits, loss = model(x_batch, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        x_batch, y_batch = val_loader[0]
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        try:
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, val_loss = model(x_batch, y_batch)
        except:
            logits, val_loss = model(x_batch, y_batch)
    
    # Generate some text
    print(f"\nFinal training loss: {loss.item():.4f}")
    print(f"Validation loss: {val_loss.item():.4f}")
    
    # Text generation demo
    print(f"\nText generation sample:")
    prompt = "Polska to piekny kraj, ktory"
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        generated = model.generate(input_tensor, max_new_tokens=20, temperature=0.7)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    # Save the trained model
    model_dir = os.path.join(root_path, '..', 'data', 'processed', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "polish_small_model.pt")

    # Convert config to dict for safer serialization
    config_dict = {
        'vocab_size': config.vocab_size,
        'context_length': config.context_length,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'embd_size': config.embd_size,
        'dropout': config.dropout,
        'bias': config.bias
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'val_loss': val_loss.item()
    }, model_path)
    
    print(f"\nModel saved to: {model_path}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.1f} seconds")
    print(f"Model ready for inference!")
    
    return model_path

if __name__ == "__main__":
    train_small_model()