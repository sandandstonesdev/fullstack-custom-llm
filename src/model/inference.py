"""
Simple inference script for the Polish language model.
"""
import sys
import os
import torch
from tokenizers import Tokenizer

# Add src to path
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_path)

from model.model import GPT
from configs.config import GPTConfig


def load_trained_model():
    """Load the trained model from file."""
    # Try both possible model names
    model_paths = [
        os.path.join(root_path, '..', 'data', 'processed', 'models', 'polish_small_model.pt')
    ]
    
    valid_model_path = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            valid_model_path = model_path
            break
    
    if valid_model_path is None:
        print(f"No trained model found in data/processed/models/")
        print("Available files:", os.listdir(os.path.join(root_path, '..', 'data', 'processed', 'models')) if os.path.exists(os.path.join(root_path, '..', 'data', 'processed', 'models')) else "Directory does not exist")
        print("Please train a model first by running: python main.py --train_model")
        return None, None
    
    try:
        # Load the saved model
        checkpoint = torch.load(valid_model_path, map_location='cpu')
        
        # Handle config dictionary
        config_data = checkpoint['config']
        
        if isinstance(config_data, dict):
            # New format: config is a dictionary
            config = GPTConfig(
                vocab_size=config_data['vocab_size'],
                context_length=config_data['context_length'],
                num_layers=config_data['num_layers'],
                num_heads=config_data['num_heads'],
                embd_size=config_data['embd_size'],
                dropout=config_data.get('dropout', 0.1),
                bias=config_data.get('bias', True)
            )
        else:
            # Old format: config is already a GPTConfig object
            config = config_data
        
        # Recreate model
        model = GPT(config=config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load the CORRECT tokenizer (full vocabulary) - UPDATED PATH
        tokenizer_path = os.path.join(root_path, '..', 'data', 'processed', 'models', 'tokenizers', 'polish_bpe_32k.json')
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        print(f"Loaded trained model from {valid_model_path}")
        print(f"Using full vocabulary tokenizer ({tokenizer.get_vocab_size()} tokens)")
        print(f"Model size: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        print(f"Training loss: {checkpoint['loss']:.4f}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Try training a new model: python src/train/train.py")
        return None, None


def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=0.5, top_k=50, device='cpu'):
    """Generate text from a prompt."""
    model.eval()
    model.to(device)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    # Generate text - REMOVED unsupported parameters
    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature
            # Removed top_k and do_sample as they're not supported by our GPT.generate()
        )
    
    # Decode and clean up
    generated_text = tokenizer.decode(generated[0].tolist())
    
    # Post-processing for Polish text
    generated_text = (
        generated_text
        .replace('�', '')  # Remove replacement characters
        .replace('  ', ' ')  # Fix double spaces
        .strip()
    )
    
    return generated_text


def main():
    """Main inference function."""
    print("Polish Language Model Inference")
    print("=" * 50)
    
    # Load trained model
    print("Loading trained model...")
    model, tokenizer = load_trained_model()
    
    if model is None:
        return
     
    # Interactive loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Type Polish prompts to see the model generate text!")
    print("Example: 'Polska to piekny kraj', 'Warszawa jest', 'Najwieksze miasta'")
    
    while True:
        try:
            # Get user input
            prompt = input("\nEnter your prompt (or 'quit' to exit): ")
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt.strip():
                continue
            
            # Generate and display response
            print(f"\nGenerating response...")
            response = generate(
                model, tokenizer, prompt,
                max_new_tokens=40,
                temperature=0.8,
                device=device
            )
            
            print(f"\nResponse:")
            print(f"{response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()