# Copied from thelmbook.com. Source:
# https://github.com/aburkov/theLMbook/blob/main/news_decoder_language_model.ipynb

# Import required libraries
import os               # For file and path operations (check_file_exists, extract_dataset)
import urllib.request   # For downloading dataset files from URLs
import tarfile          # For extracting .tar.gz dataset archives
import random           # For setting random seeds
from tqdm import tqdm   # For progress bars
import math             # For computing perplexity using exp()
import re               # For preprocessing text (replacing numbers with placeholders)
##import tempfile         # For temporary file handling during extraction
##import shutil           # For file operations during extraction

import torch            # Main PyTorch library for tensor operations and deep learning
import torch.nn as nn   # Neural network modules, layers, and utilities
#import torch.nn.functional as F  # For softmax
from torch.utils.data import DataLoader, IterableDataset  # For efficient data loading
from transformers import AutoTokenizer  # For loading pre-trained tokenizer


#
#
#
#
#import os
#import urllib.request   # For downloading dataset files from URLs
#import tarfile          # For extracting .tar.gz dataset archives
#import random           # For setting random seeds
#import re               # For preprocessing text (replacing numbers with placeholders)
#
#import torch
#import torch.nn as nn
#from torch.utils.data import DataLoader, IterableDataset  # For efficient data loading








# ----------------------------
# Utility Functions
# ----------------------------

def set_seed(seed):
    """
    Sets random seeds for reproducibility across different Python libraries.
    This ensures that random operations give the same results across runs.

    Args:
        seed (int): Seed value for random number generation
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for PyTorch's CPU random number generator
    torch.manual_seed(seed)
    # Set seed for PyTorch's GPU random number generator
    torch.cuda.manual_seed_all(seed)
    # Requests cuDNN to use deterministic algorithms when possible
    # Note: This may impact performance and might not guarantee determinism in all cases
    torch.backends.cudnn.deterministic = True
    # Disables cuDNN's auto-tuner which finds the best algorithm for your specific input size
    # Ensures consistent behavior but might be slower as it doesn't optimize for input sizes
    torch.backends.cudnn.benchmark = False


# ----------------------------
# Dataset Class
# ----------------------------

class IterableTextDataset(IterableDataset):
    """
    An iterable dataset for processing text data in a memory-efficient way.
    Instead of loading all data into memory, it streams data from disk.
    Inherits from PyTorch's IterableDataset for streaming support.

    Args:
        file_path (str): Path to the text file containing sentences
        tokenizer: Tokenizer object for converting text to tokens
        max_length (int): Maximum sequence length to process (default: 30)
    """
    def __init__(self, file_path, tokenizer, max_length=30):
        # Store file path for reading data
        self.file_path = file_path
        # Store tokenizer for text processing
        self.tokenizer = tokenizer
        # Set maximum sequence length to truncate long sequences
        self.max_length = max_length
        self._count_sentences()

    def __iter__(self):
        """
        Creates an iterator over the dataset.
        This method is called when iterating over the dataset.

        Yields:
            tuple: (input_sequence, target_sequence) pairs for language modeling
                  input_sequence is the sequence up to the last token
                  target_sequence is the sequence shifted one position right
        """
        # Open file in read mode with UTF-8 encoding
        with open(self.file_path, 'r', encoding="utf-8") as f:
            # Process each line (sentence) in the file
            for line in f:
                # Remove leading/trailing whitespace
                sentence = line.strip()
                # Replace all numbers with ### placeholder
                # This reduces vocabulary size and helps model generalize
                sentence = re.sub(r"\d+", "###", sentence)

                # Convert sentence to token IDs
                encoded_sentence = self.tokenizer.encode(
                    sentence,
                    max_length=self.max_length,
                    truncation=True
                )

                # Only use sequences with at least 2 tokens
                # (need at least one input and one target token)
                if len(encoded_sentence) >= 2:
                    # Input is all tokens except last
                    input_seq = encoded_sentence[:-1]
                    # Target is all tokens except first
                    target_seq = encoded_sentence[1:]
                    # Convert to PyTorch tensors and yield
                    yield torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)
    def __len__(self):
        return self._num_sentences

    def _count_sentences(self):
        print(f"\nCounting sentences in {self.file_path}...")
        with open(self.file_path, 'r', encoding="utf-8") as f:
            self._num_sentences = sum(1 for _ in f)
        print(f"\nFound {self._num_sentences} sentences in {self.file_path}.")


## ----------------------------
## Download and prepare data
## ----------------------------

def check_file_exists(filename):
    """
    Checks if a file exists in the current directory.
    Args:
        filename (str): Name of the file to check
    Returns:
        bool: True if file exists, False otherwise
    """
    return os.path.exists(filename)


def download_file(url, filename="./data/news.tar.gz"):
    """
    NOTE: adapted (and shortened) to include filename as argument.
    Downloads a file from the given URL if it doesn't exist locally.
    Uses a custom User-Agent to help prevent download blocks.

    Args:
        url (str): URL of the file to download
        filename (str, optional): saved filename with path
    Returns:
        str: Name of the downloaded file ("news.tar.gz")
    """

    if not check_file_exists(filename):
        print(f"\nDownloading dataset from {url}...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            with open(filename, "wb") as out_file:
                out_file.write(response.read())
        print("\nDownload completed.")
    else:
        print(f"\n{filename} already downloaded.")
    return filename


def extract_dataset(filename, extract_dir="./data/"):
    """
    NOTE: Adapted to change extract_dir
    Extracts train.txt and test.txt from the downloaded archive.
    Includes debug information about archive contents.

    Args:
        filename (str): Name of the archive file
        extract_dir (str, optional): Name of the directory to extract to
    Returns:
        tuple: Paths to extracted train and test files
    """
    data_dir = os.path.join(os.path.dirname(filename), "news")
    train_path = os.path.join(data_dir, "train.txt")
    test_path = os.path.join(data_dir, "test.txt")
    print(data_dir, train_path, test_path)

    if check_file_exists(train_path) and check_file_exists(test_path):
        print("\nData files already extracted.")
        return train_path, test_path

    print("\nListing archive contents:")
    with tarfile.open(filename, "r:gz") as tar:
        for member in tar.getmembers():
            print(f"\nArchive member: {member.name}")

        print("\nExtracting files...")
        # Extract to current directory first
        tar.extractall(extract_dir)

    if not (check_file_exists(train_path) and check_file_exists(test_path)):
        raise FileNotFoundError(f"\nRequired files not found in the archive. Please check the paths above.")

    print("\nExtraction completed.")
    return train_path, test_path


def create_datasets(train_file, test_file, tokenizer, max_length=30):
    """
    Creates IterableTextDataset objects for training and testing.
    These datasets will stream data from disk instead of loading it all into memory.

    Args:
        train_file (str): Path to training data file
        test_file (str): Path to test data file
        tokenizer: Tokenizer object for text processing

    Returns:
        tuple: (train_dataset, test_dataset) - Dataset objects for training and testing
    """
    # Create training dataset
    train_dataset = IterableTextDataset(train_file, tokenizer, max_length)
    # Create test dataset
    test_dataset = IterableTextDataset(test_file, tokenizer, max_length)

    # Print dataset sizes
    print(f"\nTraining sentences: {len(train_dataset)}")
    print(f"\nTest sentences: {len(test_dataset)}")

    return train_dataset, test_dataset


def create_collate_fn(tokenizer):
    """
    Creates a collate function for batching sequences of different lengths.
    This function pads shorter sequences to match the longest sequence in the batch.

    Args:
        tokenizer: Tokenizer object containing padding token information

    Returns:
        function: Collate function that handles padding in batches
    """
    def collate_fn(batch):
        # Separate inputs and targets from batch
        input_seqs, target_seqs = zip(*batch)
        # Get padding token ID from tokenizer
        pad_index = tokenizer.pad_token_id
        # Pad input sequences to same length
        input_padded = nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=pad_index)
        # Pad target sequences to same length
        target_padded = nn.utils.rnn.pad_sequence(target_seqs, batch_first=True, padding_value=pad_index)
        return input_padded, target_padded
    return collate_fn


def create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn):
    """
    Creates DataLoader objects for efficient data iteration.

    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size (int): Number of sequences per batch
        collate_fn: Function to handle padding and batch creation

    Returns:
        tuple: (train_dataloader, test_dataloader) - DataLoader objects for
               iterating over batches of data with proper padding
    """
    # Create training data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,    # Function to handle padding
        num_workers=0             # Number of worker processes (0 = single process)
    )
    # Create test data loader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )
    return train_dataloader, test_dataloader


def download_and_prepare_data(url, batch_size, tokenizer, max_length=30):
    """
    Main function to handle the complete data preparation pipeline.
    Downloads data, extracts it, and creates necessary dataset objects.

    Args:
        url (str): URL where the dataset archive can be downloaded
        batch_size (int): Batch size for data loading
        tokenizer: Tokenizer object for text processing
        max_length (int): Maximum sequence length for tokenization (default: 30)

    Returns:
        tuple: (train_dataloader, test_dataloader) - Ready-to-use data loaders
    """
    # Step 1: Download dataset archive from URL
    filename = download_file(url)

    # Step 2: Extract training and test files from archive
    train_file, test_file = extract_dataset(filename)

    # Step 3: Create dataset objects for streaming data
    train_dataset, test_dataset = create_datasets(train_file, test_file, tokenizer, max_length)

    # Step 4: Create function to handle batch creation
    collate_fn = create_collate_fn(tokenizer)

    # Step 5: Create and return data loaders
    return create_dataloaders(train_dataset, test_dataset, batch_size, collate_fn)


# ----------------------------
# Evaluation Functions
# ----------------------------

def compute_loss_and_perplexity(model, dataloader, tokenizer, criterion, device, max_sentences=1000):
    """
    Evaluates model performance by computing loss and perplexity on data.

    Args:
        model (nn.Module): The language model to evaluate
        dataloader (DataLoader): Data loader containing batched sequences
        tokenizer: Tokenizer for handling special tokens like padding
        criterion: Loss function (usually CrossEntropyLoss)
        device: Device to run computation on (cuda/cpu)
        max_sentences (int): Maximum number of sentences to evaluate (default: 1000)
                           Limits evaluation to a subset for faster validation

    Returns:
        tuple: (average_loss, perplexity, sentences_processed)
               - average_loss: Mean loss per token (excluding padding)
               - perplexity: exp(average_loss), lower is better
    """
    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Initialize counters for loss calculation
    total_loss = 0.0          # Accumulator for total loss across all batches
    total_tokens = 0          # Counter for total number of tokens (excluding padding)
    sentences_processed = 0    # Counter for number of sentences processed

    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Iterate through data with progress bar
        for input_seq, target_seq in tqdm(dataloader, desc="Evaluating", leave=False):
            # Move input and target sequences to specified device
            input_seq = input_seq.to(device)      # Shape: (batch_size, seq_len)
            target_seq = target_seq.to(device)    # Shape: (batch_size, seq_len)

            # Get current batch size (might be smaller for last batch)
            batch_size_current = input_seq.size(0)

            # Forward pass through the model
            logits = model(input_seq)             # Shape: (batch_size, seq_len, vocab_size)

            # Reshape logits and target for loss calculation
            logits = logits.reshape(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            target = target_seq.reshape(-1)              # Shape: (batch_size * seq_len)

            # Create mask to exclude padding tokens
            mask = target != tokenizer.pad_token_id

            # Compute loss only on non-padded tokens
            loss = criterion(logits[mask], target[mask])

            # Update counters
            loss_value = loss.item() * mask.sum().item()  # Total loss for this batch
            total_loss += loss_value                      # Accumulate batch loss
            total_tokens += mask.sum().item()             # Count non-padding tokens

            # Update sentence counter and check if we've reached maximum
            sentences_processed += batch_size_current
            if sentences_processed >= max_sentences:
                break

    # Calculate final metrics
    average_loss = total_loss / total_tokens           # Normalize loss by number of tokens
    perplexity = math.exp(average_loss)               # Convert loss to perplexity

    return average_loss, perplexity

def generate_text(model, start_string, tokenizer, device, max_length=50):
    """
    Generates text continuation from a given start string using greedy decoding.

    Args:
        model (nn.Module): Trained language model
        start_string (str): Initial text to continue from
        tokenizer: Tokenizer for text processing
        device: Device to run generation on (cuda/cpu)
        max_length (int): Maximum length of generated sequence

    Returns:
        str: Generated text continuation
    """
    # Set model to evaluation mode to disable dropout and other training-specific behaviors
    model.eval()

    # Convert input string to token indices
    input_indices = tokenizer.encode(start_string, add_special_tokens=False)

    # Convert indices to tensor and move to specified device (GPU/CPU)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    # Keep track of all generated tokens, starting with input sequence
    generated_indices = input_indices.copy()

    # Generate tokens until we hit max length or end-of-sequence token
    for _ in range(max_length - len(input_indices)):
        # Get model predictions for the entire sequence
        logits = model(input_tensor)
        # Only take predictions for the last token position
        logits = logits[:, -1, :]

        # Prevent the model from generating unknown tokens by setting their probability to negative infinity
        if tokenizer.unk_token_id is not None:
            logits[:, tokenizer.unk_token_id] = float("-inf")

        # Greedy decoding: select the token with highest probability
        next_token = torch.argmax(logits, dim=-1)

        # Add the chosen token to our generated sequence
        generated_indices.append(next_token.item())

        # If we generate an end-of-sequence token, stop generation
        if next_token.item() == tokenizer.eos_token_id:
            break

        # Add the new token to input tensor for next iteration
        input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=1)

    # Convert token indices back to text, removing any special tokens
    return tokenizer.decode(generated_indices, skip_special_tokens=True)

def save_model(model, tokenizer, model_name):
    """
    Saves the model state dictionary and tokenizer using the specified model name.

    Args:
        model (nn.Module): The trained model to save
        tokenizer: The tokenizer used with the model
        model_name (str): Name to use for the saved model files
    """
    # Create the models directory if it doesn't exist
    save_dir = os.path.join("models", model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save the model state dictionary and configuration
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": len(tokenizer),
            "emb_dim": model.embedding.embedding_dim,
            "num_heads": len(model.layers[0].attn.heads),
            "num_blocks": len(model.layers),
            "pad_idx": model.embedding.padding_idx
        }
    }, model_path)

    # Save the tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)

    print(f"Model and tokenizer saved as '{model_name}'")

def load_model(model_name, device=None):
    """
    Loads a saved model and tokenizer using the model name.

    Args:
        model_name (str): Name of the model to load
        device: Device to load the model onto (if None, uses available device)

    Returns:
        tuple: (loaded_model, loaded_tokenizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = os.path.join("models", model_name)

    # Check if model exists
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"No saved model found with name '{model_name}'")

    # Load the tokenizer
    tokenizer_path = os.path.join(save_dir, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load the model state and config
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create a new model instance with the saved configuration
    model = DecoderLanguageModel(
        vocab_size=checkpoint["model_config"]["vocab_size"],
        emb_dim=checkpoint["model_config"]["emb_dim"],
        num_heads=checkpoint["model_config"]["num_heads"],
        num_blocks=checkpoint["model_config"]["num_blocks"],
        pad_idx=checkpoint["model_config"]["pad_idx"]
    )

    # Load the saved state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"\nModel '{model_name}' loaded successfully")
    return model, tokenizer

# --------------------------------------------------------
#Model Classes
#Decoder Transformer language model classes and the initialization method:

# ----------------------------
# Weight Initialization and Core Functions
# This section contains utility functions for weight initialization
# and core computational functions used throughout the model
# ----------------------------

def initialize_weights(model):
    """
    NOTE: only DecoderLanguageModel
    Initialize the weights of different model components using appropriate schemes.
    Each layer type receives specialized initialization for optimal training.
    """
    for module in model.modules():
#        if isinstance(module, nn.Linear):
#            # Xavier uniform initialization for linear layers
#            # Helps maintain variance across network layers
#            nn.init.xavier_uniform_(module.weight)
#            if module.bias is not None:
#                nn.init.zeros_(module.bias)  # Initialize biases to zero
#        elif isinstance(module, nn.Embedding):
#            # Initialize embedding layers with normal distribution
#            nn.init.normal_(module.weight, mean=0, std=0.02)
#            if module.padding_idx is not None:
#                # Ensure padding tokens have zero embeddings
#                with torch.no_grad():
#                    module.weight[module.padding_idx].fill_(0)
#        elif isinstance(module, AttentionHead):
#            # Initialize query, key, and value projection matrices
#            # Xavier uniform helps maintain good gradient flow
#            nn.init.xavier_uniform_(module.W_Q)
#            nn.init.xavier_uniform_(module.W_K)
#            nn.init.xavier_uniform_(module.W_V)
#        elif isinstance(module, MultiHeadAttention):
#            # Initialize output projection matrix for attention mechanism
#            nn.init.xavier_uniform_(module.W_O)
#        elif isinstance(module, DecoderLanguageModel):
#            # Initialize final output projection layer
#            nn.init.xavier_uniform_(module.output)
#        elif isinstance(module, RMSNorm):
#            # Initialize RMSNorm scale parameters to ones
#            # This starts with identity transformation
#            nn.init.ones_(module.scale)
#        elif isinstance(module, MLP):
#            # Initialize feed-forward network parameters
#            nn.init.xavier_uniform_(module.W_1)
#            nn.init.xavier_uniform_(module.W_2)
#            nn.init.zeros_(module.B_1)
#            nn.init.zeros_(module.B_2)

        if isinstance(module, DecoderLanguageModel):
            # Initialize final output projection layer
            nn.init.xavier_uniform_(module.output)


def rope(x, theta_base=10000.0):
    """
    Implements Rotary Position Embedding (RoPE) for transformer attention.
    RoPE encodes position information through rotation matrices applied to pairs of dimensions.

    Args:
        x: Input tensor of shape (batch_size, seq_len, emb_dim)
        theta_base: Base for computing rotation frequencies (default: 10000.0)

    Returns:
        Tensor with position information encoded through rotations
    """
    batch_size, seq_len, emb_dim = x.size()
    assert emb_dim % 2 == 0, "Embedding dimensionality must be even for RoPE"

    # Generate sequence position indices
    pos = torch.arange(0, seq_len, dtype=torch.float32, device=x.device)
    pos = pos.unsqueeze(0).expand(batch_size, seq_len)

    # Compute frequency bands for each dimension pair
    # Modified: frequencies start from p=1 and use (p-1) in exponent
    p = torch.arange(1, emb_dim // 2 + 1, dtype=torch.float32, device=x.device)
    theta_p = 1.0 / (theta_base ** (2 * (p - 1) / emb_dim))

    # Compute rotation angles for each position and frequency
    pos = pos.unsqueeze(-1)
    theta = pos * theta_p

    # Compute rotation components
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Split input into alternating dimensions
    x1 = x[..., 0::2]  # Dimensions at indices 0,2,4,...
    x2 = x[..., 1::2]  # Dimensions at indices 1,3,5,...

    # Apply 2D rotations to each pair
    x_rotated_1 = x1 * cos_theta - x2 * sin_theta
    x_rotated_2 = x1 * sin_theta + x2 * cos_theta

    # Recombine rotated pairs into final output
    x_rotated = torch.stack((x_rotated_1, x_rotated_2), dim=-1).reshape(batch_size, seq_len, emb_dim)

    return x_rotated


# ----------------------------
# Model Components
# This section contains the building blocks of the transformer decoder
# including normalization, attention, and feed-forward layers
# ----------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    A simplified alternative to Layer Normalization that only uses RMS statistics
    """
    def __init__(self, emb_dim, epsilon=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(emb_dim))  # Learnable scale parameter
        self.epsilon = epsilon  # Small constant for numerical stability

    def forward(self, x):
        # Compute root mean square normalization
        squared_x = x ** 2
        mean_squared = torch.mean(squared_x, dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.epsilon)

        # Normalize and scale
        x_normalized = x / rms
        output = x_normalized * self.scale
        return output

class AttentionHead(nn.Module):
    """
    Single head of self-attention
    Transforms input using learned projections and computes scaled dot-product attention
    """
    def __init__(self, emb_dim, d_h):
        super().__init__()
        # Initialize projection matrices for queries, keys, and values
        self.W_Q = nn.Parameter(torch.rand(emb_dim, d_h))
        self.W_K = nn.Parameter(torch.rand(emb_dim, d_h))
        self.W_V = nn.Parameter(torch.rand(emb_dim, d_h))
        self.d_h = d_h  # Dimensionality of attention head

    def forward(self, x, mask):
        # Project input into query, key, and value spaces
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # Apply rotary position embeddings to queries and keys
        Q, K = rope(Q), rope(K)

        # Compute attention scores with scaling factor
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_h)

        # Apply causal mask and attention weights
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)

        return attention_weights @ V

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism
    Allows the model to jointly attend to information from different positions
    """
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        d_h = emb_dim // num_heads  # Dimensionality of each attention head

        # Create multiple attention heads
        self.heads = nn.ModuleList([
            AttentionHead(emb_dim, d_h)
            for _ in range(num_heads)
        ])

        # Output projection matrix
        self.W_O = nn.Parameter(torch.rand(emb_dim, emb_dim))

    def forward(self, x, mask):
        # Process input through each attention head
        head_outputs = [head(x, mask) for head in self.heads]

        # Concatenate outputs and project to final dimensionality
        x = torch.cat(head_outputs, dim=-1)
        return x @ self.W_O

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for transformer feed-forward network
    Uses a larger intermediate dimensionality (4x) with ReLU activation
    """
    def __init__(self, emb_dim):
        super().__init__()
        # Initialize weights and biases for two-layer feed-forward network
        self.W_1 = nn.Parameter(torch.rand(emb_dim, emb_dim * 4))
        self.B_1 = nn.Parameter(torch.rand(emb_dim * 4))
        self.W_2 = nn.Parameter(torch.rand(emb_dim * 4, emb_dim))
        self.B_2 = nn.Parameter(torch.rand(emb_dim))

    def forward(self, x):
        # First linear transformation and activation
        x = x @ self.W_1 + self.B_1
        x = torch.relu(x)

        # Second linear transformation
        x = x @ self.W_2 + self.B_2
        return x


class DecoderBlock(nn.Module):
    """
    Single transformer decoder block
    Combines self-attention and feed-forward layers with residual connections
    """
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        # Layer components
        self.norm1 = RMSNorm(emb_dim)
        self.attn = MultiHeadAttention(emb_dim, num_heads)
        self.norm2 = RMSNorm(emb_dim)
        self.mlp = MLP(emb_dim)

    def forward(self, x, mask):
        # Self-attention sub-block with residual connection
        attn_out = self.attn(self.norm1(x), mask)
        x = x + attn_out

        # Feed-forward sub-block with residual connection
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x


class DecoderLanguageModel(nn.Module):
    """
    Complete decoder-only transformer language model
    Processes input sequences using multiple decoder blocks and projects to vocabulary
    """
    def __init__(self, vocab_size, emb_dim, num_heads, num_blocks, pad_idx) -> None:
        super().__init__()
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(emb_dim, num_heads) for _ in range(num_blocks)
        ])

        # Output projection to vocabulary size
        self.output = nn.Parameter(torch.rand(emb_dim, vocab_size))

    def forward(self, x):# -> Any:
        # Embed input tokens
        x = self.embedding(x)

        # Create causal attention mask
        _, seq_len, _ = x.size()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))

        # Process through decoder blocks
        for layer in self.layers:
            x = layer(x, mask)

        # Project to vocabulary distribution
        return x @ self.output


