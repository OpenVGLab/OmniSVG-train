"""
OmniSVG Training Script

Supports:
- 4B model (Qwen2.5-VL-3B based)
- 8B model (Qwen2.5-VL-7B based)
- Flash Attention 2 acceleration
- Training from local files or HuggingFace datasets
- Multi-task learning (Text-to-SVG and Image-to-SVG)
- Distributed training with Accelerate

Usage:
    # Train 4B model
    accelerate launch train.py --model_size 4B --data_dir ./data
    
    # Train 8B model with Flash Attention
    accelerate launch train.py --model_size 8B --use_flash_attn --data_dir ./data
    
    # Train with HuggingFace data (auto-download)
    accelerate launch train.py --model_size 4B --use_hf_data --datasets illustration icon
    
    # Resume training from OmniSVG checkpoint
    accelerate launch train.py --model_size 4B --resume_from_checkpoint auto
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)
from accelerate import Accelerator
from safetensors.torch import load_file

# Local imports
from utils import (
    OmniSVGConfig,
    TokenizationConfig,
    TrainConfig,
    OmniSVGDataset,
    create_dataloader,
    download_omnisvg_data,
    list_available_datasets,
)
from decoder import SketchDecoder

# For Qwen2.5-VL
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("Warning: qwen_vl_utils not found. Image processing may be limited.")
    process_vision_info = None


# ============================================================================
# Model Configuration - Define checkpoint mappings
# ============================================================================

MODEL_DEFAULTS = {
    "4B": {
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "checkpoint": "OmniSVG/OmniSVG1.1_4B",
    },
    "8B": {
        "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "checkpoint": "OmniSVG/OmniSVG1.1_8B",
    },
}


# ============================================================================
# Model Loading Utilities
# ============================================================================

def is_hf_repo_id(path: str) -> bool:
    """
    Check if a string looks like a HuggingFace repo ID.
    
    Args:
        path: String to check
    
    Returns:
        True if it looks like a HuggingFace repo ID
    """
    # HuggingFace repo IDs have format "owner/repo_name"
    # Local paths typically have more slashes or start with . / ~
    if path is None:
        return False
    
    # If path exists locally, it's not a repo ID
    if os.path.exists(path):
        return False
    
    # Check for HuggingFace repo ID pattern
    parts = path.split("/")
    if len(parts) == 2 and all(part and not part.startswith(('.', '~')) for part in parts):
        return True
    
    return False


def download_omnisvg_checkpoint(
    repo_id_or_model_size: str, 
    cache_dir: str = "./checkpoints"
) -> str:
    """
    Download OmniSVG checkpoint from HuggingFace Hub.
    
    Args:
        repo_id_or_model_size: HuggingFace repo ID (e.g., "OmniSVG/OmniSVG1.1_4B") 
                               or model size ("4B" or "8B")
        cache_dir: Directory to cache downloaded checkpoints
    
    Returns:
        Path to downloaded checkpoint
    """
    # If it's a model size, get the default checkpoint
    if repo_id_or_model_size in MODEL_DEFAULTS:
        repo_id = MODEL_DEFAULTS[repo_id_or_model_size]["checkpoint"]
    else:
        repo_id = repo_id_or_model_size
    
    print(f"Downloading OmniSVG checkpoint from {repo_id}...")
    
    # Create local directory name from repo ID
    local_name = repo_id.replace("/", "_")
    local_dir = os.path.join(cache_dir, local_name)
    
    # Download the repository
    local_path = snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        resume_download=True,
    )
    
    print(f"Checkpoint downloaded to: {local_path}")
    return local_path


def find_checkpoint_file(checkpoint_path: str) -> Optional[str]:
    """
    Find the model checkpoint file in a directory.
    
    Args:
        checkpoint_path: Directory containing checkpoint files
    
    Returns:
        Path to checkpoint file or None if not found
    """
    # Priority order for checkpoint files
    checkpoint_files = [
        "pytorch_model.bin",
        "model.safetensors", 
        "adapter_model.safetensors",
        "adapter_model.bin",
    ]
    
    for filename in checkpoint_files:
        path = os.path.join(checkpoint_path, filename)
        if os.path.exists(path):
            return path
    
    # Also check for sharded checkpoints
    import glob
    sharded_patterns = [
        "pytorch_model-*.bin",
        "model-*.safetensors",
    ]
    
    for pattern in sharded_patterns:
        matches = glob.glob(os.path.join(checkpoint_path, pattern))
        if matches:
            # Return the directory for sharded checkpoints
            return checkpoint_path
    
    return None


def load_checkpoint_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from checkpoint file or directory.
    
    Args:
        checkpoint_path: Path to checkpoint file or directory
    
    Returns:
        State dictionary
    """
    import glob
    
    if os.path.isfile(checkpoint_path):
        # Single file
        if checkpoint_path.endswith(".safetensors"):
            return load_file(checkpoint_path)
        else:
            return torch.load(checkpoint_path, map_location='cpu')
    
    elif os.path.isdir(checkpoint_path):
        # Check for sharded safetensors
        safetensor_files = sorted(glob.glob(os.path.join(checkpoint_path, "model-*.safetensors")))
        if safetensor_files:
            print(f"Loading sharded safetensors from {len(safetensor_files)} files...")
            state_dict = {}
            for sf_file in safetensor_files:
                state_dict.update(load_file(sf_file))
            return state_dict
        
        # Check for sharded pytorch model
        bin_files = sorted(glob.glob(os.path.join(checkpoint_path, "pytorch_model-*.bin")))
        if bin_files:
            print(f"Loading sharded pytorch model from {len(bin_files)} files...")
            state_dict = {}
            for bin_file in bin_files:
                state_dict.update(torch.load(bin_file, map_location='cpu'))
            return state_dict
        
        # Single file in directory
        model_file = find_checkpoint_file(checkpoint_path)
        if model_file and os.path.isfile(model_file):
            if model_file.endswith(".safetensors"):
                return load_file(model_file)
            else:
                return torch.load(model_file, map_location='cpu')
    
    raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")


def load_model(
    model_size: str,
    pix_len: int,
    text_len: int,
    use_flash_attn: bool = True,
    checkpoint_path: Optional[str] = None,
    device_map: str = "auto",
) -> nn.Module:
    """
    Load OmniSVG model with appropriate settings.
    
    Args:
        model_size: "4B" or "8B"
        pix_len: Maximum SVG token length
        text_len: Maximum text length
        use_flash_attn: Whether to use Flash Attention 2
        checkpoint_path: Path to checkpoint, HuggingFace repo ID, or "auto"
        device_map: Device mapping strategy
    
    Returns:
        Loaded model
    """
    if model_size not in MODEL_DEFAULTS:
        raise ValueError(f"Invalid model_size: {model_size}. Must be one of {list(MODEL_DEFAULTS.keys())}")
    
    base_model = MODEL_DEFAULTS[model_size]["base_model"]
    default_checkpoint = MODEL_DEFAULTS[model_size]["checkpoint"]
    
    # Set attention implementation
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    
    print(f"=" * 60)
    print(f"Loading OmniSVG {model_size} Model")
    print(f"=" * 60)
    print(f"Base model: {base_model}")
    print(f"Attention implementation: {attn_implementation}")
    print(f"Max SVG tokens: {pix_len}")
    print(f"Max text length: {text_len}")
    
    # Initialize model from base
    model = SketchDecoder(
        pix_len=pix_len,
        text_len=text_len,
        model_path=base_model,
        attn_implementation=attn_implementation,
    )
    
    # Load checkpoint if specified
    if checkpoint_path:
        resolved_path = None
        
        # Handle "auto" keyword - use default checkpoint for model size
        if checkpoint_path.lower() == "auto":
            print(f"\nUsing 'auto' - downloading default checkpoint: {default_checkpoint}")
            resolved_path = download_omnisvg_checkpoint(default_checkpoint)
        
        # Handle HuggingFace repo ID (e.g., "OmniSVG/OmniSVG1.1_4B")
        elif is_hf_repo_id(checkpoint_path):
            print(f"\nDetected HuggingFace repo ID: {checkpoint_path}")
            resolved_path = download_omnisvg_checkpoint(checkpoint_path)
        
        # Handle local path
        elif os.path.exists(checkpoint_path):
            print(f"\nUsing local checkpoint: {checkpoint_path}")
            resolved_path = checkpoint_path
        
        else:
            # Try to download as HuggingFace repo
            print(f"\nPath not found locally, attempting to download from HuggingFace: {checkpoint_path}")
            try:
                resolved_path = download_omnisvg_checkpoint(checkpoint_path)
            except Exception as e:
                print(f"Warning: Failed to download checkpoint: {e}")
                resolved_path = None
        
        # Load the checkpoint
        if resolved_path:
            print(f"\nLoading checkpoint from: {resolved_path}")
            
            try:
                # Find and load checkpoint file
                if os.path.isfile(resolved_path):
                    model_file = resolved_path
                else:
                    model_file = find_checkpoint_file(resolved_path)
                
                if model_file:
                    print(f"Found checkpoint: {model_file}")
                    state_dict = load_checkpoint_state_dict(model_file if os.path.isfile(model_file) else resolved_path)
                    
                    # Load state dict with detailed logging
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    
                    print(f"\nCheckpoint loading summary:")
                    print(f"  - Total keys in checkpoint: {len(state_dict)}")
                    print(f"  - Missing keys: {len(missing_keys)}")
                    print(f"  - Unexpected keys: {len(unexpected_keys)}")
                    
                    if missing_keys and len(missing_keys) < 20:
                        print(f"  - Missing key examples: {missing_keys[:5]}")
                    if unexpected_keys and len(unexpected_keys) < 20:
                        print(f"  - Unexpected key examples: {unexpected_keys[:5]}")
                    
                    print("\n✓ Checkpoint loaded successfully!")
                else:
                    print(f"Warning: No checkpoint file found in {resolved_path}")
                    print("Available files:", os.listdir(resolved_path) if os.path.isdir(resolved_path) else "N/A")
                    
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                import traceback
                traceback.print_exc()
                print("Continuing with base model weights...")
    else:
        print("\nNo checkpoint specified - using base model weights")
    
    print(f"=" * 60)
    
    return model


# ============================================================================
# Collate Functions
# ============================================================================

def create_collate_fn(
    processor: Any,
    text_len: int = 800,
    text_only_ratio: float = 0.5,
):
    """
    Create collate function for DataLoader.
    
    Args:
        processor: HuggingFace processor
        text_len: Maximum text length
        text_only_ratio: Ratio of text-only samples
    
    Returns:
        Collate function
    """
    system_prompt = "You are an expert SVG code generator."
    
    task_counter = {"total": 0, "text": 0}

    def collate_fn(batch):
        text_oris, pil_images, pix_seq_lists = zip(*batch)
        
        batch_messages = []
        batch_task_types = []
        
        task_assignments = []
        for _ in range(len(text_oris)):
            task_counter["total"] += 1
            target_text_count = task_counter["total"] // 2
            
            if task_counter["text"] < target_text_count:
                task_assignments.append("text")
                task_counter["text"] += 1
            else:
                task_assignments.append("image")
        
        indices = list(range(len(task_assignments)))
        np.random.shuffle(indices)
        task_assignments = [task_assignments[i] for i in indices]
        
        for text_ori, pil_image, task_type in zip(text_oris, pil_images, task_assignments):
            if task_type == 'text':
                # Text-to-SVG task
                messages = [{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": [{"type": "text", "text": f"Generate SVG code for this text description: {text_ori}"}]
                }]
                batch_task_types.append("text")
            else:
                # Image-to-SVG task
                messages = [{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate SVG code that accurately represents this image:"},
                        {"type": "image", "image": pil_image},
                    ]
                }]
                batch_task_types.append("image")
            
            batch_messages.append(messages)
        
        return batch_messages, list(pix_seq_lists), batch_task_types
    
    return collate_fn




def process_mixed_batch(
    batch_messages: List[Any],
    pix_seq_lists: List[List[int]],
    batch_task_types: List[str],
    processor: Any,
    config: OmniSVGConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], 
           Optional[torch.Tensor], torch.Tensor, Dict[str, List[int]]]:
    """
    Process a mixed batch of text and image tasks.
    
    Returns:
        input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks
    """
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    task_masks = {'text': [], 'image': []}
    
    pad_token_id = config.tokenization.pad_token_id
    max_len = config.training.max_seq_length + config.training.text_max_length
    
    # Separate tasks
    text_indices = []
    image_indices = []
    image_messages = []
    
    for i, (task_type, messages) in enumerate(zip(batch_task_types, batch_messages)):
        if task_type == "text":
            text_indices.append(i)
            task_masks['text'].append(i)
        else:
            image_indices.append(i)
            task_masks['image'].append(i)
            image_messages.append(messages)
    
    # Process text-only samples
    if text_indices:
        text_inputs = []
        for i in text_indices:
            text_input = processor.apply_chat_template(
                batch_messages[i], tokenize=False, add_generation_prompt=True
            )
            text_inputs.append(text_input)
        
        inputs = processor(
            text=text_inputs,
            padding=False,
            truncation=False,
            return_tensors=None
        )
        
        for idx, i in enumerate(text_indices):
            input_ids, attention_mask, labels = _process_sample(
                inputs['input_ids'][idx],
                inputs['attention_mask'][idx],
                pix_seq_lists[i],
                max_len,
                pad_token_id,
            )
            batch_input_ids.append((i, input_ids))
            batch_attention_mask.append((i, attention_mask))
            batch_labels.append((i, labels))
    
    # Process image samples
    pixel_values = None
    image_grid_thw = None
    
    if image_indices and process_vision_info is not None:
        all_text_inputs = []
        all_image_inputs = []
        
        for messages in image_messages:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            all_text_inputs.append(text_input)
            all_image_inputs.extend(image_inputs)
        
        inputs = processor(
            text=all_text_inputs,
            images=all_image_inputs,
            padding=False,
            truncation=False,
            return_tensors="pt"
        )
        
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        
        for idx, i in enumerate(image_indices):
            input_ids, attention_mask, labels = _process_sample(
                inputs['input_ids'][idx].tolist(),
                inputs['attention_mask'][idx].tolist(),
                pix_seq_lists[i],
                max_len,
                pad_token_id,
            )
            batch_input_ids.append((i, input_ids))
            batch_attention_mask.append((i, attention_mask))
            batch_labels.append((i, labels))
    
    # Sort by original index and stack
    batch_input_ids.sort(key=lambda x: x[0])
    batch_attention_mask.sort(key=lambda x: x[0])
    batch_labels.sort(key=lambda x: x[0])
    
    input_ids = torch.stack([x[1] for x in batch_input_ids])
    attention_mask = torch.stack([x[1] for x in batch_attention_mask])
    labels = torch.stack([x[1] for x in batch_labels])
    
    return input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks


def _process_sample(
    base_input_ids: List[int],
    base_attention_mask: List[int],
    pix_seq: List[int],
    max_len: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a single sample."""
    current_input_ids = base_input_ids + pix_seq
    current_attention_mask = base_attention_mask + [1] * len(pix_seq)
    
    instruction_len = len(base_input_ids)
    current_labels = [-100] * instruction_len + pix_seq
    
    pad_len = max_len - len(current_input_ids)
    
    if pad_len > 0:
        input_ids = [pad_token_id] * pad_len + current_input_ids
        attention_mask = [0] * pad_len + current_attention_mask
        labels = [-100] * pad_len + current_labels
    else:
        input_ids = current_input_ids[:max_len]
        attention_mask = current_attention_mask[:max_len]
        labels = current_labels[:max_len] if len(current_labels) <= max_len else current_labels[:max_len]
    
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


# ============================================================================
# Loss Computation
# ============================================================================

def compute_task_specific_losses(
    outputs: Any,
    labels: torch.Tensor,
    task_masks: Dict[str, List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute separate losses for text and image tasks.
    
    Returns:
        text_loss, image_loss
    """
    batch_size = outputs.logits.size(0)
    device = outputs.logits.device
    
    logits = outputs.logits[:, :-1].contiguous()
    labels = labels[:, 1:].contiguous().to(device)
    
    loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    
    per_token_loss = loss_fct(
        logits.view(-1, logits.size(-1)),
        labels.view(-1)
    ).view(batch_size, -1)
    
    valid_tokens = (labels != -100).float()
    per_sample_loss = per_token_loss.sum(dim=1) / (valid_tokens.sum(dim=1) + 1e-8)
    
    # Separate by task
    text_losses = [per_sample_loss[i] for i in task_masks['text']]
    image_losses = [per_sample_loss[i] for i in task_masks['image']]
    
    text_loss = torch.stack(text_losses).mean() if text_losses else torch.tensor(0.0, device=device)
    image_loss = torch.stack(image_losses).mean() if image_losses else torch.tensor(0.0, device=device)
    
    return text_loss, image_loss


# ============================================================================
# Training Functions
# ============================================================================

def train(args, config: OmniSVGConfig):
    """Main training function."""
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps
    )
    
    # Set seed
    set_seed(config.training.seed)
    
    # Get base model path
    base_model_path = MODEL_DEFAULTS[config.model_size]["base_model"]
    accelerator.print(f"Using base model: {base_model_path}")
    accelerator.print(f"Model size: {config.model_size}")
    
    # Initialize tokenizer and processor
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, padding_side="left"
    )
    processor = AutoProcessor.from_pretrained(
        base_model_path, padding_side="left"
    )
    processor.tokenizer.padding_side = "left"
    
    # Load or download data
    if args.use_hf_data:
        accelerator.print("Downloading datasets from HuggingFace...")
        train_csv, val_csv, svg_folder, png_folder = download_omnisvg_data(
            output_dir=args.data_dir,
            datasets=args.datasets,
            train_ratio=0.95,
            max_token_length=config.training.max_seq_length,
        )
    else:
        # Use local data
        train_csv = config.data.train_meta_file
        val_csv = config.data.val_meta_file
        svg_folder = config.data.svg_folder
        png_folder = config.data.png_folder
    
    accelerator.print(f"Train data: {train_csv}")
    accelerator.print(f"Val data: {val_csv}")
    accelerator.print(f"SVG folder: {svg_folder}")
    accelerator.print(f"PNG folder: {png_folder}")
    
    # Create datasets
    train_dataset = OmniSVGDataset(
        meta_file=train_csv,
        svg_folder=svg_folder,
        png_folder=png_folder,
        max_len=config.training.max_seq_length,
        text_len=config.training.text_max_length,
        tokenizer=tokenizer,
        processor=processor,
        token_config=config.tokenization,
        train_config=config.training,
        require_aug=True,
        text_only_ratio=config.training.text_only_ratio,
    )
    
    val_dataset = OmniSVGDataset(
        meta_file=val_csv,
        svg_folder=svg_folder,
        png_folder=png_folder,
        max_len=config.training.max_seq_length,
        text_len=config.training.text_max_length,
        tokenizer=tokenizer,
        processor=processor,
        token_config=config.tokenization,
        train_config=config.training,
        require_aug=False,
        text_only_ratio=0.5,
    )
    
    # Create collate functions
    train_collate = create_collate_fn(
        processor, 
        text_len=config.training.text_max_length,
        text_only_ratio=config.training.text_only_ratio,
    )
    val_collate = create_collate_fn(
        processor,
        text_len=config.training.text_max_length,
        text_only_ratio=0.5,
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=train_collate,
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=val_collate,
    )
    
    # Initialize model
    model = load_model(
        model_size=config.model_size,
        pix_len=config.training.max_seq_length,
        text_len=config.training.text_max_length,
        use_flash_attn=config.training.use_flash_attn,
        checkpoint_path=args.resume_from_checkpoint if args.resume_from_checkpoint else None,
    )
    
    # Optimizer
    lr = config.training.learning_rate * accelerator.num_processes
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    # Scheduler
    total_steps = len(train_dataloader) * config.training.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5,
    )
    
    # Prepare for distributed training
    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )
    
    # Setup logging
    output_dir = Path(args.output_dir) / args.project_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(output_dir / "logs"))
    
    # Save config
    if accelerator.is_main_process:
        config.save(str(output_dir / "config.yaml"))
        with open(output_dir / "args.json", 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Resume from checkpoint
    starting_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Training loop
    accelerator.print("Starting training...")
    num_update_steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
    
    text_losses = []
    image_losses = []
    grad_norms = []
    
    for epoch in range(starting_epoch, config.training.epochs):
        model.train()
        progress_bar = tqdm(
            total=num_update_steps_per_epoch,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch + 1}"
        )
        
        for batch_messages, pix_seq_lists, batch_task_types in train_dataloader:
            with accelerator.accumulate(model):
                # Process batch
                input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks = \
                    process_mixed_batch(
                        batch_messages, pix_seq_lists, batch_task_types, processor, config
                    )
                
                # Move to device
                input_ids = input_ids.to(accelerator.device)
                attention_mask = attention_mask.to(accelerator.device)
                labels = labels.to(accelerator.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(accelerator.device)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(accelerator.device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=None,
                )
                
                # Compute losses
                text_loss, image_loss = compute_task_specific_losses(outputs, labels, task_masks)
                
                # Weighted loss
                loss = (config.training.text_loss_weight * text_loss + 
                       config.training.image_loss_weight * image_loss)
                
                # Track losses
                if text_loss.item() > 0:
                    text_losses.append(text_loss.item())
                if image_loss.item() > 0:
                    image_losses.append(image_loss.item())
                
                # Backward pass
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=config.training.max_grad_norm
                    )
                    if grad_norm is not None:
                        grad_norms.append(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if global_step % config.training.log_every == 0:
                        log_metrics(
                            writer, global_step, text_losses, image_losses, 
                            grad_norms, lr_scheduler, accelerator
                        )
                        text_losses = []
                        image_losses = []
                        grad_norms = []
                    
                    # Save checkpoint
                    if global_step % config.training.save_every == 0:
                        save_checkpoint(
                            output_dir, global_step, epoch,
                            model, optimizer, lr_scheduler,
                            accelerator,
                        )
                    
                    # Validation
                    if global_step % config.training.val_every == 0:
                        val_loss = validate(
                            model, val_dataloader, processor, config,
                            accelerator, writer, global_step
                        )
                        
                        if accelerator.is_main_process and val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                output_dir, global_step, epoch,
                                model, optimizer, lr_scheduler,
                                accelerator, is_best=True,
                            )
                        
                        model.train()
        
        progress_bar.close()
        
        # End of epoch
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            writer.flush()
        
        torch.cuda.empty_cache()
    
    # Cleanup
    if accelerator.is_main_process:
        writer.close()
    
    accelerator.print("Training complete!")


def validate(
    model: nn.Module,
    val_dataloader: Any,
    processor: Any,
    config: OmniSVGConfig,
    accelerator: Accelerator,
    writer: SummaryWriter,
    step: int,
) -> float:
    """Run validation."""
    model.eval()
    accelerator.print(f"Validation at step {step}...")
    
    total_losses = []
    text_losses = []
    image_losses = []
    
    with torch.no_grad():
        for batch_messages, pix_seq_lists, batch_task_types in tqdm(
            val_dataloader, 
            disable=not accelerator.is_local_main_process
        ):
            input_ids, attention_mask, pixel_values, image_grid_thw, labels, task_masks = \
                process_mixed_batch(
                    batch_messages, pix_seq_lists, batch_task_types, processor, config
                )
            
            # Move to device
            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)
            labels = labels.to(accelerator.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(accelerator.device)
            if image_grid_thw is not None:
                image_grid_thw = image_grid_thw.to(accelerator.device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=None,
            )
            
            text_loss, image_loss = compute_task_specific_losses(outputs, labels, task_masks)
            total_loss = text_loss + image_loss
            
            # Gather across processes
            all_total = accelerator.gather_for_metrics(total_loss)
            all_text = accelerator.gather_for_metrics(text_loss)
            all_image = accelerator.gather_for_metrics(image_loss)
            
            total_losses.append(all_total.mean().item())
            if all_text.sum() > 0:
                text_losses.append(all_text.mean().item())
            if all_image.sum() > 0:
                image_losses.append(all_image.mean().item())
    
    # Compute averages
    avg_total = np.mean(total_losses) if total_losses else 0
    avg_text = np.mean(text_losses) if text_losses else 0
    avg_image = np.mean(image_losses) if image_losses else 0
    
    accelerator.print(f"Validation - Total: {avg_total:.4f}, Text: {avg_text:.4f}, Image: {avg_image:.4f}")
    
    if accelerator.is_main_process:
        writer.add_scalar("validation/total_loss", avg_total, step)
        writer.add_scalar("validation/text_loss", avg_text, step)
        writer.add_scalar("validation/image_loss", avg_image, step)
    
    return avg_total


def log_metrics(
    writer: SummaryWriter,
    step: int,
    text_losses: List[float],
    image_losses: List[float],
    grad_norms: List[float],
    lr_scheduler: Any,
    accelerator: Accelerator,
):
    """Log training metrics."""
    if not accelerator.is_main_process:
        return
    
    avg_text = np.mean(text_losses) if text_losses else 0
    avg_image = np.mean(image_losses) if image_losses else 0
    avg_total = np.mean(text_losses + image_losses) if (text_losses + image_losses) else 0
    avg_grad = np.mean(grad_norms) if grad_norms else 0
    
    writer.add_scalar("loss/total", avg_total, step)
    writer.add_scalar("loss/text_task", avg_text, step)
    writer.add_scalar("loss/image_task", avg_image, step)
    writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], step)
    writer.add_scalar("grad_norm", avg_grad, step)


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    model: nn.Module,
    optimizer: Any,
    lr_scheduler: Any,
    accelerator: Accelerator,
    is_best: bool = False,
):
    """Save training checkpoint."""
    accelerator.wait_for_everyone()
    
    if not accelerator.is_main_process:
        return
    
    accelerator.print(f"Saving checkpoint at step {step}")
    
    unwrapped_model = accelerator.unwrap_model(model)
    
    if is_best:
        ckpt_path = output_dir / "best_model"
    else:
        ckpt_path = output_dir / f"step_{step}"
    
    ckpt_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(unwrapped_model.state_dict(), ckpt_path / "pytorch_model.bin")
    
    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")
    torch.save(lr_scheduler.state_dict(), ckpt_path / "scheduler.pt")
    
    # Save training state
    torch.save({
        'step': step,
        'epoch': epoch,
    }, ckpt_path / "training_state.pt")
    
    # Save info
    with open(ckpt_path / "info.txt", 'w') as f:
        f.write(f"step: {step}\n")
        f.write(f"epoch: {epoch}\n")
        f.write(f"is_best: {is_best}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def print_model_info():
    """Print available model configurations."""
    print("\n" + "=" * 60)
    print("Available OmniSVG Model Configurations")
    print("=" * 60)
    
    for size, info in MODEL_DEFAULTS.items():
        print(f"\n{size} Model:")
        print(f"  Base Model:  {info['base_model']}")
        print(f"  Checkpoint:  {info['checkpoint']}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="OmniSVG Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 4B model with local data
  accelerate launch train.py --model_size 4B --data_dir ./data
  
  # Train 8B model with Flash Attention
  accelerate launch train.py --model_size 8B --use_flash_attn --data_dir ./data
  
  # Train with HuggingFace data (auto-download)
  accelerate launch train.py --model_size 4B --use_hf_data --datasets illustration icon
  
  # Resume from OmniSVG checkpoint (auto-download based on model size)
  accelerate launch train.py --model_size 4B --resume_from_checkpoint auto
  
  # Resume from specific HuggingFace checkpoint
  accelerate launch train.py --model_size 4B --resume_from_checkpoint OmniSVG/OmniSVG1.1_4B
"""
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model_size", type=str, default="4B",
                            choices=["4B", "8B"],
                            help="Model size: 4B (Qwen2.5-VL-3B based) or 8B (Qwen2.5-VL-7B based)")
    model_group.add_argument("--use_flash_attn", action="store_true",
                            help="Enable Flash Attention 2 for faster training")
    model_group.add_argument("--no_flash_attn", action="store_true",
                            help="Disable Flash Attention 2")
    
    # Data configuration
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--data_dir", type=str, default="./data",
                           help="Data directory containing train_meta.csv, val_meta.csv, svg/, png/")
    data_group.add_argument("--use_hf_data", action="store_true",
                           help="Download and use HuggingFace datasets")
    data_group.add_argument("--datasets", type=str, nargs="+",
                           default=["illustration", "icon"],
                           choices=["illustration", "icon"],
                           help="HuggingFace datasets to use (when --use_hf_data)")
    
    # Training configuration
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--config_dir", type=str, default="./configs",
                            help="Directory containing config files")
    train_group.add_argument("--output_dir", type=str, default="./output",
                            help="Output directory for checkpoints and logs")
    train_group.add_argument("--project_name", type=str, default=None,
                            help="Project name (default: omnisvg_{model_size})")
    train_group.add_argument("--batch_size", type=int, default=4,
                            help="Batch size per device")
    train_group.add_argument("--max_seq_length", type=int, default=2048,
                            help="Maximum SVG sequence length")
    train_group.add_argument("--resume_from_checkpoint", type=str, default=None,
                            help="Path to checkpoint, HuggingFace repo ID (e.g., OmniSVG/OmniSVG1.1_4B), or 'auto'")
    
    # Utility options
    parser.add_argument("--list_datasets", action="store_true",
                       help="List available HuggingFace datasets and exit")
    parser.add_argument("--list_models", action="store_true",
                       help="List available model configurations and exit")
    
    args = parser.parse_args()
    
    # List datasets and exit
    if args.list_datasets:
        list_available_datasets()
        return
    
    # List models and exit
    if args.list_models:
        print_model_info()
        return
    
    # Set default project name
    if args.project_name is None:
        args.project_name = f"omnisvg_{args.model_size.lower()}"
    
    # Load configuration
    config = OmniSVGConfig(
        config_dir=args.config_dir,
        model_size=args.model_size,
    )
    
    # Override config with command line args
    if args.max_seq_length:
        config.training.max_seq_length = args.max_seq_length
    if args.data_dir:
        config.training.data_dir = args.data_dir
        config.data.data_dir = args.data_dir
    
    # Handle flash attention flag
    if args.no_flash_attn:
        config.training.use_flash_attn = False
    elif args.use_flash_attn:
        config.training.use_flash_attn = True
    
    print(f"\n{'='*60}")
    print(f"OmniSVG Training Configuration")
    print(f"{'='*60}")
    print(f"Model Size:        {config.model_size}")
    print(f"Base Model:        {MODEL_DEFAULTS[config.model_size]['base_model']}")
    print(f"Default Checkpoint:{MODEL_DEFAULTS[config.model_size]['checkpoint']}")
    print(f"Flash Attention:   {config.training.use_flash_attn}")
    print(f"Data Directory:    {config.training.data_dir}")
    print(f"Max Seq Length:    {config.training.max_seq_length}")
    print(f"Batch Size:        {args.batch_size}")
    print(f"Output Directory:  {args.output_dir}/{args.project_name}")
    if args.resume_from_checkpoint:
        print(f"Resume From:       {args.resume_from_checkpoint}")
    print(f"{'='*60}\n")
    
    # Run training
    train(args, config)


if __name__ == "__main__":
    main()
