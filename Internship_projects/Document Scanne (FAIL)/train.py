# ============================================================================
# COMPLETE DONUT OCR FINE-TUNING SCRIPT - OPTIMIZED FOR CPU
# ============================================================================

from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import torch
import pandas as pd
import numpy as np
import io
import json
import re
import time
import os

print("="*70)
print("DONUT OCR FINE-TUNING - OPTIMIZED VERSION")
print("="*70)

# ============================================================================
# STEP 0: LOAD PARQUET FILE
# ============================================================================

print("\n📁 Loading parquet file...")

# Load your parquet file
# Replace with your actual parquet file path
parquet_file = "0000.parquet"  # ← CHANGE THIS

# Option 1: If you know the exact path
try:
    df = pd.read_parquet(parquet_file)
    print(f"✓ Loaded {len(df)} samples from {parquet_file}")
except FileNotFoundError:
    print(f"❌ File not found: {parquet_file}")
    print("\nPlease provide the correct path to your .parquet file")
    print("\nCommon locations:")
    print("  • Current directory: './data.parquet'")
    print("  • Downloads: '~/Downloads/data.parquet'")
    print("  • HF cache: '~/.cache/huggingface/datasets/...'")
    
    # Try to find parquet files
    import glob
    found = glob.glob("**/*.parquet", recursive=True)
    if found:
        print(f"\nFound these .parquet files:")
        for f in found[:5]:
            print(f"  • {f}")
    raise

# Verify structure
print("\n📋 Dataset info:")
print(f"  Columns: {list(df.columns)}")
print(f"  Shape: {df.shape}")
print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ============================================================================
# STEP 1: MODEL LOADING
# ============================================================================

device = "cpu"
print(f"\n📱 Using device: {device}")
print("Note: CPU training is slower but stable\n")

print("Loading Donut model and processor...")
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

print(f"✓ Model loaded: {model.__class__.__name__}")
print(f"  Encoder: {model.encoder.__class__.__name__}")
print(f"  Decoder: {model.decoder.__class__.__name__}")

# Get expected input size
image_processor = processor.image_processor
image_size = image_processor.size
if isinstance(image_size, dict):
    expected_height = image_size.get('height', 1280)
    expected_width = image_size.get('width', 960)
else:
    expected_height = expected_width = image_size
print(f"  Expected input size: {expected_height}x{expected_width}")

# ============================================================================
# STEP 2: FREEZE ENCODER (Critical optimization!)
# ============================================================================

print("\n🔒 Freezing encoder...")
for param in model.encoder.parameters():
    param.requires_grad = False

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = total_params - trainable_params

print(f"✓ Encoder frozen!")
print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print(f"  Frozen: {(frozen_params/total_params)*100:.1f}%")
print(f"  Speed improvement: ~{total_params/trainable_params:.1f}x faster")

# ============================================================================
# STEP 3: ANALYZE DATA FOR OPTIMAL MAX_LENGTH
# ============================================================================

print("\n📊 Analyzing dataset to optimize max_length...")

# Sample data for analysis
df_analysis = df.sample(n=min(1000, len(df)), random_state=42)
text_lengths = df_analysis['ground_truth'].apply(
    lambda x: len(json.loads(x)['gt_parse']['text_sequence'])
)

max_95th = int(text_lengths.quantile(0.95))
optimal_max_length = min(256, max_95th + 50)

print(f"  Mean length: {text_lengths.mean():.1f}")
print(f"  Median: {text_lengths.median():.1f}")
print(f"  95th percentile: {max_95th}")
print(f"  Max: {text_lengths.max()}")
print(f"✓ Using max_length: {optimal_max_length}")

# ============================================================================
# STEP 4: CONFIGURE MODEL
# ============================================================================

print("\n⚙️  Configuring model...")

# Special tokens
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]
model.config.eos_token_id = processor.tokenizer.eos_token_id

# Set max_length
model.config.max_length = optimal_max_length
model.decoder.config.max_length = optimal_max_length

# Complete generation config
model.generation_config.max_length = optimal_max_length
model.generation_config.num_beams = 1
model.generation_config.early_stopping = True
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 1.0
model.generation_config.repetition_penalty = 1.2
model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
model.generation_config.pad_token_id = model.config.pad_token_id
model.generation_config.eos_token_id = model.config.eos_token_id
model.generation_config.forced_bos_token_id = None

# Gradient checkpointing
model.config.use_cache = False
model.gradient_checkpointing_enable()

# Move to device
model.to(device)

print("✓ Model configured")
print(f"  Vocabulary size: {len(processor.tokenizer)}")
print(f"  PAD token ID: {processor.tokenizer.pad_token_id}")
print(f"  EOS token ID: {processor.tokenizer.eos_token_id}")
print(f"  Max length: {optimal_max_length}")

# ============================================================================
# STEP 5: DATASET CLASS
# ============================================================================

class DocumentOCRDataset(Dataset):
    def __init__(self, dataframe, processor, max_length=256, augment=False, 
                 cache_images=True, max_cache_size=200, task_name="document-ocr"):
        self.df = dataframe
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        self.max_cache_size = max_cache_size if cache_images else 0
        self.image_cache = {}
        self.task_name = task_name
        self.task_prompt = f"<s_{task_name}>"
        
        if augment:
            self.augmentation = transforms.Compose([
                transforms.RandomRotation(degrees=2),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            ])
        
        # Cache images
        if self.max_cache_size > 0:
            print(f"  Caching up to {self.max_cache_size} images...")
            for idx in range(min(len(self.df), self.max_cache_size)):
                self._load_image(idx)
            print(f"  ✓ Cached {len(self.image_cache)} images")
    
    def _load_image(self, idx):
        if idx in self.image_cache:
            return self.image_cache[idx]
        
        row = self.df.iloc[idx]
        img_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        if len(self.image_cache) < self.max_cache_size:
            self.image_cache[idx] = image
        
        return image
    
    def clean_text(self, text):
        return re.sub(r'\s+', ' ', text).strip()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            
            # Load image
            if 'image' not in row or 'bytes' not in row['image']:
                raise ValueError(f"Missing image data at index {idx}")
            
            image = self._load_image(idx)
            
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions at index {idx}")
            
            if self.augment:
                image = self.augmentation(image)
            
            # Get text
            if 'ground_truth' not in row:
                raise ValueError(f"Missing ground_truth at index {idx}")
                
            gt = json.loads(row['ground_truth'])
            text = gt['gt_parse']['text_sequence']
            
            if not text or not text.strip():
                text = " "
            
            text = self.clean_text(text)
            
            # Add task prompt
            text_with_task = f"{self.task_prompt} {text}"
            
            # Process
            pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
            
            labels = self.processor.tokenizer(
                text_with_task,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze()
            
            # Mask padding
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
            return {
                "pixel_values": pixel_values, 
                "labels": labels,
                "idx": idx,
                "text_length": (labels != -100).sum().item(),
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            raise

# ============================================================================
# STEP 6: COMPUTE METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute Character Error Rate"""
    predictions, labels = eval_pred
    
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
    
    decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Simple character accuracy
    total_chars = sum(len(label) for label in decoded_labels)
    correct_chars = sum(
        sum(1 for p, l in zip(pred, label) if p == l)
        for pred, label in zip(decoded_preds, decoded_labels)
    )
    cer_score = 1 - (correct_chars / max(total_chars, 1))
    
    return {
        "cer": cer_score,
        "avg_pred_length": np.mean([len(p) for p in decoded_preds]),
    }

# ============================================================================
# STEP 7: PREPARE DATA
# ============================================================================

print("\n📦 Preparing dataset...")

# Sample subset
n_samples = 150  # Adjust for your needs
df_subset = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)
print(f"  Using {len(df_subset)} samples")

# Stratification
df_subset['text_length'] = df_subset['ground_truth'].apply(
    lambda x: len(json.loads(x)['gt_parse']['text_sequence'])
)
df_subset['text_length_bin'] = pd.cut(
    df_subset['text_length'],
    bins=min(5, len(df_subset)//10),
    labels=False,
    duplicates='drop'
)

# Split
train_df, val_df = train_test_split(
    df_subset, 
    test_size=0.15, 
    random_state=42,
    stratify=df_subset['text_length_bin'] if len(df_subset) > 30 else None
)
print(f"  Train: {len(train_df)}, Val: {len(val_df)}")

# Create datasets
print("\n  Creating training dataset...")
train_dataset = DocumentOCRDataset(
    train_df.reset_index(drop=True), 
    processor, 
    max_length=optimal_max_length,
    augment=False,
    cache_images=True,
    max_cache_size=200,
    task_name="document-ocr"
)

print("\n  Creating validation dataset...")
val_dataset = DocumentOCRDataset(
    val_df.reset_index(drop=True), 
    processor,
    max_length=optimal_max_length,
    augment=False,
    cache_images=True,
    max_cache_size=50,
    task_name="document-ocr"
)

# Collate function
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# Test data loading
print("\n  Testing data loading...")
test_batch = train_dataset[0]
print(f"  ✓ Pixel values shape: {test_batch['pixel_values'].shape}")
print(f"  ✓ Labels shape: {test_batch['labels'].shape}")
print(f"  ✓ Text length: {test_batch['text_length']}")

# ============================================================================
# STEP 8: TRAINING SETUP
# ============================================================================

print("\n🎯 Setting up training...")

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.training_start_time = None
        self.epoch_start_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start_time = time.time()
        print("\n" + "="*70)
        print("TRAINING STARTED")
        print("="*70)
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)}")
        print(f"{'='*70}")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        print(f"\n✓ Epoch completed in {epoch_time/60:.2f} minutes")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if 'loss' in logs:
                print(f"  Step {state.global_step}: Loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                print(f"  Val Loss = {logs['eval_loss']:.4f}")
            if 'cer' in logs:
                print(f"  CER = {logs['cer']:.4f}")
                
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.training_start_time
        print("\n" + "="*70)
        print(f"TRAINING COMPLETED in {total_time/60:.2f} minutes")
        print("="*70)

# Training configuration
batch_size = 1
gradient_accum = 16
num_epochs = 2

print(f"  Batch size: {batch_size}")
print(f"  Gradient accumulation: {gradient_accum}")
print(f"  Effective batch size: {batch_size * gradient_accum}")
print(f"  Epochs: {num_epochs}")

training_args = Seq2SeqTrainingArguments(
    output_dir="./donut-ocr-decoder-only",
    save_total_limit=2,
    save_steps=50,
    save_strategy="steps",
    
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accum,
    
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_steps=30,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    
    eval_strategy="steps",
    eval_steps=50,
    eval_accumulation_steps=1,
    
    predict_with_generate=True,
    generation_max_length=optimal_max_length,
    generation_num_beams=1,
    
    logging_dir="./logs",
    logging_steps=10,
    logging_first_step=True,
    logging_strategy="steps",
    
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    fp16=False,
    
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    gradient_checkpointing=True,
    optim="adamw_torch",
    
    report_to="none",
    push_to_hub=False,
    seed=42,
    data_seed=42,
)

# Estimate time
steps_per_epoch = len(train_dataset) // (batch_size * gradient_accum)
total_steps = steps_per_epoch * num_epochs

print(f"\n  Estimated steps: {total_steps}")
print(f"  Estimated time: {total_steps * 2 / 60:.1f}-{total_steps * 4 / 60:.1f} minutes")

# Create trainer
callbacks = [
    ProgressCallback(),
    EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)
]

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

# Check for checkpoints
resume_from = None
if os.path.exists(training_args.output_dir):
    checkpoints = [d for d in os.listdir(training_args.output_dir) 
                   if d.startswith("checkpoint-")]
    if checkpoints:
        latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        resume_from = os.path.join(training_args.output_dir, latest)
        print(f"\n⚠️  Found checkpoint: {resume_from}")

# ============================================================================
# READY TO TRAIN
# ============================================================================

print("\n" + "="*70)
print("✅ SETUP COMPLETE - READY TO TRAIN")
print("="*70)

print(f"\n📊 Summary:")
print(f"  • Device: {device}")
print(f"  • Training samples: {len(train_dataset)}")
print(f"  • Validation samples: {len(val_dataset)}")
print(f"  • Trainable params: {trainable_params/1e6:.1f}M")
print(f"  • Max sequence length: {optimal_max_length}")
print(f"  • Effective batch size: {batch_size * gradient_accum}")
print(f"  • Total training steps: {total_steps}")

print(f"\n🚀 To start training:")
print(f"   >>> trainer.train()")

print(f"\n💾 To save model after training:")
print(f"   >>> trainer.save_model('./donut-ocr-final')")
print(f"   >>> processor.save_pretrained('./donut-ocr-final')")

print(f"\n📈 To evaluate:")
print(f"   >>> trainer.evaluate()")

if resume_from:
    print(f"\n⏯️  To resume from checkpoint:")
    print(f"   >>> trainer.train(resume_from_checkpoint='{resume_from}')")

print("\n" + "="*70)

trainer.train()

trainer.save_model('./donut-ocr-final')
processor.save_pretrained('./donut-ocr-final')
trainer.evaluate()