# Fine-tuning TinyLlama-1.1B-Chat with QLoRA on ArXiv QA Dataset
# This script demonstrates how to fine-tune TinyLlama-1.1B-Chat using QLoRA with 8-bit quantization on the ArXiv QA dataset

# Install required packages with specific versions for compatibility
# !pip install -q torch==2.1.0 transformers==4.36.2 datasets==2.14.6 accelerate==0.25.0
# !pip install -q peft==0.7.1 bitsandbytes==0.41.1
# !pip install -q nltk rouge-score

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import load_dataset
import random
import numpy as np
import os
import bitsandbytes as bnb
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json
from tqdm import tqdm

# Download NLTK data for BLEU score calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# Load and prepare the dataset
def load_and_prepare_dataset():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("taesiri/arxiv_qa")
    print(f"Original dataset size: {len(dataset['train'])}")
    
    # Take 5% of the training data
    train_size = len(dataset['train'])
    sample_size = max(int(train_size * 0.05), 100)  # Ensure at least 100 samples
    sampled_indices = random.sample(range(train_size), sample_size)
    
    # Create a smaller dataset
    full_dataset = dataset['train'].select(sampled_indices)
    print(f"Sampled dataset size: {len(full_dataset)}")
    
    # Split into train and validation (90% train, 10% validation)
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"Train size after split: {len(split_dataset['train'])}")
    print(f"Validation size after split: {len(split_dataset['test'])}")
    
    return split_dataset

# Format and tokenize the data
def format_and_tokenize(example, tokenizer):
    # Format the input as a conversation with system instruction
    prompt = f"""<|system|>
You are a helpful AI assistant specialized in answering questions about research papers. 
Provide detailed, accurate, and well-structured answers based on the given questions.
</s>
<|user|>
{example['question']}
</s>
<|assistant|>
{example['answer']}
</s>"""
    
    # Tokenize the text
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Convert to numpy arrays and ensure they're the right type
    input_ids = tokenized["input_ids"][0].numpy()
    attention_mask = tokenized["attention_mask"][0].numpy()
    labels = input_ids.copy()
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Initialize model and tokenizer with 8-bit quantization
def initialize_model_and_tokenizer():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,  # Enable 8-bit quantization
        low_cpu_mem_usage=True,
        use_cache=False  # Disable KV cache for gradient checkpointing
    )
    
    # Enable gradient checkpointing
    model.config.use_cache = False  # Ensure config is also updated
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()  # Ensure model is in training mode
    
    # Prepare model for mixed precision training
    model = prepare_model_for_kbit_training(model)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

# Setup LoRA configuration with 8-bit quantization
def setup_lora(model):
    # Define LoRA Config with more conservative settings
    lora_config = LoraConfig(
        r=8,  # Reduced rank for better stability
        lora_alpha=16,  # Reduced alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,  # Increased dropout
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model

# Function to generate answer
def generate_answer(question, model, tokenizer, max_length=512):
    prompt = f"""<|system|>
You are a helpful AI assistant specialized in answering questions about research papers. 
Provide detailed, accurate, and well-structured answers based on the given questions.
</s>
<|user|>
{question}
</s>
<|assistant|>"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response
    response = response.split("<|assistant|>")[-1].strip()
    return response

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    # Tokenize the reference and candidate
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Calculate BLEU score
    return sentence_bleu([reference_tokens], candidate_tokens)

# Function to calculate ROUGE scores
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

# Function to compare base model and fine-tuned model
def compare_models(base_model, base_tokenizer, fine_tuned_model, fine_tuned_tokenizer, test_samples, num_samples=10):
    print("\n" + "="*50)
    print("COMPARING BASE MODEL AND FINE-TUNED MODEL")
    print("="*50)
    
    # Select random samples for comparison
    if len(test_samples) > num_samples:
        comparison_samples = random.sample(range(len(test_samples)), num_samples)
    else:
        comparison_samples = range(len(test_samples))
    
    # Initialize metrics
    bleu_scores_base = []
    bleu_scores_fine_tuned = []
    rouge1_scores_base = []
    rouge1_scores_fine_tuned = []
    rouge2_scores_base = []
    rouge2_scores_fine_tuned = []
    rougeL_scores_base = []
    rougeL_scores_fine_tuned = []
    
    # Store results for detailed analysis
    comparison_results = []
    
    # Process each sample
    for idx in tqdm(comparison_samples, desc="Comparing models"):
        sample = test_samples[idx]
        
        # Extract question and reference answer
        question = sample['question']
        reference_answer = sample['answer']
        
        # Generate answers from both models
        base_answer = generate_answer(question, base_model, base_tokenizer)
        fine_tuned_answer = generate_answer(question, fine_tuned_model, fine_tuned_tokenizer)
        
        # Calculate metrics
        bleu_base = calculate_bleu(reference_answer, base_answer)
        bleu_fine_tuned = calculate_bleu(reference_answer, fine_tuned_answer)
        
        rouge_base = calculate_rouge(reference_answer, base_answer)
        rouge_fine_tuned = calculate_rouge(reference_answer, fine_tuned_answer)
        
        # Store metrics
        bleu_scores_base.append(bleu_base)
        bleu_scores_fine_tuned.append(bleu_fine_tuned)
        rouge1_scores_base.append(rouge_base['rouge1'].fmeasure)
        rouge1_scores_fine_tuned.append(rouge_fine_tuned['rouge1'].fmeasure)
        rouge2_scores_base.append(rouge_base['rouge2'].fmeasure)
        rouge2_scores_fine_tuned.append(rouge_fine_tuned['rouge2'].fmeasure)
        rougeL_scores_base.append(rouge_base['rougeL'].fmeasure)
        rougeL_scores_fine_tuned.append(rouge_fine_tuned['rougeL'].fmeasure)
        
        # Store detailed results
        comparison_results.append({
            "question": question,
            "reference_answer": reference_answer,
            "base_answer": base_answer,
            "fine_tuned_answer": fine_tuned_answer,
            "bleu_base": bleu_base,
            "bleu_fine_tuned": bleu_fine_tuned,
            "rouge1_base": rouge_base['rouge1'].fmeasure,
            "rouge1_fine_tuned": rouge_fine_tuned['rouge1'].fmeasure,
            "rouge2_base": rouge_base['rouge2'].fmeasure,
            "rouge2_fine_tuned": rouge_fine_tuned['rouge2'].fmeasure,
            "rougeL_base": rouge_base['rougeL'].fmeasure,
            "rougeL_fine_tuned": rouge_fine_tuned['rougeL'].fmeasure
        })
        
        # Print sample comparison
        print(f"\nSample {idx + 1}:")
        print(f"Question: {question}")
        print(f"\nReference Answer: {reference_answer}")
        print(f"\nBase Model Answer: {base_answer}")
        print(f"BLEU Score: {bleu_base:.4f}")
        print(f"ROUGE-1 Score: {rouge_base['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2 Score: {rouge_base['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L Score: {rouge_base['rougeL'].fmeasure:.4f}")
        
        print(f"\nFine-tuned Model Answer: {fine_tuned_answer}")
        print(f"BLEU Score: {bleu_fine_tuned:.4f}")
        print(f"ROUGE-1 Score: {rouge_fine_tuned['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2 Score: {rouge_fine_tuned['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L Score: {rouge_fine_tuned['rougeL'].fmeasure:.4f}")
        print("\n" + "-"*50)
    
    # Calculate average metrics
    avg_bleu_base = np.mean(bleu_scores_base)
    avg_bleu_fine_tuned = np.mean(bleu_scores_fine_tuned)
    avg_rouge1_base = np.mean(rouge1_scores_base)
    avg_rouge1_fine_tuned = np.mean(rouge1_scores_fine_tuned)
    avg_rouge2_base = np.mean(rouge2_scores_base)
    avg_rouge2_fine_tuned = np.mean(rouge2_scores_fine_tuned)
    avg_rougeL_base = np.mean(rougeL_scores_base)
    avg_rougeL_fine_tuned = np.mean(rougeL_scores_fine_tuned)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY OF COMPARISON")
    print("="*50)
    print(f"Average BLEU Score - Base Model: {avg_bleu_base:.4f}")
    print(f"Average BLEU Score - Fine-tuned Model: {avg_bleu_fine_tuned:.4f}")
    print(f"BLEU Score Improvement: {(avg_bleu_fine_tuned - avg_bleu_base) / avg_bleu_base * 100:.2f}%")
    
    print(f"\nAverage ROUGE-1 Score - Base Model: {avg_rouge1_base:.4f}")
    print(f"Average ROUGE-1 Score - Fine-tuned Model: {avg_rouge1_fine_tuned:.4f}")
    print(f"ROUGE-1 Score Improvement: {(avg_rouge1_fine_tuned - avg_rouge1_base) / avg_rouge1_base * 100:.2f}%")
    
    print(f"\nAverage ROUGE-2 Score - Base Model: {avg_rouge2_base:.4f}")
    print(f"Average ROUGE-2 Score - Fine-tuned Model: {avg_rouge2_fine_tuned:.4f}")
    print(f"ROUGE-2 Score Improvement: {(avg_rouge2_fine_tuned - avg_rouge2_base) / avg_rouge2_base * 100:.2f}%")
    
    print(f"\nAverage ROUGE-L Score - Base Model: {avg_rougeL_base:.4f}")
    print(f"Average ROUGE-L Score - Fine-tuned Model: {avg_rougeL_fine_tuned:.4f}")
    print(f"ROUGE-L Score Improvement: {(avg_rougeL_fine_tuned - avg_rougeL_base) / avg_rougeL_base * 100:.2f}%")
    
    # Save detailed results to JSON
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_bleu_base": avg_bleu_base,
                "avg_bleu_fine_tuned": avg_bleu_fine_tuned,
                "bleu_improvement_percent": (avg_bleu_fine_tuned - avg_bleu_base) / avg_bleu_base * 100,
                "avg_rouge1_base": avg_rouge1_base,
                "avg_rouge1_fine_tuned": avg_rouge1_fine_tuned,
                "rouge1_improvement_percent": (avg_rouge1_fine_tuned - avg_rouge1_base) / avg_rouge1_base * 100,
                "avg_rouge2_base": avg_rouge2_base,
                "avg_rouge2_fine_tuned": avg_rouge2_fine_tuned,
                "rouge2_improvement_percent": (avg_rouge2_fine_tuned - avg_rouge2_base) / avg_rouge2_base * 100,
                "avg_rougeL_base": avg_rougeL_base,
                "avg_rougeL_fine_tuned": avg_rougeL_fine_tuned,
                "rougeL_improvement_percent": (avg_rougeL_fine_tuned - avg_rougeL_base) / avg_rougeL_base * 100
            },
            "detailed_results": comparison_results
        }, f, ensure_ascii=False, indent=2)
    
    print("\nDetailed comparison results saved to 'model_comparison_results.json'")
    
    return {
        "bleu_base": avg_bleu_base,
        "bleu_fine_tuned": avg_bleu_fine_tuned,
        "rouge1_base": avg_rouge1_base,
        "rouge1_fine_tuned": avg_rouge1_fine_tuned,
        "rouge2_base": avg_rouge2_base,
        "rouge2_fine_tuned": avg_rouge2_fine_tuned,
        "rougeL_base": avg_rougeL_base,
        "rougeL_fine_tuned": avg_rougeL_fine_tuned
    }

# Main execution
if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you're using a GPU environment.")
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    # Initialize model and tokenizer first
    print("\nInitializing model and tokenizer with 8-bit quantization...")
    model, tokenizer = initialize_model_and_tokenizer()
    
    # Load and prepare dataset
    split_dataset = load_and_prepare_dataset()
    
    # Format and tokenize dataset
    print("Formatting and tokenizing dataset...")
    formatted_dataset = split_dataset.map(
        lambda x: format_and_tokenize(x, tokenizer),
        remove_columns=split_dataset["train"].column_names,
        desc="Formatting and tokenizing dataset"
    )
    
    print(f"Training samples: {len(formatted_dataset['train'])}")
    print(f"Validation samples: {len(formatted_dataset['test'])}")
    
    # Verify dataset is not empty
    if len(formatted_dataset['train']) == 0 or len(formatted_dataset['test']) == 0:
        raise ValueError("Dataset is empty after processing!")
    
    # Print a sample to verify format
    print("\nSample formatted data:")
    print(formatted_dataset['train'][0])
    
    # Setup LoRA
    print("Setting up LoRA...")
    model = setup_lora(model)
    
    # Training arguments with more conservative settings
    training_args = TrainingArguments(
        output_dir="./tinllama-arxiv-qa-8bit",
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced batch size for memory constraints
        gradient_accumulation_steps=8,  # Increased gradient accumulation
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-4,
        fp16=True,  # Enable fp16 training
        bf16=False,  # Disable bf16
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Use 8-bit AdamW optimizer
        max_grad_norm=0.3,  # Add gradient clipping
        ddp_find_unused_parameters=False,  # Disable unused parameter detection
        dataloader_pin_memory=False  # Disable pin memory for better compatibility
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],  # Using test split as validation
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print("Saving model...")
    trainer.save_model("./tinllama-arxiv-qa-8bit-final")
    
    # Evaluation Section
    print("\nStarting evaluation...")
    
    # Load the base model for comparison
    print("Loading base model for comparison...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device_map="auto",
        load_in_8bit=True,
        trust_remote_code=True
    )
    base_model.eval()
    
    # Load the best model for evaluation
    print("Loading best model for evaluation...")
    best_model = AutoModelForCausalLM.from_pretrained(
        "./tinllama-arxiv-qa-8bit-final",
        device_map="auto",
        load_in_8bit=True,  # Load in 8-bit for evaluation
        trust_remote_code=True
    )
    best_model.eval()  # Set to evaluation mode
    
    # Extract original questions and answers from the test dataset
    test_samples = []
    for idx in range(min(20, len(split_dataset["test"]))):
        test_samples.append({
            "question": split_dataset["test"][idx]["question"],
            "answer": split_dataset["test"][idx]["answer"]
        })
    
    # Compare base model and fine-tuned model
    comparison_metrics = compare_models(
        base_model, 
        tokenizer, 
        best_model, 
        tokenizer, 
        test_samples, 
        num_samples=10
    )
    
    # Calculate evaluation metrics
    print("\nCalculating evaluation metrics...")
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    print("\nEvaluation complete!")
