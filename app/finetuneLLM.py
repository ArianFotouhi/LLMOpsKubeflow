import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline, logging
from peft import LoraConfig
from trl import SFTTrainer
import torch
import gc

class LLM():
    def __init__(self, dataset_file):
        
        gc.collect()
        print("\n--------------------------------------------------\n")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print("\n--------------------------------------------------\n")
        
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

        self.base_model = "NousResearch/Meta-Llama-3-8B"
        dataset = dataset_file
        self.new_model = "llama-3-8b-chat-finetuned"
        
        # Check if the dataset file exists
        if not os.path.isfile(dataset):
            raise FileNotFoundError(f"The dataset file {dataset} was not found in the current directory.")

        # Load dataset
        self.dataset = load_dataset("parquet", data_files=dataset, split="train")

        # 4-bit Quantization Configuration
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, quantization_config=quant_config, device_map={"": 0})
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def setUpTrainer(self):

        # Set PEFT Parameters
        self.peft_params = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, bias="none", task_type="CAUSAL_LM")


        self.training_params = TrainingArguments(output_dir="./results", num_train_epochs=1, per_device_train_batch_size=4, gradient_accumulation_steps=1, optim="paged_adamw_32bit", save_steps=25, logging_steps=25, learning_rate=2e-4, weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1, warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant", report_to="tensorboard")

        # Initialize the trainer
        self.trainer = SFTTrainer(model=self.model, train_dataset=self.dataset, peft_config=self.peft_params, dataset_text_field="text", max_seq_length=None, tokenizer=self.tokenizer, args=self.training_params, packing=False)

        #Force clean the pytorch cache
        gc.collect()

        torch.cuda.empty_cache()

    def startFineTuner(self):
        # Train the model
        self.trainer.train()

        # Save the model and tokenizer
        self.trainer.model.save_pretrained(self.new_model)
        self.trainer.tokenizer.save_pretrained(self.new_model)
    
    def Evaluate(self, input_prompt):
        logging.set_verbosity(logging.CRITICAL)
        prompt = input_prompt
        pipe = pipeline(task="text-generation", model=self.model, tokenizer=self.tokenizer, max_length=200)
        result = pipe(f"[INST] {prompt} [/INST]")
        print(result[0]['generated_text'])