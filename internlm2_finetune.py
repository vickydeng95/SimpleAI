import argparse
import os
import time
import torch
import transformers
import deepspeed
import deepspeed.comm as dist
from deepspeed.runtime.utils import memory_status
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TextGenerationPipeline
from prompt_template import PROMPT_TEMPLATE
from utils import template_map_fn_factory
from transformers import TrainingArguments, Trainer
from peft import LoraConfig
import numpy as np
import evaluate
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



def train(model_dir, batch_size, training_steps, stage, output_dir, *args):
    transformers.logging.set_verbosity_warning()
    rank = int(os.getenv("LOCAL_RANK", 3))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(rank)
    device = torch.device("cuda", rank)

    train_data_dir = 'MedQA2019-structured-train.jsonl' 
    data_cache_dir = 'model_cache/'
    
    model_quant_dir = model_dir #model_dir.split('/')[-1].split('-')[0]+ "-Quan/"
    prompt_template = PROMPT_TEMPLATE.internlm2_chat
    

    if not os.path.exists(model_quant_dir):
        ### Quantization -- TODO in InternLM2
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Load quantize config, model and tokenizer https://huggingface.co/docs/transformers/main/en/peft
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        """Train a PEFT adapter"""
        model.add_adapter(peft_config)
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
        
        train_dataset = load_dataset(path='json',
                                     data_files={'train': train_data_dir},
                                     cache_dir=data_cache_dir
                                    )
        template_map_fn = template_map_fn_factory(template=prompt_template)
        train_dataset = train_dataset.map(template_map_fn, num_proc=32)
        def process(x):
            input_ids = torch.tensor(tokenizer(x['conversation'][0]['input'], max_length=512, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True)['input_ids'])
            labels = torch.tensor(tokenizer(x['conversation'][0]['output'], max_length=512, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True)['input_ids'])
            x["input_ids"] = torch.clamp(torch.squeeze(input_ids, -2), int(input_ids.min()), model.model.tok_embeddings.num_embeddings - 1)
            x["labels"] = torch.clamp(torch.squeeze(labels, -2), int(labels.min()), model.model.tok_embeddings.num_embeddings - 1)
            return x
        tokenized_datasets = train_dataset.map(process, batch_size=True)
        
        small_train_dataset = tokenized_datasets["train"].select(range(batch_size * training_steps))   #.shuffle(seed=42).select(range(1000))
        small_eval_dataset = tokenized_datasets["train"].select(range(batch_size * training_steps))   #.shuffle(seed=42).select(range(1000))
        
        deepspeed_config = {
            "zero_optimization": {
                "stage": 1
            },
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 'auto',  #2
            "optimizer": {"type": "Adam",
                          "params": {"lr": 5e-5},
                         },
            "fp16": {
              "enabled": True
            }
        }
        from accelerate.utils import DistributedType
        # Define the training arguments
        training_args = TrainingArguments(
            output_dir='test_trainer',
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10,
            save_total_limit=2,
            fp16=True, 
            deepspeed=deepspeed_config,  # Path to DeepSpeed config file        
            gradient_checkpointing=True, 
        )
        # set distributed_state manually
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
        
        # eval function
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            """add this"""
            preds = preds.argmax(-1) #np.argmax(preds, axis=-1) #数字
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)#文字
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #数字
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) #文字
            score_dict = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": []
            }
            for pred, label in zip(decoded_preds, decoded_labels):
                hypothesis = tokenizer.tokenize(pred)  
                reference = tokenizer.tokenize(label) 
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference)) #文字 #文字
                result = scores[0]

                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            for k, v in score_dict.items():
                score_dict[k] = float(np.mean(v))
            return score_dict

        trainer = Trainer(
            model=model.to('cuda:0'),
            args=training_args,
            data_collator=data_collator,
            train_dataset= small_train_dataset, 
            eval_dataset= small_eval_dataset, 
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        trainer.train()
        ### save quantized model
        model_quant_dir = model_dir.split('/')[-1].split('-')[0]+ "-Quan/"
        os.mkdir(model_quant_dir)
        model.save_pretrained(model_quant_dir)
        #tokenizer.save_pretrained(model_quant_dir)
    else:
        ### Finetuning
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
        # model.load_adapter(model_quant_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)#.to(torch.bfloat16)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) 
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        train_dataset = load_dataset(path='json',
                                     data_files={'train': train_data_dir},
                                     cache_dir=data_cache_dir
                                    )
        prompt_template = PROMPT_TEMPLATE.internlm2_chat
        template_map_fn = template_map_fn_factory(template=prompt_template)
        train_dataset = train_dataset.map(template_map_fn, num_proc=32)
        def process(x):
            input_ids = torch.tensor(tokenizer(x['conversation'][0]['input'], max_length=512, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True)['input_ids'])
            labels = torch.tensor(tokenizer(x['conversation'][0]['output'], max_length=512, padding='max_length', truncation=True, return_tensors='pt', return_special_tokens_mask=True)['input_ids'])
            x["input_ids"] = torch.clamp(torch.squeeze(input_ids, -2), int(input_ids.min()), model.model.tok_embeddings.num_embeddings - 1)
            x["labels"] = torch.clamp(torch.squeeze(labels, -2), int(labels.min()), model.model.tok_embeddings.num_embeddings - 1)
            return x
        tokenized_datasets = train_dataset.map(process, batch_size=True)
        
        small_train_dataset = tokenized_datasets["train"].select(range(batch_size * training_steps))   #.shuffle(seed=42).select(range(1000))
        small_eval_dataset = tokenized_datasets["train"].select(range(batch_size * training_steps))   #.shuffle(seed=42).select(range(1000))
        
        deepspeed_config = {
            "zero_optimization": {
                "stage": 1
            },
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 'auto',  #2
            "optimizer": {"type": "Adam",
                          "params": {"lr": 5e-5},
                         },
            "fp16": {
              "enabled": True
            }
        }
        from accelerate.utils import DistributedType
        # Define the training arguments
        training_args = TrainingArguments(
            #output_dir='results',
            overwrite_output_dir=True,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            save_steps=10,
            save_total_limit=2, #最多保存save_total_limit=2 个模型
            fp16=True, 
            deepspeed=deepspeed_config,  # Path to DeepSpeed config file        
            gradient_checkpointing=True, 
        )
        # set distributed_state manually
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
        
        # eval function
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            if isinstance(preds, tuple):
                preds = preds[0]
            preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
            """add this"""
            preds = preds.argmax(-1) #np.argmax(preds, axis=-1) #数字
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)#文字
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id) #数字
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True) #文字

            score_dict = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": []
            }
            for pred, label in zip(decoded_preds, decoded_labels):
                hypothesis = tokenizer.tokenize(pred)  
                reference = tokenizer.tokenize(label) 
                rouge = Rouge()
                scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference)) #文字 #文字
                result = scores[0]

                for k, v in result.items():
                    score_dict[k].append(round(v["f"] * 100, 4))
                bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
                score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            for k, v in score_dict.items():
                score_dict[k] = float(np.mean(v))
            return score_dict
        
        trainer = Trainer(
            model=model.to('cuda:0'),
            args=training_args,
            data_collator=data_collator,
            train_dataset= small_train_dataset,#.to(device), #train_datasets_sampled11, #small_train_dataset,
            eval_dataset= small_eval_dataset,#.to(device),
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
        
        # reset communcations logs so that only communications
        # made during the training loop will be reflected in 
        # total_comms_latency below
        dist.comms_logger.comms_dict = {}

        training_start_time = time.time()
        
        trainer.train()
        eval_results = trainer.evaluate()
        print("Evaluation Results:", eval_results)
        
        # Predict on the evaluation dataset
        predictions = trainer.predict(test_dataset=small_eval_dataset)
        #print("Predictions:", predictions.metrics)
        
        training_end_time = time.time()
        if rank == 0:
            print(f"\nTotal Training Time: {training_end_time - training_start_time:.2f} seconds")

        # print average comms latency across all comms
        total_comms_latency = 0
        for op_name in dist.comms_logger.comms_dict.keys():
            if op_name == "log_summary_barrier":
                continue
            for _, vals in sorted(dist.comms_logger.comms_dict[op_name].items()):
                total_comms_latency += sum(vals[1])

        dist.log_summary()

        if rank == 0:
            print(f"\nTotal Communication Latency: {total_comms_latency:.2f} seconds")
            
        
        # save the merged model and the tokenizer
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # To reload Fine-tuned model, run
        # ft_model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """@@@ make pretrained model smaller"""
    parser.add_argument("--model-dir", type=str, default='/mnt/sgnfsdata/tolo-03-97/pretrained_models/internlm2-chat-1_8b') 
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--training-steps", type=int, default=1) #100
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default='internlm2_quantized_saved_finetune_model/')
    #parser.add_argument("--tmplate", type=str, default='internlm2_chat') #PROMPT_TEMPLATE.internlm2_chat
    args, extra_args = parser.parse_known_args()
    
    train(args.model_dir, args.batch_size, args.training_steps, args.stage, args.output_dir, *extra_args)
