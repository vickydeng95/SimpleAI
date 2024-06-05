# SimpleAI

The fine-tuning codes of **llama** and **internlm2** include the training part, and evaluation part.

- `llama_finetune.py`: for llama. 
In command line, please run 
```python
  deepspeed --include localhost:3 --master_port 3010 llama_finetune.py; 
```
