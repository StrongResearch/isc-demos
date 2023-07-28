# Trains BERT on SQAD dataset

From https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering

Run on ISC with

```bash
isc-train -c config.isc run_qa_no_trainer.py   --model_name_or_path bert-base-uncased   --dataset_name squad   --max_seq_length 384   --doc_stride 128   --output_dir ~/tmp/debug_squad
```