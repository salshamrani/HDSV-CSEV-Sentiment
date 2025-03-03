# fine_tuning.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_from_disk
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from accelerate import Accelerator
import numpy as np
import json
import os
import random

os.environ["TOKENIZERS_PARALLELISM"] = "true"
accelerator = Accelerator()

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    f1 = f1_score(labels, predictions, average='binary')
    accuracy = accuracy_score(labels, predictions)
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tnr": tnr
    }

def train_and_evaluate(model_name, train_dataset, val_dataset, test_dataset, num_epochs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=num_epochs,
        fp16=True,
        dataloader_num_workers=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.train_dataloader = accelerator.prepare(
        trainer.model, trainer.optimizer, trainer.lr_scheduler, trainer.get_train_dataloader()
    )
    
    trainer.train()
    
    metrics = trainer.evaluate(test_dataset)
    return model, tokenizer, metrics

def main():
    dataset = load_from_disk("Datasets/yelp_CSEV")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    
    model_name = "distilbert-base-uncased"
    num_epochs = 3
    
    model, tokenizer, metrics = train_and_evaluate(model_name, train_dataset, val_dataset, test_dataset, num_epochs)
    
    if accelerator.is_main_process:
        output_file = f"evaluation_results/{model_name.replace('/', '-')}_results.json"
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Evaluation results saved to {output_file}")
    
    accelerator.wait_for_everyone()
    print("Fine-tuning completed!")

if __name__ == "__main__":
    main()
