from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from src.utils import compute_metrics

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def finetune_model(tokenized_train, tokenized_test, learning_rate, batch_size, num_train_epochs):
    logging_steps = len(tokenized_train) // batch_size
    model_name = "distillbert-finetuned-question-classifier"

    training_args = TrainingArguments(
        output_dir=model_name,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        eval_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        log_level="error",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )

    trainer.train()

    # Here we are optimizing for accuracy, might implement f1 score after
    eval_results = trainer.evaluate()
    return eval_results["eval_accuracy"]




