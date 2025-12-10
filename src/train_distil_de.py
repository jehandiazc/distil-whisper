import torch
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_ID = "distil-whisper/distil-medium.en"
SAMPLE_RATE = 16000

print("Loading model + processor...")
processor = WhisperProcessor.from_pretrained(MODEL_ID)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

print("Loading German dataset...")
dataset = load_dataset("mozilla-foundation/common_voice_17_0", "de")

# Keep training time small for demo
train_dataset = dataset["train"].select(range(3000))  # small subset
test_dataset  = dataset["test"].select(range(500))    # small subset

print("Resampling audio...")
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
test_dataset  = test_dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))


def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=SAMPLE_RATE
    ).input_features[0]

    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    return batch


print("Preprocessing examples...")
train_dataset = train_dataset.map(prepare, num_proc=4)
test_dataset = test_dataset.map(prepare, num_proc=4)

training_args = Seq2SeqTrainingArguments(
    output_dir="./distil-whisper-de",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=1e-5,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=25,
    predict_with_generate=True,
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

print("Training starting...")
trainer.train()

trainer.save_model("./distil-whisper-de-finetuned")
processor.save_pretrained("./distil-whisper-de-finetuned")
print("Done!")
