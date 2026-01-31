import torch
import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

def train_model(model, optimizer, device, criterion, bar, train_step_losses=None, global_steps=None, start_step=0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    step = start_step

    for batch in bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        timestamps = batch["timestamps"].to(device)
        speakers = batch["speakers"].to(device)
        utterance_mask = batch["utterance_mask"].to(device)

        logits = model(input_ids, attention_mask, timestamps, speakers, labels, utterance_mask)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )

        # print("logits finite:", torch.isfinite(logits).all().item())
        # print("labels finite:", torch.isfinite(labels).all().item())
        # print("timestamps finite:", torch.isfinite(timestamps).all().item())
        # print("max logits:", logits.abs().max().item())
        # print("yuh")

        loss.backward()
        optimizer.step()

        if train_step_losses is not None:
            train_step_losses.append(loss.item())
        if global_steps is not None:
            global_steps.append(step)
        step += 1

        total_loss += loss.item()

        num_batches += 1

        bar.set_postfix(step = "Training", loss=loss.item(), average=total_loss / num_batches)

    avg_loss = total_loss / num_batches

    return avg_loss, step

def validate_model(model, device, criterion, bar, val_step_losses=None, val_steps=None, start_step=0):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_preds = []
    all_labels = []
    step = start_step

    with torch.no_grad():
        for batch in bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            timestamps = batch["timestamps"].to(device)
            speakers = batch["speakers"].to(device)
            labels = batch["labels"].to(device)   # (B, T)
            utterance_mask = batch["utterance_mask"].to(device)

            logits = model(
                input_ids,
                attention_mask,
                timestamps,
                speakers,
                labels,
                utterance_mask
            )

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )


            total_loss += loss.item()
            num_batches += 1

            preds = logits.argmax(dim=-1)
            mask = labels != -1

            all_preds.append(preds[mask].cpu())
            all_labels.append(labels[mask].cpu())

            if val_step_losses is not None:
                val_step_losses.append(loss.item())
            if val_steps is not None:
                val_steps.append(step)
            step += 1

            bar.set_postfix(step="Validating", loss=loss.item())



    avg_val_loss = total_loss / num_batches

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    macro_f1 = f1_score(
        all_labels.numpy(),
        all_preds.numpy(),
        average="macro"
    )

    report = classification_report(
        all_labels.numpy(),
        all_preds.numpy(),
        digits=4,
        output_dict=True,
        zero_division=0.0,
    )

    return avg_val_loss, macro_f1, step, report

def test_model(model, dataloader, device, emotion_labels=None, save_results=False, save_dir=None):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            timestamps = batch["timestamps"].to(device)
            speakers = batch["speakers"].to(device)
            labels = batch["labels"].to(device)
            utterance_mask = batch["utterance_mask"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                timestamps=timestamps,
                speakers=speakers,
                labels=labels,
                utterance_mask=utterance_mask
            )

            preds = logits.argmax(dim=-1)

            mask = labels != -1  # ignore padded utterances

            all_preds.append(preds[mask].cpu())
            all_labels.append(labels[mask].cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    macro_f1 = f1_score(
        all_labels.numpy(),
        all_preds.numpy(),
        average="macro"
    )

    print(f"Test Macro F1: {macro_f1:.4f}")

    if emotion_labels is not None:
        print("\nPer-emotion results:")
        report_to_print = classification_report(
            all_labels.numpy(),
            all_preds.numpy(),
            target_names=emotion_labels,
            digits=3,
            zero_division = 0.0
        )
        print(report_to_print)

        if save_results:
            report_to_save = classification_report(
                all_labels.numpy(),
                all_preds.numpy(),
                target_names=emotion_labels,
                digits=3,
                output_dict=True,
                zero_division = 0.0
            )
            df = pd.DataFrame(report_to_save).transpose()
            df.to_csv(save_dir)


    return macro_f1, all_preds, all_labels

def collate_conversations(batch, tokenizer, max_len=128):
    B = len(batch)
    T_max = max(len(item["texts"]) for item in batch)

    padded_texts = []
    padded_labels = []
    padded_timestamps = []
    padded_speakers = []
    utterance_mask = []
    for item in batch:
        texts = item["texts"]
        labels = item["labels"]
        timestamps = item["timestamps"]
        speakers = item["speakers"]

        pad_len = T_max - len(texts)

        padded_texts.extend(texts + [""] * pad_len)

        utterance_mask.append(
            torch.cat([torch.ones(len(texts)), torch.zeros(pad_len)])
        )

        padded_labels.append(
            torch.cat([labels, torch.full((pad_len,), -1, dtype=torch.long)])
        )

        # changed to delta
        # padded_timestamps.append(
        #     torch.cat([timestamps, torch.zeros(pad_len, dtype=torch.long)])
        # )

        # timestamps are in ms
        timestamps = timestamps.float()

        # compute deltas
        delta = torch.zeros_like(timestamps)
        delta[1:] = timestamps[1:] - timestamps[:-1]

        # close extreme gaps
        delta = torch.clamp(delta, max=5 * 60 * 1000, min=0)  # 5 minutes

        # scale
        delta = delta / 1000.0

        # log scaling
        delta = torch.log1p(delta)

        padded_timestamps.append(
            torch.cat([delta, torch.zeros(pad_len)])
        )

        padded_speakers.append(
            torch.cat([speakers, torch.zeros(pad_len, dtype=torch.long)])
        )


    encoding = tokenizer(
        padded_texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": torch.stack(padded_labels),
        "timestamps": torch.stack(padded_timestamps),
        "speakers": torch.stack(padded_speakers),
        "utterance_mask": torch.stack(utterance_mask)
    }

def compute_class_weights(dataloader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float)

    for batch in dataloader:
        labels = batch["labels"].view(-1)
        # i want to ignore padding
        labels = labels[labels != -1]

        counts += torch.bincount(labels, minlength=num_classes).float()

    counts[counts == 0] = 1.0

    # softer than 1/count
    weights = torch.sqrt(counts) # 1.0 / counts
    weights = weights / weights.sum() * num_classes

    return weights.to(device)

def save_table_as_csv(table, path):
    df = pd.DataFrame(table)
    df.to_csv(path, index=False)

def create_new_run_dir(base_dir="runs", new_dir=True):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_pattern = re.compile(r"^run(\d+)$")
    runs = []

    for p in base_dir.iterdir():
        if p.is_dir():
            match = run_pattern.match(p.name)
            if match:
                runs.append((int(match.group(1)), p))

    if not runs:
        run_dir = base_dir / "run1"
        run_dir.mkdir()
        return run_dir

    runs.sort(key=lambda x: x[0])
    last_num, last_dir = runs[-1]

    if not new_dir:
        return last_dir

    if not any(last_dir.iterdir()):
        return last_dir

    new_run = base_dir / f"run{last_num + 1}"
    new_run.mkdir()
    return new_run