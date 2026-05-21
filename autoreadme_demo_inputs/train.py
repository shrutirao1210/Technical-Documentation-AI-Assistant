import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import argparse
from tqdm import tqdm

from config import (ROBERTA_MODEL_NAME, BART_MODEL_NAME, NUM_EMOTIONS,
                    LEARNING_RATE_PHASE1_BASE, LEARNING_RATE_PHASE1_HEAD,
                    LEARNING_RATE_PHASE2, EPOCHS_PHASE1, EPOCHS_PHASE2, DEVICE)
from dataset import get_tokenizers, get_phase1_dataloaders, get_phase2_dataloaders
from model import DecoupledEmpatheticModel


def train(resume_phase2_epoch=0):
    print(f"Using device: {DEVICE}")

    rob_tokenizer, bart_tokenizer = get_tokenizers()
    os.makedirs('checkpoints', exist_ok=True)

    print("--- Initializing Decoupled Model ---")
    model = DecoupledEmpatheticModel(ROBERTA_MODEL_NAME, BART_MODEL_NAME, NUM_EMOTIONS)
    model.to(DEVICE)

    # ── PHASE 1: Full RoBERTa fine-tune on GoEmotions ─────────────────────────
    if resume_phase2_epoch == 0:
        print("\n==================================")
        print(" PHASE 1: Full RoBERTa Fine-tuning")
        print("==================================")
        p1_train_loader, p1_val_loader = get_phase1_dataloaders(rob_tokenizer)

        for param in model.roberta.parameters():
            param.requires_grad = True

        p1_all_params = list(model.roberta.parameters()) + list(model.classifier.parameters())

        optimizer_p1 = AdamW([
            {'params': model.roberta.parameters(),    'lr': LEARNING_RATE_PHASE1_BASE},
            {'params': model.classifier.parameters(), 'lr': LEARNING_RATE_PHASE1_HEAD},
        ])

        total_steps = len(p1_train_loader) * EPOCHS_PHASE1
        scheduler_p1 = get_linear_schedule_with_warmup(
            optimizer_p1,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        best_p1_loss = float('inf')

        for epoch in range(EPOCHS_PHASE1):
            # FIX: only put trainable components in train mode
            model.roberta.train()
            model.classifier.train()

            train_loss, correct, total = 0, 0, 0
            loop = tqdm(p1_train_loader, desc=f"Phase 1 Epoch {epoch+1}/{EPOCHS_PHASE1}")
            for batch in loop:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels         = batch['labels'].to(DEVICE)

                optimizer_p1.zero_grad()
                outputs = model.forward_phase1(input_ids, attention_mask, labels)
                loss = outputs['loss']
                loss.backward()

                # FIX: gradient clipping for stable full RoBERTa fine-tune
                nn.utils.clip_grad_norm_(p1_all_params, max_norm=1.0)

                optimizer_p1.step()
                scheduler_p1.step()

                train_loss += loss.item()
                preds    = outputs['logits'].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

            avg_train = train_loss / len(p1_train_loader)

            model.roberta.eval()
            model.classifier.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch in tqdm(p1_val_loader, desc="  Validating"):
                    input_ids      = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    labels         = batch['labels'].to(DEVICE)
                    outputs = model.forward_phase1(input_ids, attention_mask, labels)
                    val_loss    += outputs['loss'].item()
                    preds        = outputs['logits'].argmax(dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += labels.size(0)

            avg_val = val_loss / len(p1_val_loader)
            val_acc = val_correct / val_total
            print(f"Phase 1 Epoch {epoch+1} | "
                  f"Train Loss: {avg_train:.4f} | "
                  f"Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.3f}")

            if avg_val < best_p1_loss:
                best_p1_loss = avg_val
                # FIX: save both roberta base AND head together
                torch.save({
                    'roberta':    model.roberta.state_dict(),
                    'classifier': model.classifier.state_dict(),
                }, 'checkpoints/best_roberta_head.pt')
                print(f"  → Saved best classifier (val_acc={val_acc:.3f})")

        del p1_train_loader, p1_val_loader
    else:
        print(f"\nSkipping Phase 1 — resuming Phase 2 from epoch {resume_phase2_epoch + 1}.")

    # ── PHASE 2: Train Fusion Layer + BART ────────────────────────────────────
    print("\n==================================")
    print(" PHASE 2: Training Fusion Layer + BART")
    print("==================================")

    head_ckpt = 'checkpoints/best_roberta_head.pt'
    if os.path.exists(head_ckpt):
        # FIX: weights_only=True required on PyTorch 2.x (Kaggle)
        ckpt = torch.load(head_ckpt, weights_only=True, map_location=DEVICE)
        model.roberta.load_state_dict(ckpt['roberta'])
        model.classifier.load_state_dict(ckpt['classifier'])
        print("Loaded best Phase 1 RoBERTa + classifier.")
    else:
        print("WARNING: Phase 1 checkpoint not found — using current weights.")

    # Freeze both completely
    for param in model.roberta.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False
    print("RoBERTa + classifier permanently frozen.")

    # Resume mid-Phase-2 if requested
    if resume_phase2_epoch > 0:
        resume_ckpt = f'checkpoints/phase2_epoch_{resume_phase2_epoch}.pt'
        if os.path.exists(resume_ckpt):
            partial = torch.load(resume_ckpt, weights_only=True, map_location=DEVICE)
            model.load_state_dict(partial, strict=False)
            print(f"Resumed Phase 2 from epoch {resume_phase2_epoch}.")
        else:
            print(f"WARNING: {resume_ckpt} not found — starting Phase 2 from scratch.")
            resume_phase2_epoch = 0

    p2_train_loader, p2_val_loader = get_phase2_dataloaders(rob_tokenizer, bart_tokenizer)

    p2_params = (
        list(model.bart.parameters()) +
        list(model.fusion.parameters()) +
        list(model.emotion_embeddings.parameters())
    )
    optimizer_p2 = AdamW(p2_params, lr=LEARNING_RATE_PHASE2)

    remaining_epochs = EPOCHS_PHASE2 - resume_phase2_epoch
    total_p2_steps   = len(p2_train_loader) * remaining_epochs
    scheduler_p2 = get_linear_schedule_with_warmup(
        optimizer_p2,
        num_warmup_steps=int(0.1 * total_p2_steps),
        num_training_steps=total_p2_steps
    )

    best_p2_loss = float('inf')

    for epoch in range(resume_phase2_epoch, EPOCHS_PHASE2):
        # FIX: only trainable components go into train mode — frozen parts stay eval
        model.roberta.eval()
        model.classifier.eval()
        model.bart.train()
        model.fusion.train()
        model.emotion_embeddings.train()

        train_gen_loss = 0
        loop = tqdm(p2_train_loader, desc=f"Phase 2 Epoch {epoch+1}/{EPOCHS_PHASE2}")
        for batch in loop:
            rob_input_ids       = batch['rob_input_ids'].to(DEVICE)
            rob_attention_mask  = batch['rob_attention_mask'].to(DEVICE)
            bart_input_ids      = batch['bart_input_ids'].to(DEVICE)
            bart_attention_mask = batch['bart_attention_mask'].to(DEVICE)
            labels              = batch['labels'].to(DEVICE)

            optimizer_p2.zero_grad()
            outputs = model.forward_phase2(
                rob_input_ids, rob_attention_mask,
                bart_input_ids, bart_attention_mask, labels
            )
            loss_gen = outputs['loss_gen']
            loss_gen.backward()

            # FIX: gradient clipping — BART decoder can explode without this
            nn.utils.clip_grad_norm_(p2_params, max_norm=1.0)

            optimizer_p2.step()
            scheduler_p2.step()

            train_gen_loss += loss_gen.item()
            loop.set_postfix(loss_gen=f"{loss_gen.item():.4f}")

        avg_train_gen = train_gen_loss / len(p2_train_loader)

        model.bart.eval()
        model.fusion.eval()
        model.emotion_embeddings.eval()
        val_gen_loss = 0
        with torch.no_grad():
            for batch in tqdm(p2_val_loader, desc="  Validating"):
                rob_input_ids       = batch['rob_input_ids'].to(DEVICE)
                rob_attention_mask  = batch['rob_attention_mask'].to(DEVICE)
                bart_input_ids      = batch['bart_input_ids'].to(DEVICE)
                bart_attention_mask = batch['bart_attention_mask'].to(DEVICE)
                labels              = batch['labels'].to(DEVICE)
                outputs = model.forward_phase2(
                    rob_input_ids, rob_attention_mask,
                    bart_input_ids, bart_attention_mask, labels
                )
                val_gen_loss += outputs['loss_gen'].item()

        avg_val_gen = val_gen_loss / len(p2_val_loader)
        print(f"Phase 2 Epoch {epoch+1} | "
              f"Train Gen Loss: {avg_train_gen:.4f} | Val Gen Loss: {avg_val_gen:.4f}")

        # FIX: save every epoch so Kaggle session death doesn't lose progress
        torch.save(model.state_dict(), f'checkpoints/phase2_epoch_{epoch+1}.pt')
        print(f"  → Saved epoch {epoch+1} checkpoint.")

        if avg_val_gen < best_p2_loss:
            best_p2_loss = avg_val_gen
            torch.save(model.state_dict(), 'checkpoints/best_decoupled_model.pt')
            print("  → Saved best model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_phase2_epoch", type=int, default=0,
                        help="Resume Phase 2 from this epoch (0 = run from Phase 1)")
    args = parser.parse_args()
    train(resume_phase2_epoch=args.resume_phase2_epoch)
