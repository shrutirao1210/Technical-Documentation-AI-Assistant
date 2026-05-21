import torch
import torch.nn as nn
from transformers import RobertaModel, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from config import NEUTRAL_IDX, CONFIDENCE_THRESHOLD

class EmotionGatedCrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(EmotionGatedCrossAttention, self).__init__()
        # Ensure batch_first=True so dimensions match cleanly
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)
        # Bug 2 Fix: Gate matrix learns from both H and the emotion signal
        self.gate_w = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, H, e_emotion):
        attn_output, _ = self.cross_attn(query=H, key=e_emotion, value=e_emotion)
        
        # Bug 2 Fix: Broadcast emotion to match Sequence Length, then concatenate
        e_expanded = e_emotion.expand(-1, H.size(1), -1)
        gate_input = torch.cat([H, e_expanded], dim=-1)
        
        # Calculate per-token gate alpha
        alpha = torch.sigmoid(self.gate_w(gate_input))
        
        # Element-wise gate application
        # Broadcasting alpha across the sequence length of the attention output
        H_fused = H + (alpha * attn_output)
        
        return H_fused

class DecoupledEmpatheticModel(nn.Module):
    def __init__(self, roberta_name, bart_name, num_emotions):
        super(DecoupledEmpatheticModel, self).__init__()
        
        # 1. RoBERTa - Emotion Classification
        self.roberta = RobertaModel.from_pretrained(roberta_name)
        
        rob_hidden = self.roberta.config.hidden_size
        
        # Trainable Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(rob_hidden, rob_hidden),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(rob_hidden, num_emotions)
        )
        self.emotion_loss = nn.CrossEntropyLoss()
        
        # 2. BART - Context Encoding & Conditonal Generation
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)
        bart_hidden = self.bart.config.hidden_size
        
        # Trainable Emotion Embeddings mapping discrete classes to dense vectors
        self.emotion_embeddings = nn.Embedding(num_emotions, bart_hidden)
        
        # 3. Fusion Layer
        self.fusion = EmotionGatedCrossAttention(bart_hidden)

    def forward_phase1(self, input_ids, attention_mask, labels=None):
        """ Used exclusively to train the classification head """
        rob_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        cls_output = rob_outputs.last_hidden_state[:, 0, :] 
        
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = self.emotion_loss(logits, labels)
            
        return {'loss': loss, 'logits': logits}

    def forward_phase2(self, rob_input_ids, rob_attention_mask, bart_input_ids, bart_attention_mask, labels=None):
        """ Used to train the generation stack - RoBERTa head acts as a frozen emotion oracle """
        
        # 1. Run RoBERTa to get Emotion Prediction (No grad)
        with torch.no_grad():
            rob_outputs = self.roberta(input_ids=rob_input_ids, attention_mask=rob_attention_mask)
            cls_output = rob_outputs.last_hidden_state[:, 0, :]
            emotion_logits = self.classifier(cls_output)
            
            # Confidence Gate Mechanism (Calculated for Metrics/Inference limits, but we bypass for continuous training)
            probs = torch.softmax(emotion_logits, dim=-1)
            max_prob, predicted_class = torch.max(probs, dim=-1)
            
        # Bug 3 Fix: Use a Soft Embedding Matrix Multiplication for completely differentiable gradients
        # Soft embedding is natively differentiable backwards up to the probability distribution
        e_emotion = torch.matmul(probs, self.emotion_embeddings.weight).unsqueeze(1) # [batch, 1, hidden]
        
        # 3. Process Context via BART Encoder
        bart_encoder_outputs = self.bart.model.encoder(
            input_ids=bart_input_ids,
            attention_mask=bart_attention_mask,
            return_dict=True
        )
        H = bart_encoder_outputs.last_hidden_state
        
        # 4. Gated Cross-Attention Fusion
        H_fused = self.fusion(H, e_emotion)
        
        # Override the encoder output with our fused representation
        encoder_outputs = BaseModelOutput(
            last_hidden_state=H_fused,
            hidden_states=bart_encoder_outputs.hidden_states,
            attentions=bart_encoder_outputs.attentions
        )
        
        # 5. BART Decoder Conditional Generation
        outputs = self.bart(
            encoder_outputs=encoder_outputs,
            attention_mask=bart_attention_mask,
            labels=labels
        )
        
        return {
            'loss_gen': outputs.loss,
            'logits': outputs.logits,
            'predicted_emotion': predicted_class,
            'emotion_probs': max_prob
        }
        
    def generate_response(self, rob_input_ids, rob_attention_mask, bart_input_ids, bart_attention_mask, **kwargs):
        """ Inference wrapper — uses soft emotion embedding to exactly match training behaviour """
        
        with torch.no_grad():
            rob_outputs = self.roberta(input_ids=rob_input_ids, attention_mask=rob_attention_mask)
            cls_output = rob_outputs.last_hidden_state[:, 0, :]
            emotion_logits = self.classifier(cls_output)

            probs = torch.softmax(emotion_logits, dim=-1)
            max_prob, predicted_class = torch.max(probs, dim=-1)

            # Confidence gate: collapse uncertain predictions to neutral
            fallback_mask = max_prob < CONFIDENCE_THRESHOLD
            if fallback_mask.any():
                probs[fallback_mask] = 0.0
                probs[fallback_mask, NEUTRAL_IDX] = 1.0
                predicted_class[fallback_mask] = NEUTRAL_IDX

            # FIX: use soft embedding — matches forward_phase2 exactly
            # Training used: matmul(probs, weight)  NOT  embedding[argmax]
            e_emotion = torch.matmul(probs, self.emotion_embeddings.weight).unsqueeze(1)

            bart_encoder_outputs = self.bart.model.encoder(
                input_ids=bart_input_ids,
                attention_mask=bart_attention_mask,
                return_dict=True
            )
            H = bart_encoder_outputs.last_hidden_state

            H_fused = self.fusion(H, e_emotion)
            encoder_outputs = BaseModelOutput(
                last_hidden_state=H_fused,
                hidden_states=bart_encoder_outputs.hidden_states,
                attentions=bart_encoder_outputs.attentions
            )

            generated_ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=bart_attention_mask,
                **kwargs
            )

            return generated_ids, predicted_class, max_prob
