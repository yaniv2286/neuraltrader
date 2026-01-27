"""
GPU-Optimized Transformer Model
Requires strong GPU for optimal performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, Any, Tuple, Optional

class TransformerModel(nn.Module):
    """
    GPU-optimized Transformer model for stock prediction
    Requires strong GPU for training due to attention mechanism
    """
    
    def __init__(self, input_size: int, d_model: int = 256, n_heads: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, max_seq_len: int = 1000):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.price_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Softplus()
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 3),  # Bull, Bear, Neutral
            nn.Softmax(dim=-1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # GPU optimization
        self.scaler = GradScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_positional_encoding(self, max_seq_len: int, d_model: int):
        """Create positional encoding"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def _initialize_weights(self):
        """Initialize weights for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Transformer processing
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Multiple outputs
        price_pred = self.price_head(x)
        volatility_pred = self.volatility_head(x)
        regime_pred = self.regime_head(x)
        
        return {
            'price': price_pred.squeeze(),
            'volatility': volatility_pred.squeeze(),
            'regime': regime_pred
        }
    
    def to_gpu(self):
        """Move model to GPU"""
        return self.to(self.device)
    
    def train_step(self, batch, optimizer, criterion):
        """Single training step with mixed precision"""
        self.train()
        optimizer.zero_grad()
        
        # Move batch to GPU
        batch_x = batch['x'].to(self.device)
        batch_y = batch['y'].to(self.device)
        
        # Mixed precision forward pass
        with autocast():
            outputs = self.forward(batch_x)
            loss = criterion(outputs['price'], batch_y)
        
        # Mixed precision backward pass
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def validate_step(self, batch, criterion):
        """Single validation step"""
        self.eval()
        
        with torch.no_grad():
            batch_x = batch['x'].to(self.device)
            batch_y = batch['y'].to(self.device)
            
            with autocast():
                outputs = self.forward(batch_x)
                loss = criterion(outputs['price'], batch_y)
        
        return loss.item()
    
    def predict(self, x):
        """Make predictions"""
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            x = x.to(self.device)
            
            with autocast():
                outputs = self.forward(x)
            
            # Move predictions back to CPU
            predictions = {k: v.cpu().numpy() for k, v in outputs.items()}
        
        return predictions
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: dict, loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer_state,
            'loss': loss,
            'model_config': {
                'input_size': self.input_projection.in_features,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'num_layers': self.num_layers
            }
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint['epoch'], checkpoint['loss']
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'Transformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'num_layers': self.num_layers,
            'memory_usage_mb': total_params * 4 / (1024 * 1024)  # Rough estimate
        }
