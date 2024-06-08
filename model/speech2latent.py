import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
from espnet.nets.pytorch_backend.conformer.encoder import Encoder as ConformerEncoder
from torch import optim
import pytorch_lightning as pl
import numpy as np

# train stage2
class Seq2SeqModel(pl.LightningModule):
    def __init__(self, 
                 learning_rate=1e-4, 
                 reference_time=3, 
                 input_lenth=10, 
                 mfcc=False, 
                 video_fps=25, 
                 audio_fps=50):
        super().__init__()
        self.seq2seq = GradTalker( mfcc=mfcc)
        
        self.reference_time = reference_time
        self.input_lenth = input_lenth
        self.learning_rate = learning_rate
        self.mfcc = mfcc
        self.video_fps = video_fps
        self.audio_fps = audio_fps   # 50 for hubert, 100 for mfcc

    def training_step(self, batch, batch_idx):
        audio_feats, gt_latent_feats, masking = batch[0].float(), batch[1].float(), batch[2].float()
        refer_visual_feats = gt_latent_feats[:,self.input_lenth*self.video_fps:,:]
        ref_len = self.reference_time * self.video_fps
        refer_visual_feats = refer_visual_feats[:,:ref_len,:]

        if self.mfcc:
            audio_feats = audio_feats[:,:self.input_lenth*self.audio_fps*2,:]
        else:
            audio_feats = audio_feats[:,:,:self.input_lenth*self.audio_fps,:]

        gt_latent_feats = gt_latent_feats[:,:self.input_lenth*self.video_fps,:]
        masking = masking[:,:self.input_lenth*self.video_fps]
        masking = masking.unsqueeze(-1)
        
        predictions = self.seq2seq(audio_feats, refer_visual_feats)

        loss = self.mean_flat(masking*((gt_latent_feats-predictions)**2))
        self.log("train_loss", loss)
        return loss

    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(0, len(tensor.shape))))

    def validation_step(self, batch, batch_idx):
        audio_feats, gt_latent_feats, masking = batch[0].float(), batch[1].float(), batch[2].float()
        
        refer_visual_feats = gt_latent_feats[:,self.input_lenth*self.video_fps:,:]
        ref_len = self.reference_time * self.video_fps
        
        refer_visual_feats = refer_visual_feats[:,:ref_len,:]
        if self.mfcc:
            audio_feats = audio_feats[:,:self.input_lenth*self.audio_fps*2,:]
        else:
            audio_feats = audio_feats[:,:,:self.input_lenth*self.audio_fps,:]
        gt_latent_feats = gt_latent_feats[:,:self.input_lenth*self.video_fps,:]
        masking = masking[:,:self.input_lenth*self.video_fps]
        
        masking = masking.unsqueeze(-1)
        
        predictions = self.seq2seq(audio_feats, refer_visual_feats)
        
        loss = self.mean_flat(masking*((gt_latent_feats-predictions)**2))
        self.log("val_loss", loss)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class BaseModule(torch.nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        """
        Returns number of trainable parameters of the module.
        """
        num_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params += np.prod(param.detach().cpu().numpy().shape)
        return num_params


    def relocate_input(self, x: list):
        """
        Relocates provided tensors to the same device set for the module.
        """
        device = next(self.parameters()).device
        for i in range(len(x)):
            if isinstance(x[i], torch.Tensor) and x[i].device != device:
                x[i] = x[i].to(device)
        return x

    
class attentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super(attentionLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, tar):
        src = src.transpose(0, 1) # B, T, C -> T, B, C
        tar = tar.transpose(0, 1) # B, T, C -> T, B, C
        src2 = self.self_attn(tar, src, src, attn_mask=None,
                              key_padding_mask=None)[0]
        
        src = self.dropout1(src2)
        src = self.linear1(src)
        src = self.norm1(src)

        src = src.transpose(0, 1) # T, B, C -> B, T, C
        return src

class LatentPredictor(BaseModule):
    
    def __init__(self, audio_dim=1024, latent_dim=512, mfcc=False):
        super(LatentPredictor, self).__init__()
        self.mfcc = mfcc
        if mfcc:
            self.down_sample_audio_nar = torch.nn.Sequential(
                torch.nn.Conv1d(
                    audio_dim, 
                    latent_dim, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                ),  # audio_dim is 13 for mfcc
                torch.nn.Conv1d(
                    latent_dim, 
                    latent_dim, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1
                )
            )
        else:
            self.down_sample_audio_nar = torch.nn.Conv1d(
                in_channels=audio_dim,
                out_channels=latent_dim,
                kernel_size=3,
                stride=2,
                padding=1
            )
        self.reference_encoder = ConformerEncoder(
                idim=0,
                attention_dim=latent_dim,
                attention_heads=2,
                linear_units=latent_dim,
                num_blocks=2,
                input_layer=None,
                dropout_rate=0.2,
                positional_dropout_rate=0.2,
                attention_dropout_rate=0.2,
                normalize_before=False,
                concat_after=False,
                positionwise_layer_type="linear",
                positionwise_conv_kernel_size=3,
                macaron_style=True,
                pos_enc_layer_type="rel_pos",
                selfattention_layer_type="rel_selfattn",
                use_cnn_module=True,
                cnn_module_kernel=13
            )
        
        self.speech_encoder = ConformerEncoder(
                idim=0,
                attention_dim=latent_dim,
                attention_heads=2,
                linear_units=latent_dim,
                num_blocks=4,
                input_layer=None,
                dropout_rate=0.2,
                positional_dropout_rate=0.2,
                attention_dropout_rate=0.2,
                normalize_before=False,
                concat_after=False,
                positionwise_layer_type="linear",
                positionwise_conv_kernel_size=3,
                macaron_style=True,
                pos_enc_layer_type="rel_pos",
                selfattention_layer_type="rel_selfattn",
                use_cnn_module=True,
                cnn_module_kernel=13
            )


        self.coarse_decoder = ConformerEncoder(
                idim=0,
                attention_dim=latent_dim,
                attention_heads=2,
                linear_units=latent_dim,
                num_blocks=4,
                input_layer=None,
                dropout_rate=0.2,
                positional_dropout_rate=0.2,
                attention_dropout_rate=0.2,
                normalize_before=False,
                concat_after=False,
                positionwise_layer_type="linear",
                positionwise_conv_kernel_size=3,
                macaron_style=True,
                pos_enc_layer_type="rel_pos",
                selfattention_layer_type="rel_selfattn",
                use_cnn_module=True,
                cnn_module_kernel=13
            )
        self.out_proj = torch.nn.Linear(latent_dim, latent_dim)

        if not mfcc:
            self.weights = torch.nn.Parameter(torch.zeros(25))  # weighted sum strategy

        self.cross_attention = attentionLayer(d_model=latent_dim, nhead=8)
    
    def forward(self, audio_feats, refer_visual_feats):
        
        if self.mfcc:
            x = self.down_sample_audio_nar(audio_feats.transpose(1,2)).transpose(1,2)
        else:
            norm_weights = F.softmax(self.weights, dim=-1)
            weighted_feature = (norm_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * audio_feats).sum(dim=1)
            x = self.down_sample_audio_nar(weighted_feature.transpose(1,2)).transpose(1,2)

        # reference network
        # torch.Size([B, 75, 512]) Cross attention allows any length
        ref_condition, _ = self.reference_encoder(refer_visual_feats, masks=None) 
        x = self.cross_attention(ref_condition, x)
        
        x, _ = self.speech_encoder(x, masks=None)
        x, _ = self.coarse_decoder(x, masks=None)
        
        y = self.out_proj(x)
        return y

class GradTalker(BaseModule):
    def __init__(self, mfcc=False):
        super(GradTalker, self).__init__()
        self.encoder = LatentPredictor(mfcc=mfcc)
        
    def forward(self, audio_feats, refer_visual_feats):
        return self.encoder(audio_feats, refer_visual_feats)

    def compute_loss(self, audio_feats, ref_latent):
        return self.encoder(audio_feats, ref_latent)
