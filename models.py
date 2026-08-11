import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class PhysicsTransformerEstimator(nn.Module):
    """
    Sequence -> (mass, mu) estimator.

    Two optional extensions to the original velocity-only model:

    1. `input_dim` is respected and validated (it was already a parameter, but
       a mismatch used to fail deep inside the projection with an opaque error).

    2. STATIC CONDITIONING TOKEN (`cond_dim`)
       cond_dim=0 -> disabled (original behaviour, forward() ignores `cond`)
       cond_dim>0 -> a per-sample static vector (e.g. arm configuration) is
       projected to d_model and PREPENDED to the encoder sequence as an extra
       token. The encoder self-attention, and the mass/friction query attention
       downstream, can then attend to it.

       Prepending is used rather than FiLM or per-timestep addition because it
       is the smallest change that still lets every downstream head see the
       condition, and it needs no change to the attention shapes: the learned
       queries already attend over a key sequence whose length they do not
       assume.

       The condition token deliberately receives NO positional encoding -- it is
       not part of the time series and should not be assigned a position.
    """
    def __init__(
        self,
        input_dim=1,
        d_model=64,
        nhead=4,
        num_encoder_layers=2, 
        dim_feedforward=128, 
        seq_len=60, 
        dropout=0.1,
        sharpness=10.0,
        cross_sharpness=10.0,
        m_sharpness=30.0,
        mu_sharpness=10.0,
        version=1,
        max_mass_scale=3.0,
        max_mu_scale=1.0,
        cond_dim=0,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.sharpness = sharpness 
        self.cross_sharpness = cross_sharpness
        self.m_sharpness = m_sharpness
        self.mu_sharpness = mu_sharpness
        self.version = version
        self.max_mass_scale = max_mass_scale
        self.max_mu_scale = max_mu_scale
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10)

        # --- Static conditioning branch ---
        if self.cond_dim > 0:
            self.cond_proj = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.cond_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        if not self.version in [3, 4, 5, 6, 7, 8]:
            self.time_queries = nn.Parameter(torch.randn(1, seq_len, d_model))
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
            self.norm_dec = nn.LayerNorm(d_model)
            self.ffn_dec = nn.Sequential(
                nn.Linear(d_model, dim_feedforward), 
                nn.ReLU(), 
                nn.Linear(dim_feedforward, d_model)
            )
            self.norm_ffn = nn.LayerNorm(d_model)

        self.net_force_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        ) 
        self.fric_force_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        ) 

        if self.version in [1, 2, 3, 4, 5, 6, 7, 8]:
            self.phys_net_proj = nn.Linear(d_model, 1)
            self.phys_fric_proj = nn.Linear(d_model, 1)
        elif self.version == 2.5:
            self.phys_net_proj = nn.Linear(d_model * 2, 1)
            self.phys_fric_proj = nn.Linear(d_model * 2, 1)

        # MASS ESTIMATION
        if self.version in [1, 3, 4, 6]:
            self.q_mass = nn.Parameter(torch.randn(1, 1, d_model))
            self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        elif self.version in [2, 2.5]:
            self.q_mass = nn.Parameter(torch.randn(1, 1, d_model * 2))
            self.mass_attn = nn.MultiheadAttention(d_model * 2, 1, batch_first=True)
            self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model * 2, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        elif self.version == 5:
            self.q_mass = nn.Parameter(torch.randn(1, 1, d_model))
            self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Softplus())
        elif self.version == 7:
            self.q_mass = nn.Parameter(torch.randn(1, 1, d_model))
            self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        elif self.version == 8:
            self.q_mass = nn.Parameter(torch.randn(1, 1, d_model))
            self.mass_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mass_pred_mlp = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
            nn.init.constant_(self.mass_pred_mlp[2].bias, -1.6)

        # FRICTION ESTIMATION
        if self.version in [1, 3]:
            self.q_fric = nn.Parameter(torch.randn(1, 1, d_model))
            self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model + 1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        elif self.version in [2, 2.5]:
            self.q_fric = nn.Parameter(torch.randn(1, 1, d_model * 2))
            self.fric_attn = nn.MultiheadAttention(d_model * 2, 1, batch_first=True) 
            self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model * 2 + 1, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        elif self.version == 4:
            self.q_fric = nn.Parameter(torch.randn(1, 1, d_model))
            self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())
        elif self.version == 5:
            self.q_fric = nn.Parameter(torch.randn(1, 1, d_model))
            self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
        elif self.version in [6, 7]:
            self.q_fric = nn.Parameter(torch.randn(1, 1, d_model))
            self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
        elif self.version == 8:
            self.q_fric = nn.Parameter(torch.randn(1, 1, d_model))
            self.fric_attn = nn.MultiheadAttention(d_model, 1, batch_first=True)
            self.mu_pred_mlp = nn.Sequential(nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1), nn.Sigmoid())
            nn.init.constant_(self.mu_pred_mlp[2].bias, -0.85)

    def forward(self, extracted_vel, cond=None):
        """
        Args:
            extracted_vel: (B, T) or (B, T, input_dim) velocity sequence.
            cond: (B, cond_dim) static per-sample features, or None. Ignored
                when the model was built with cond_dim=0.
        """
        DEBUG_MODEL = False
        
        if len(extracted_vel.shape) == 2:
            x = extracted_vel.unsqueeze(-1)
        else:
            x = extracted_vel 
            
        B, T, C = x.shape

        if C != self.input_dim:
            raise ValueError(
                f"Model built with input_dim={self.input_dim} but received "
                f"{C} channels."
            )

        if DEBUG_MODEL:
            print("\n" + "="*60)
            print(" PHYSICS TRANSFORMER INTERNAL DEBUGGER ")
            print("="*60)
            print(f"1. Raw Input (x):       Mean {x.mean().item():.4f} | Std {x.std().item():.4f}")

        z = self.input_proj(x)
        
        if self.version in [5, 8]:
            z = z * math.sqrt(self.d_model)
        else:
            z = self.input_norm(z)
            z = z * math.sqrt(self.d_model)

        z = self.pos_encoder(z)

        # --- Prepend the static conditioning token (after positional encoding,
        #     so the condition is not assigned a time position) ---
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(
                    "Model was built with cond_dim>0 but forward() received "
                    "cond=None. Pass the static feature tensor, or rebuild with "
                    "cond_dim=0."
                )
            cond_tok = self.cond_norm(self.cond_proj(cond)).unsqueeze(1)  # (B, 1, d)
            z = torch.cat([cond_tok, z], dim=1)                           # (B, T+1, d)

            if DEBUG_MODEL:
                print(f"   Cond token: Mean {cond_tok.mean().item():.4f} | "
                      f"Std {cond_tok.std().item():.4f}")

        h_enc = self.transformer_encoder(z)
        
        if not self.version in [3, 4, 5, 6, 7, 8]:
            q_dec = self.time_queries.expand(B, -1, -1)
            attn_output, cross_weights_raw = self.cross_attn(
                query=q_dec*self.cross_sharpness, 
                key=h_enc, 
                value=h_enc, 
                average_attn_weights=False  
            )
            cross_weights = cross_weights_raw / (cross_weights_raw.sum(dim=-1, keepdim=True) + 1e-10)
            h_dec = self.norm_dec(q_dec + attn_output)
            h_dec = self.norm_ffn(h_dec + self.ffn_dec(h_dec))

        if self.version in [3, 4, 5, 6, 7, 8]:
            feat_net = self.net_force_mlp(h_enc)   
            feat_fric = self.fric_force_mlp(h_enc) 
        else:
            feat_net = self.net_force_mlp(h_dec)   
            feat_fric = self.fric_force_mlp(h_dec) 

        if self.version in [2, 2.5]:
            # Versions 2/2.5 concatenate the decoder features (length seq_len)
            # with the encoder output. Drop the prepended condition token so the
            # two sequences line up.
            h_enc_time = h_enc[:, 1:, :] if self.cond_dim > 0 else h_enc
            feat_net_enriched = torch.cat([feat_net, h_enc_time], dim=-1)
            feat_fric_enriched = torch.cat([feat_fric, h_enc_time], dim=-1)
        
        if self.version in [1, 2, 3, 4, 5, 6, 7, 8]:
            phys_net = self.phys_net_proj(feat_net)   
            phys_fric = self.phys_fric_proj(feat_fric) 
            phys_net = torch.abs(phys_net)
            phys_fric = torch.abs(phys_fric)
        elif self.version == 2.5:
            phys_net = self.phys_net_proj(feat_net_enriched)   
            phys_fric = self.phys_fric_proj(feat_fric_enriched) 
            phys_net = torch.abs(phys_net)
            phys_fric = torch.abs(phys_fric)

        # The PINN losses expect per-timestep sequences of length T. When a
        # condition token is prepended, versions 3-8 build feat_* from h_enc and
        # so carry an extra leading entry; strip it so the physics residuals stay
        # aligned with the recorded force/acceleration windows.
        if self.cond_dim > 0 and self.version in [3, 4, 5, 6, 7, 8]:
            phys_net = phys_net[:, 1:, :]
            phys_fric = phys_fric[:, 1:, :]

        q_m_batch = self.q_mass.expand(B, -1, -1)
        if self.version in [1, 3, 4, 5, 6, 7, 8]:
            mass_ctx, mass_weights = self.mass_attn(query=q_m_batch*self.m_sharpness, key=feat_net, value=feat_net)
        elif self.version in [2, 2.5]:
            mass_ctx, mass_weights = self.mass_attn(query=q_m_batch*self.m_sharpness, key=feat_net_enriched, value=feat_net_enriched)
        
        raw_mass_pred = self.mass_pred_mlp(mass_ctx.squeeze(1))
        mass_pred = raw_mass_pred * self.max_mass_scale

        q_f_batch = self.q_fric.expand(B, -1, -1)
        if self.version in [1, 3, 4, 5, 6, 7, 8]:
            fric_ctx, fric_weights = self.fric_attn(query=q_f_batch*self.mu_sharpness, key=feat_fric, value=feat_fric)
        elif self.version in [2, 2.5]:
            fric_ctx, fric_weights = self.fric_attn(query=q_f_batch*self.mu_sharpness, key=feat_fric_enriched, value=feat_fric_enriched)

        if self.version in [4, 5, 6, 7, 8]:
            fric_input = torch.cat([fric_ctx.squeeze(1)], dim=-1)
        else:
            fric_input = torch.cat([fric_ctx.squeeze(1), mass_pred], dim=-1)
            
        raw_mu_pred = self.mu_pred_mlp(fric_input)
        mu_pred = raw_mu_pred * self.max_mu_scale

        preds = torch.cat([mass_pred, mu_pred], dim=-1)

        if self.version in [3, 4, 5, 6, 7, 8]:
            return preds, (None, mass_weights, fric_weights), (phys_net, phys_fric)
        else:
            return preds, (cross_weights, mass_weights, fric_weights), (phys_net, phys_fric)