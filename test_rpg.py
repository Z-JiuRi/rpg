import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from models.rpg import RecurrentParameterGenerator

def test_rpg():
    # Configuration
    config = {
        "pos_embedding": {
            "hidden_dim": 64,
            "pe_fusion_type": "concat",
            "token_pe_type": "learnable",
            "max_layers": 4,
            "max_tokens": 64
        },
        "proto": {
            "cond_dim": 64,
            "hidden_dim": 64,
            "prototype_dim": 32,
            "num_seeds": 5,
            "guidance_p": 0.1,
            "learnable_null": True,
            "rnn_type": "gru",
            "num_layers": 1,
            "dropout": 0.0,
            "bidirectional": False
        },
        "denoiser": {
            "type": "unet",
            "dim": 32,
            "layer_channels": [32, 64],
            "init_ch": 16,
            "final_ch": 16,
            "num_res_blocks": 1,
            "attention_resolutions": [16],
            "dropout": 0.1,
            "use_linear_attention": True,
            "cond_dim": 64,
            "proto_dim": 32
        },
        "ddpm": {
            "timesteps": 100,
            "beta_scheduler": "cosine",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "prediction_type": "eps"
        }
    }

    print("Initializing model...")
    model = RecurrentParameterGenerator(**config)
    print("Model initialized successfully.")

    # Mock Data
    B, L = 2, 32
    x_0 = torch.randn(B, 16, L)  # init_ch=16
    cond = torch.randn(B, 64)    # cond_dim=64
    layer_idx = torch.randint(0, 4, (B, L))
    token_idx = torch.randint(0, 64, (B, L))
    perm_state = torch.randint(0, 5, (B,))

    # Test Forward
    print("Testing Forward...")
    loss_dict = model(x_0, cond, layer_idx, token_idx, perm_state)
    print(f"Loss: {loss_dict['loss'].item()}")

    # Test Sample
    print("Testing Sample...")
    samples = model.sample(cond, layer_idx, token_idx)
    print(f"Sample shape: {samples.shape}")
    
    assert samples.shape == (B, 16, L)
    print("Test Passed!")

if __name__ == "__main__":
    test_rpg()
