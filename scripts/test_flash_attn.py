"""Quick test for flash attention."""

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test flash attention import
try:
    from flash_attn import flash_attn_func
    print("\n[OK] flash_attn imported successfully!")

    # Quick functional test
    batch, heads, seq_len, dim = 2, 8, 128, 64
    q = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, seq_len, heads, dim, device="cuda", dtype=torch.float16)

    out = flash_attn_func(q, k, v)
    print(f"[OK] flash_attn_func works! Output shape: {out.shape}")

except ImportError as e:
    print(f"\n[FAIL] flash_attn import failed: {e}")
except Exception as e:
    print(f"\n[FAIL] flash_attn test failed: {e}")

# Test torch.compile
print("\nTesting torch.compile...")
try:
    model = torch.nn.Linear(64, 64).cuda()
    compiled = torch.compile(model, mode="reduce-overhead")
    x = torch.randn(2, 64, device="cuda")
    out = compiled(x)
    print(f"[OK] torch.compile works! Output shape: {out.shape}")
except Exception as e:
    print(f"[FAIL] torch.compile failed: {e}")
