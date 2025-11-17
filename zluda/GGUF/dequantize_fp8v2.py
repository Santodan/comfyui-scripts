#!/usr/bin/env python
"""dequantize_fp8.py — *streaming* FP8 → full‑precision converter (v3.2 - Modified)

### What’s new in **v3.2**
* **`scaled_fp8` is now always removed** when `--strip-fp8` is set,
  regardless of its dtype.
* Documentation cleaned up to reflect this behaviour.
### MODIFIED: Now uses --src and --dst arguments for clarity.
"""

import argparse
import re
import sys
import torch
from safetensors.torch import load_file, save_file

# --------- helpers & constants ---------
_WEIGHT_RE       = re.compile(r"\.weight$")
_FP8_DTYPES      = {torch.float8_e4m3fn, torch.float8_e5m2}
_SCALE_PAT       = re.compile(r"\.(?:scale_weight|scale_input)$")
DTYPE_MAP        = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def find_reciprocal_scale(state: dict[str, torch.Tensor], base: str) -> float:
    """Return the reciprocal scale associated with *base*."""
    for suffix in ("scale_weight", "scale_reciprocal"):
        key = f"{base}.{suffix}"
        if key in state:
            return state[key].to(torch.float32).item()

    scale_key = f"{base}.scale"
    if scale_key in state:
        return 1.0 / state[scale_key].to(torch.float32).item()

    raise KeyError(f"No scale tensor found for base '{base}'")


def in_place_convert(state: dict[str, torch.Tensor], *, out_dtype: torch.dtype, strip_fp8: bool):
    """Cast **all** tensors to *out_dtype* in‑place, restoring FP8 weights."""
    # ---- 1) Restore FP8 weights ----
    fp8_weight_keys = [k for k, t in state.items() if _WEIGHT_RE.search(k) and t.dtype in _FP8_DTYPES]

    restored = 0
    for key in fp8_weight_keys:
        tensor = state[key]
        base   = key[:-7]
        recip  = find_reciprocal_scale(state, base)

        state[key] = (tensor.to(torch.float32) * recip).to(out_dtype)
        restored += 1
        print(f"↩︎ {key:>60} | recip {recip:.6g} | → {out_dtype}")

        if strip_fp8:
            del tensor
            for suf in ("scale_weight", "scale_input"):
                state.pop(f"{base}.{suf}", None)

    # ---- 2) Cast remaining tensors & cleanup ----
    for k in list(state.keys()):
        t = state[k]

        if strip_fp8:
            if k.endswith(".scaled_fp8"):
                del state[k]
                continue
            if _SCALE_PAT.search(k) or (_WEIGHT_RE.search(k) and t.dtype in _FP8_DTYPES):
                del state[k]
                continue

        if t.dtype != out_dtype:
            state[k] = t.to(out_dtype)

    print("\n―――――――― CONVERSION SUMMARY ―――――――")
    print(f"FP8 weights restored : {restored}")
    print(f"Total tensors         : {len(state)} (after cast/clean)")
    print("――――――――――――――――――――――――――――――――")


# ---------------- CLI ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="In‑place FP8 → full‑precision conversion with global dtype cast")
    
    # --- THIS IS THE CORRECTED SECTION ---
    # Changed from positional arguments to named arguments for clarity and robustness
    ap.add_argument("--src", required=True, help="Input FP8 .safetensors file")
    ap.add_argument("--dst", required=True, help="Output .safetensors file")
    # --- END OF CORRECTION ---
    
    ap.add_argument("--dtype", choices=DTYPE_MAP.keys(), default="bf16", help="Target dtype for *all* tensors (default: bf16)")
    ap.add_argument("--strip-fp8", action="store_true", help="Remove FP8 & scale tensors after convert to minimise size")
    args = ap.parse_args()

    out_dtype = DTYPE_MAP[args.dtype]

    # --- Use the new argument names ---
    print("Loading", args.src)
    sd = load_file(args.src, device="cpu")

    in_place_convert(sd, out_dtype=out_dtype, strip_fp8=args.strip_fp8)

    print("Saving", args.dst)
    try:
        # --- Use the new argument names ---
        save_file(sd, args.dst)
    except Exception as err:
        print("❌ Failed to save .safetensors:", err, file=sys.stderr)
        sys.exit(1)

    print("Done ✅")


if __name__ == "__main__":
    main()