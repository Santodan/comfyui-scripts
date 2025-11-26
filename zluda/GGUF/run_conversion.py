import os
import sys
import subprocess
import logging
import re
import glob
import shutil
from datetime import datetime

# --- CONFIGURATION ---
# Sorted Logical Order (Low -> High Quality)
QUANTIZATION_OPTIONS = [
    "IQ2_XS", "IQ2_S", "Q2_K",
    "IQ3_XXS", "IQ3_S", "IQ3_M", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "IQ4_NL", "IQ4_XS", "Q4_0", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0",
    "BF16", "F16",
    "FP8_E4M3FN", "FP8_E4M3FN (All)",
    "FP8_E5M2", "FP8_E5M2 (All)"
]

# --- SETUP LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# --- IMPORTS ---
try:
    import upload_to_hf_v4 as uploader
    # We need HfApi for the interactive selector
    from huggingface_hub import HfApi, login, create_repo
    UPLOADER_AVAILABLE = True
except ImportError:
    uploader = None
    UPLOADER_AVAILABLE = False

try:
    import torch
    from safetensors.torch import save_file, load_file
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- FP8 LOGIC ---
if TORCH_AVAILABLE:
    class FP8Quantizer:
        def __init__(self, quant_dtype: str = "float8_e5m2"):
            if not hasattr(torch, quant_dtype): raise ValueError(f"Unsupported: {quant_dtype}")
            self.quant_dtype = quant_dtype

        def quantize_weights(self, weight: torch.Tensor, name: str) -> torch.Tensor:
            if not weight.is_floating_point(): return weight
            
            # Quality Guards (Skip sensitive layers)
            if weight.ndim == 1: return weight.to(dtype=torch.float16)
            for kw in ["norm", "time_emb", "proj_in", "proj_out", "guidance_in"]:
                if kw in name: return weight.to(dtype=torch.float16)

            target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            weight_on_target = weight.to(target_device)
            max_val = torch.max(torch.abs(weight_on_target))
            
            if max_val == 0:
                target_torch_dtype = getattr(torch, self.quant_dtype)
                return torch.zeros_like(weight_on_target, dtype=target_torch_dtype)
            
            divisor = 57344.0 if "e5m2" in self.quant_dtype else 448.0
            scale = max_val / divisor
            scale = torch.max(scale, torch.tensor(1e-12, device=target_device, dtype=weight_on_target.dtype))
            
            quantized = torch.round(weight_on_target / scale * 127.0) / 127.0 * scale
            return quantized.to(dtype=getattr(torch, self.quant_dtype))

        def convert_file(self, src, dst, unet_only=True):
            logging.info(f"FP8 Conversion ({self.quant_dtype}) -> {os.path.basename(dst)}")
            try:
                if src.endswith(".safetensors"): state_dict = load_file(src)
                else: state_dict = torch.load(src, map_location="cpu")
                
                new_dict = {}
                total = len(state_dict)
                for i, (name, param) in enumerate(state_dict.items()):
                    if unet_only and "model.diffusion_model" not in name: continue
                    if i % 200 == 0: print(f"  Processing tensor {i}/{total}...", end="\r")
                    
                    if isinstance(param, torch.Tensor) and param.is_floating_point():
                        new_dict[name] = self.quantize_weights(param, name)
                    else:
                        new_dict[name] = param
                print("")
                save_file(new_dict, dst)
                del state_dict; del new_dict
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                return True
            except Exception as e:
                logging.error(f"FP8 Error: {e}")
                return False
else:
    class FP8Quantizer:
        def __init__(self, *args, **kwargs): pass

# --- HELPER FUNCTIONS ---
def get_input(prompt_text, default=None):
    if default:
        user_in = input(f"{prompt_text} [{default}]: ").strip()
        return user_in if user_in else default
    return input(f"{prompt_text}: ").strip()

def run_cmd(cmd):
    logging.info(f"CMD: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        logging.error("Command Failed.")
        return False

def interactive_repo_select(api, token, label="GGUF"):
    """ Fetches user repos and allows selection via number """
    try:
        user_info = api.whoami(token=token)
        username = user_info['name']
        print(f"\n--- Fetching repositories for user: {username} ---")
        
        # List models (filter by owner)
        models = list(api.list_models(author=username, limit=100))
        # Sort alphabetically
        models.sort(key=lambda x: x.modelId)
        
        print(f"Select Target Repo for {label}:")
        for i, model in enumerate(models):
            print(f"  [{i+1}] {model.modelId}")
        
        print("  [N] Create New Repository")
        print("  [M] Manual Entry (Type ID manually)")
        print("  [S] Skip this upload type")
        
        while True:
            choice = input(f"Select option for {label}: ").strip().upper()
            if choice == 'S': return None
            
            if choice == 'M':
                return input(f"Enter manual Repo ID for {label}: ").strip()
            
            if choice == 'N':
                new_name = input("Enter new repo name: ").strip()
                is_private = get_input("Private repo? (y/n)", default="y").lower() == "y"
                full_id = f"{username}/{new_name}"
                try:
                    create_repo(repo_id=full_id, private=is_private, token=token, exist_ok=True)
                    print(f"Created {full_id}")
                    return full_id
                except Exception as e:
                    print(f"Error creating repo: {e}")
                    continue

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    return models[idx].modelId
            
            print("Invalid selection.")

    except Exception as e:
        print(f"Error fetching repos: {e}")
        return input(f"Enter Repo ID manually for {label}: ").strip()

# --- MAIN WIZARD ---
def main():
    print("\n=== GGUF & FP8 CONVERTER (CLI v7) ===")
    
    # 1. Files
    print("\n--- 1. Input Files ---")
    files_str = get_input("Enter file paths (separated by space)")
    input_files = []
    for pattern in files_str.split():
        input_files.extend(glob.glob(pattern))
    
    if not input_files:
        print("No files found. Exiting.")
        return
    
    print(f"Found {len(input_files)} files.")

    # 2. Output
    print("\n--- 2. Output Configuration ---")
    out_root = get_input("Base Output Directory", default="./output")
    use_subfolder = get_input("Create subfolder per model? (y/n)", default="y").lower() == "y"

    # 3. Quants
    print("\n--- 3. Quantization Selection ---")
    for i, q in enumerate(QUANTIZATION_OPTIONS):
        print(f"{i+1:2d}. {q}")
    
    q_indices = get_input("Enter numbers to generate (e.g. 1 5 24)")
    selected_quants = []
    for idx in q_indices.split():
        if idx.isdigit():
            i = int(idx) - 1
            if 0 <= i < len(QUANTIZATION_OPTIONS):
                selected_quants.append(QUANTIZATION_OPTIONS[i])
    
    print(f"Selected: {selected_quants}")

    # 4. Upload
    do_upload = False
    repo_gguf = ""
    repo_fp8 = ""
    dest_folder_gguf = ""
    dest_folder_fp8 = ""
    token = os.getenv("HUGGING_FACE_HUB_TOKEN", "")

    if UPLOADER_AVAILABLE:
        print("\n--- 4. Upload Configuration ---")
        if get_input("Upload to Hugging Face? (y/n)", default="n").lower() == "y":
            do_upload = True
            if not token:
                token = get_input("Enter HF Token")
            
            # Login to get API access
            login(token=token, add_to_git_credential=False)
            api = HfApi(token=token)

            # Interactive Selection for GGUF
            if any("FP8" not in q for q in selected_quants):
                repo_gguf = interactive_repo_select(api, token, "GGUF")
                if repo_gguf:
                    dest_folder_gguf = get_input(f"Folder inside '{repo_gguf}' [Enter for root]")

            # Interactive Selection for FP8
            if any("FP8" in q for q in selected_quants):
                repo_fp8 = interactive_repo_select(api, token, "FP8")
                if repo_fp8:
                    dest_folder_fp8 = get_input(f"Folder inside '{repo_fp8}' [Enter for root]")

    # 5. Cleanup Strategy
    print("\n--- 5. Cleanup Strategy ---")
    cleanup_mode = get_input("Cleanup after each model? (y/n) [Saves disk space]", default="y").lower() == "y"
    
    keep_dequant = get_input("Keep intermediate Dequant file? (y/n)", default="n").lower() == "y"
    keep_convert = get_input("Keep intermediate GGUF Source (CONVERT)? (y/n)", default="n").lower() == "y"

    # --- START PROCESSING ---
    quant_cmd = "./llama-quantize" if os.path.exists("./llama-quantize") else "llama-quantize"
    
    for fpath in input_files:
        model_base = os.path.basename(fpath)
        name = re.sub(r'-(f16|F16|BF16|CONVERT|UnFixed|FIXED)$', '', os.path.splitext(model_base)[0], flags=re.IGNORECASE)
        
        out_dir = os.path.join(out_root, name) if use_subfolder else out_root
        os.makedirs(out_dir, exist_ok=True)

        logging.info(f"\n>>> PROCESSING MODEL: {name}")
        generated_files = []

        # --- FP8 ---
        fp8_quants = [q for q in selected_quants if "FP8" in q]
        for q in fp8_quants:
            is_e5m2 = "E5M2" in q
            is_all = "(All)" in q
            dtype = "float8_e5m2" if is_e5m2 else "float8_e4m3fn"
            suffix = "_All" if is_all else ""
            
            dst = os.path.join(out_dir, f"{name}-{q.split(' ')[0]}{suffix}.safetensors")
            
            if TORCH_AVAILABLE:
                qzer = FP8Quantizer(dtype)
                if qzer.convert_file(fpath, dst, unet_only=(not is_all)):
                    generated_files.append(dst)
            else:
                logging.error("Torch missing. Skipping FP8.")

        # --- GGUF ---
        gguf_quants = [q for q in selected_quants if "FP8" not in q]
        if gguf_quants:
            gguf_src = None
            dq = None

            # Convert to GGUF Source
            if fpath.endswith(".safetensors"):
                curr = fpath
                dq = os.path.join(out_dir, f"{name}-dequant.safetensors")
                
                if os.path.exists("dequantize_fp8v2.py"):
                    logging.info("Dequantizing (FP8 check)...")
                    subprocess.run([sys.executable, "dequantize_fp8v2.py", "--src", fpath, "--dst", dq, "--strip-fp8", "--dtype", "fp16"])
                    if os.path.exists(dq): curr = dq
                
                conv = os.path.join(out_dir, f"{name}-CONVERT.gguf")
                if not os.path.exists(conv):
                    logging.info("Converting to GGUF F16...")
                    subprocess.run([sys.executable, "convert.py", "--src", curr, "--dst", conv])
                
                if os.path.exists(conv): gguf_src = conv
            
            elif fpath.endswith(".gguf"):
                gguf_src = fpath

            # Run Quants
            if gguf_src:
                for q in gguf_quants:
                    final_path = os.path.join(out_dir, f"{name}-{q}.gguf")
                    
                    if q in ["F16", "BF16"]:
                        shutil.copy(gguf_src, final_path)
                        generated_files.append(final_path)
                        continue

                    unfixed = os.path.join(out_dir, f"{name}-{q}-UnFixed.gguf")
                    logging.info(f"Quantizing {q}...")
                    subprocess.run([quant_cmd, gguf_src, unfixed, q])
                    
                    if os.path.exists(unfixed):
                        # Fix Tensors
                        fixes = glob.glob("fix_5d_tensors_*.safetensors")
                        if fixes:
                            logging.info("Applying Tensor Fix...")
                            fixed = os.path.join(out_dir, f"{name}-{q}-FIXED.gguf")
                            subprocess.run([sys.executable, "fix_5d_tensors.py", "--src", unfixed, "--dst", fixed, "--fix", fixes[0], "--overwrite"])
                            if os.path.exists(fixed):
                                os.rename(fixed, final_path)
                                os.remove(unfixed)
                            else:
                                os.rename(unfixed, final_path)
                        else:
                            os.rename(unfixed, final_path)
                        
                        if os.path.exists(final_path): generated_files.append(final_path)
            
            # Cleanup Intermediates
            if gguf_src and "CONVERT" in gguf_src and not keep_convert:
                if os.path.exists(gguf_src): os.remove(gguf_src)
            if dq and os.path.exists(dq) and not keep_dequant:
                os.remove(dq)

        # --- UPLOAD ---
        if do_upload:
            fp8s = [f for f in generated_files if "FP8" in f]
            ggufs = [f for f in generated_files if "FP8" not in f]
            
            # Dest folder per model (append name)
            d_f = f"{dest_folder_fp8}/{name}" if dest_folder_fp8 else name
            d_g = f"{dest_folder_gguf}/{name}" if dest_folder_gguf else name

            # If dest folder was explicitly root "/", we don't append name
            if dest_folder_fp8 == "/": d_f = ""
            if dest_folder_gguf == "/": d_g = ""

            if fp8s and repo_fp8:
                logging.info(f"Uploading FP8 to {repo_fp8} -> {d_f}")
                uploader.main(repo_id=repo_fp8, local_paths_args=fp8s, dest_folder=d_f)
            
            if ggufs and repo_gguf:
                logging.info(f"Uploading GGUF to {repo_gguf} -> {d_g}")
                uploader.main(token=token, repo_id=repo_gguf, local_paths_args=ggufs, dest_folder=d_g)

        # --- CLEANUP ---
        if cleanup_mode:
            logging.info("Cleaning up local generated files...")
            for f in generated_files:
                if os.path.exists(f): os.remove(f)

    print("\n--- All Tasks Complete ---")

if __name__ == "__main__":
    main()
