# run_conversion_final.py
import os
import subprocess
import sys
import importlib
import glob
import platform

try:
    import upload_to_hf_v4 as uploader
except ImportError:
    uploader = None

required_packages = ['gguf', 'torch', 'safetensors', 'tqdm', 'huggingface-hub', 'prompt-toolkit']
missing_packages = []

print("Checking for required libraries...")
for package in required_packages:
    try:
        import_name = package.replace('-', '_')
        importlib.import_module(import_name)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print("\n[ERROR] The following required libraries are not installed:", ", ".join(missing_packages))
    print("Please install them to continue by running this command:")
    print(f"pip install {' '.join(missing_packages)}")
    sys.exit(1)
else:
    print("All required libraries are installed.")

from huggingface_hub import HfApi, login, whoami, create_repo
from huggingface_hub.errors import HfHubHTTPError
try:
    from prompt_toolkit import PromptSession, prompt
    from prompt_toolkit.completion import PathCompleter
    prompt_toolkit_available = True
except ImportError:
    prompt_toolkit_available = False

QUANTIZATION_OPTIONS = sorted([
    "F16", "BF16", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"
])

def run_command(command, step_name=""):
    """Executes a shell command, prints its output, and checks for errors."""
    print(f"\n{'='*20}\n[STEP] Running: {step_name}\n{' '.join(command)}\n{'='*20}")
    try:
        subprocess.run(command, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"\n[SUCCESS] Step '{step_name}' completed successfully.")
        return True
    except FileNotFoundError:
        print(f"\n[ERROR] Command not found: '{command[0]}'. Please ensure it's in your PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Step '{step_name}' failed with return code {e.returncode}.")
        return False

def get_user_input(prompt_text, validator, error_message):
    """A generic function to get and validate user input."""
    while True:
        user_input = input(prompt_text).strip()
        if validator(user_input):
            return user_input
        else:
            print(error_message)

def get_and_validate_input_files():
    """Prompts the user for file paths and validates them until a valid input is given."""
    prompt_func = input
    if prompt_toolkit_available:
        print("\nEnter path(s) to .safetensors file(s) (use Tab to autocomplete; separate paths with spaces):")
        session = PromptSession(completer=PathCompleter())
        prompt_func = lambda p: session.prompt(p)
    else:
        print("\nEnter path(s) to .safetensors file(s) (space-separated, wildcards like *.safetensors are OK):")
    while True:
        user_input = prompt_func("> ").strip()
        if not user_input: continue
        patterns, all_files, has_error = user_input.split(), [], False
        for pattern in patterns:
            matched_files = glob.glob(os.path.expanduser(pattern))
            if not matched_files:
                print(f"\n❌ ERROR: No files found matching '{pattern}'."); has_error = True; break
            all_files.extend(matched_files)
        if has_error: continue
        final_valid_files = []
        for file_path in sorted(list(set(all_files))):
            if not os.path.isfile(file_path):
                print(f"\n❌ ERROR: Path is not a file: '{file_path}'"); has_error = True; break
            if not file_path.lower().endswith('.safetensors'):
                print(f"\n❌ ERROR: File is not a .safetensors file: '{file_path}'"); has_error = True; break
            final_valid_files.append(file_path)
        if has_error: continue
        return final_valid_files

def process_single_model(source_file, output_dir, selected_quant_types, files_to_keep):
    """Runs the full conversion and quantization process for a single model file."""
    print(f"\n{'#'*80}\n### Starting Process for: {os.path.basename(source_file)}\n{'#'*80}")
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    all_created_files, final_model_files = [], []
    current_input_file = source_file
    dequant_script_path = "dequantize_fp8v2.py"
    if os.path.exists(dequant_script_path):
        dequantized_file = os.path.join(output_dir, f"{base_name}-dequantized.safetensors")
        
        # --- THIS IS THE CORRECTED COMMAND ---
        # It now correctly uses the named --src and --dst arguments
        cmd_dequant = [
            sys.executable, dequant_script_path,
            "--src", source_file,
            "--dst", dequantized_file,
            "--strip-fp8",
            "--dtype", "fp16"  # Using fp16 as the safe default for GGUF conversion
        ]
        
        if run_command(cmd_dequant, "Dequantize FP8") and os.path.exists(dequantized_file):
            current_input_file, all_created_files = dequantized_file, all_created_files + [dequantized_file]
        else: print("[WARNING] Dequantize script failed.")

    converted_file = os.path.join(output_dir, f"{base_name}-CONVERT.gguf")
    if not run_command([sys.executable, "convert.py", "--src", current_input_file, "--dst", converted_file], "Convert to GGUF"):
        print(f"\n[FATAL] Conversion for {source_file} failed."); return []
    all_created_files.append(converted_file)
    
    if 'F16' in selected_quant_types or 'BF16' in selected_quant_types:
        final_model_files.append(converted_file)
    actual_quant_types = [q for q in selected_quant_types if q not in ["F16", "BF16"]]
    fix_tensor_file = None
    possible_fix_files = glob.glob("fix_5d_tensors_*.safetensors")
    if possible_fix_files:
        fix_tensor_file = os.path.join(output_dir, os.path.basename(possible_fix_files[0]))
        if os.path.exists(fix_tensor_file): os.remove(fix_tensor_file)
        os.rename(possible_fix_files[0], fix_tensor_file)
        all_created_files.append(fix_tensor_file)
    for quant_type in actual_quant_types:
        unfixed = os.path.join(output_dir, f"{base_name}-{quant_type}-UnFixed.gguf")
        if not run_command(["llama-quantize.exe", converted_file, unfixed, quant_type], f"Quantize to {quant_type}"): continue
        all_created_files.append(unfixed)
        loop_output = unfixed
        if fix_tensor_file:
            fixed = os.path.join(output_dir, f"{base_name}-{quant_type}-FIXED.gguf")
            cmd = [sys.executable, "fix_5d_tensors.py", "--src", unfixed, "--dst", fixed, "--fix", fix_tensor_file, "--overwrite"]
            if run_command(cmd, f"Apply 5D Tensor Fix for {quant_type}"):
                loop_output, all_created_files = fixed, all_created_files + [fixed]
        final_model_files.append(loop_output)
    
    user_kept_final_files = []
    for f in final_model_files:
        if (f.endswith('-FIXED.gguf') and 'fixed' in files_to_keep) or \
           (f.endswith('-UnFixed.gguf') and 'unfixed' in files_to_keep) or \
           (f.endswith('-CONVERT.gguf') and ('F16' in selected_quant_types or 'BF16' in selected_quant_types)):
            user_kept_final_files.append(f)
    files_to_delete = set(all_created_files) - set(user_kept_final_files)
    if 'converted' not in files_to_keep and converted_file not in user_kept_final_files: files_to_delete.add(converted_file)
    if fix_tensor_file and 'fix_tensor' not in files_to_keep: files_to_delete.add(fix_tensor_file)
    if files_to_delete:
        print("\nCleaning up intermediate files...")
        for f in sorted(list(files_to_delete)):
            if os.path.exists(f):
                try: os.remove(f); print(f"  - Deleted {f}")
                except OSError as e: print(f"  - Error deleting {f}: {e}")

    print("\nRenaming final model files...")
    renamed_final_files, kept_files_set = [], set(user_kept_final_files)
    for old_path in user_kept_final_files:
        new_path, filename = old_path, os.path.basename(old_path)
        clean_path = old_path
        if filename.endswith("-CONVERT.gguf"):
            clean_path = os.path.join(output_dir, f"{base_name}-F16.gguf")
        elif filename.endswith("-FIXED.gguf") or filename.endswith("-UnFixed.gguf"):
            sibling = old_path.replace("-FIXED.gguf", "-UnFixed.gguf") if filename.endswith("-FIXED.gguf") else old_path.replace("-UnFixed.gguf", "-FIXED.gguf")
            if sibling not in kept_files_set:
                quant_name = filename.split('-')[-2]
                clean_path = os.path.join(output_dir, f"{base_name}-{quant_name}.gguf")
            else: print(f"  - Keeping original name for '{filename}' to avoid conflict.")
        if clean_path != old_path:
            try:
                if os.path.exists(clean_path): print(f"  - ⚠️  Warning: Target file '{os.path.basename(clean_path)}' already exists.")
                else: os.rename(old_path, clean_path); new_path = clean_path; print(f"  - Renamed '{filename}' to '{os.path.basename(clean_path)}'")
            except OSError as e: print(f"  - ❌ Error renaming file: {e}")
        renamed_final_files.append(new_path)
    
    print(f"\n### Finished processing for: {os.path.basename(source_file)} ###")
    return renamed_final_files

def main():
    """Main function to orchestrate the conversion process for multiple files."""
    print("\n--- Model Conversion and Quantization Script ---")
    
    source_files = get_and_validate_input_files()
    if not source_files: print("[ERROR] No valid input files provided. Exiting."); return
    print("\n✅ The following models will be processed:", *[f"\n  - {f}" for f in source_files])
    
    output_dir_str = prompt("\nEnter output directory (blank for source dir): ", completer=PathCompleter(only_directories=True)).strip() if prompt_toolkit_available else input("\nEnter output directory (blank for source dir): ").strip()
    output_dir = output_dir_str or None
    if output_dir: os.makedirs(output_dir, exist_ok=True); print(f"All output will be saved in: {os.path.abspath(output_dir)}")
    else: print("Files will be saved alongside originals.")

    print("\nAvailable quantization types:", *[f"\n  {i+1:2}. {q}" for i, q in enumerate(QUANTIZATION_OPTIONS)])
    indices = get_user_input("Select quantization types (e.g., '9 18 20'): ", lambda s: all(c in '0123456789, ' for c in s.replace(",", " ")), "Invalid selection.")
    selected_quant_types = [QUANTIZATION_OPTIONS[int(i) - 1] for i in indices.replace(',', ' ').split()]
    
    file_options = {'1': "converted", '2': "unfixed", '3': "fixed", '4': "fix_tensor"}
    print("\nWhich types of files to keep?", "\n  1. ...-CONVERT.gguf", "\n  2. ...-UnFixed.gguf", "\n  3. ...-FIXED.gguf", "\n  4. ...fix_tensor.safetensors", "\n  5. All of the above")
    keep_indices = get_user_input("Enter numbers for file types to keep (e.g., '3 4'): ", lambda s: all(c in '12345, ' for c in s.replace(",", " ")), "Invalid selection.").split()
    files_to_keep = set(file_options.values()) if '5' in keep_indices else {file_options[k] for k in keep_indices}
    
    upload_enabled, hf_repo, hf_dest_folder, hf_token = False, "", "", None
    if uploader and input("\nUpload to Hugging Face Hub? (y/n): ").lower().strip() == 'y':
        try:
            hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if hf_token: print("✅ Found HUGGING_FACE_HUB_TOKEN. Logging in automatically.")
            else: print("HUGGING_FACE_HUB_TOKEN not found. Manual login required.")
            login(token=hf_token, add_to_git_credential=False)
            username = whoami(token=hf_token)['name']; print(f"✅ Logged in as: {username}"); api = HfApi(token=hf_token)
            while not hf_repo:
                print("\nFetching repos..."); repos = sorted([r.id for r in api.list_models(author=username, token=hf_token)])
                if repos: print("Select a repo:", *[f"\n  {i+1}. {r}" for i, r in enumerate(repos)])
                print("\n  N. Create new repo")
                choice = input("Enter number, 'N' to create, or 'Q' to quit upload: ").strip().upper()
                if choice == 'Q': break
                elif choice == 'N':
                    new_repo = input("Enter new repo name: ").strip()
                    if not new_repo: print("❌ Repo name cannot be empty."); continue
                    if '/' not in new_repo: new_repo = f"{username}/{new_repo}"
                    is_private = input("Make private? [y/n]: ").strip().lower() == 'y'
                    try: hf_repo = create_repo(repo_id=new_repo, private=is_private, repo_type='model', token=hf_token).repo_id; print(f"✅ Created '{hf_repo}'")
                    except HfHubHTTPError as e: print(f"❌ Error creating repo: {e}")
                else:
                    try: hf_repo = repos[int(choice) - 1]
                    except (ValueError, IndexError): print("❌ Invalid selection.")
            if hf_repo: upload_enabled = True; hf_dest_folder = input(f"\nEnter destination folder in '{hf_repo}' (blank for root): ").strip()
        except Exception as e: print(f"❌ HF login/selection failed: {e}\nSkipping upload.")
    
    shutdown_when_done = input("\nShutdown when complete? (y/n): ").lower().strip() in ['y', 'yes']
    if shutdown_when_done: print("✅ OK, shutdown scheduled.")
    else: print("✅ OK, no shutdown scheduled.")

    all_final_files = []
    for f in source_files:
        out_dir = output_dir or os.path.dirname(f)
        all_final_files.extend(process_single_model(f, out_dir, selected_quant_types, files_to_keep))
    
    print("\n--- All models processed. ---")
    if upload_enabled and all_final_files:
        print("\nUploading to Hugging Face Hub...")
        try:
            uploader.main(token=hf_token, repo_id=hf_repo, local_paths_args=all_final_files, dest_folder=hf_dest_folder, non_interactive=True)
            print("\n[SUCCESS] Upload complete.")
        except Exception as e: print(f"\n[ERROR] Upload failed: {e}")
    
    if shutdown_when_done:
        print("\nInitiating shutdown...")
        try:
            if platform.system() == "Windows": subprocess.run(["shutdown", "/s", "/t", "60"], check=True)
            else: subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
        except Exception as e: print(f"❌ Failed to initiate shutdown: {e}")
    else: print("\nAll done!")

if __name__ == "__main__":
    main()