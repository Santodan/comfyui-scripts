# run_conversion_v5.py
import os
import subprocess
import sys
import importlib
import glob
import platform
import logging
import re
from datetime import datetime

try:
    import upload_to_hf_v4 as uploader
except ImportError:
    uploader = None

def setup_logging():
    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"conversion_log_{timestamp}.log")
    logger = logging.getLogger(); logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode='w'); file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'); file_handler.setFormatter(file_formatter); logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout); console_formatter = logging.Formatter('%(message)s'); console_handler.setFormatter(console_formatter); logger.addHandler(console_handler)
    return log_filename

required_packages = ['gguf', 'torch', 'safetensors', 'tqdm', 'huggingface-hub', 'prompt-toolkit']

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
    """Executes a shell command, logs its output in real-time, and checks for errors."""
    logging.info(f"\n{'='*20}\n[STEP] Running: {step_name}\n{' '.join(command)}\n{'='*20}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''): logging.info(line.strip())
        process.stdout.close()
        return_code = process.wait()
        if return_code == 0: logging.info(f"\n[SUCCESS] Step '{step_name}' completed successfully."); return True
        else: logging.error(f"\n[ERROR] Step '{step_name}' failed with return code {return_code}."); return False
    except FileNotFoundError: logging.error(f"\n[ERROR] Command not found: '{command[0]}'."); return False
    except Exception as e: logging.error(f"\n[ERROR] An unexpected error occurred in step '{step_name}': {e}"); return False

def get_user_input(prompt_text, validator, error_message):
    """A generic function to get and validate user input."""
    while True:
        sys.stdout.write(prompt_text)
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()
        if validator(user_input): return user_input
        else: logging.warning(error_message)

def get_and_validate_input_files():
    """Prompts the user for file paths and validates them until a valid input is given."""
    prompt_text = "\nEnter path(s) to .safetensors or .gguf file(s) (wildcards OK):\n> "
    if prompt_toolkit_available:
        print("\nEnter path(s) to .safetensors or .gguf file(s) (use Tab, spaces, wildcards):")
        session = PromptSession(completer=PathCompleter())
        prompt_func, prompt_text = (lambda p: session.prompt(p)), "> "
    else:
        prompt_func = lambda p: input(p)
    
    while True:
        user_input = prompt_func(prompt_text).strip()
        if not user_input: continue
        patterns, all_files, has_error = user_input.split(), [], False
        for pattern in patterns:
            matched_files = glob.glob(os.path.expanduser(pattern))
            if not matched_files: logging.error(f"\n❌ No files found matching '{pattern}'."); has_error = True; break
            all_files.extend(matched_files)
        if has_error: continue
        final_valid_files = []
        for file_path in sorted(list(set(all_files))):
            if not os.path.isfile(file_path): logging.error(f"\n❌ Path is not a file: '{file_path}'"); has_error = True; break
            if not file_path.lower().endswith(('.safetensors', '.gguf')): 
                logging.error(f"\n❌ File must be .safetensors or .gguf: '{file_path}'"); has_error = True; break
            final_valid_files.append(file_path)
        if has_error: continue
        return final_valid_files

def process_single_model(source_file, output_dir, selected_quant_types, files_to_keep):
    """Runs the full conversion and quantization process for a single model file."""
    logging.info(f"\n{'#'*80}\n### Starting Process for: {os.path.basename(source_file)}\n{'#'*80}")
    all_created_files, final_model_files = [], []
    
    if source_file.lower().endswith('.safetensors'):
        base_name = os.path.splitext(os.path.basename(source_file))[0]
        current_input_file = source_file
        dequant_script_path = "dequantize_fp8v2.py"
        if os.path.exists(dequant_script_path):
            dequantized_file = os.path.join(output_dir, f"{base_name}-dequantized.safetensors")
            cmd = [sys.executable, dequant_script_path, "--src", source_file, "--dst", dequantized_file, "--strip-fp8", "--dtype", "fp16"]
            if run_command(cmd, "Dequantize FP8") and os.path.exists(dequantized_file):
                current_input_file, all_created_files = dequantized_file, all_created_files + [dequantized_file]
            else: logging.warning("[WARNING] Dequantize script failed.")
        quantization_source_file = os.path.join(output_dir, f"{base_name}-CONVERT.gguf")
        if not run_command([sys.executable, "convert.py", "--src", current_input_file, "--dst", quantization_source_file], "Convert to GGUF"):
            logging.error(f"\n[FATAL] Conversion for {source_file} failed."); return []
        all_created_files.append(quantization_source_file)
    elif source_file.lower().endswith('.gguf'):
        logging.info("GGUF file detected. Skipping dequantize and convert steps.")
        quantization_source_file = source_file
        raw_basename = os.path.splitext(os.path.basename(source_file))[0]
        base_name = re.sub(r'-(f16|F16|CONVERT)$', '', raw_basename, flags=re.IGNORECASE)
    else:
        logging.error(f"Unsupported file type for: {source_file}"); return []

    if 'F16' in selected_quant_types or 'BF16' in selected_quant_types:
        final_model_files.append(quantization_source_file)
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
        if not run_command(["llama-quantize.exe", quantization_source_file, unfixed, quant_type], f"Quantize to {quant_type}"): continue
        all_created_files.append(unfixed)
        loop_output = unfixed
        if fix_tensor_file:
            fixed = os.path.join(output_dir, f"{base_name}-{quant_type}-FIXED.gguf")
            cmd = [sys.executable, "fix_5d_tensors.py", "--src", unfixed, "--dst", fixed, "--fix", fix_tensor_file, "--overwrite"]
            if run_command(cmd, f"Apply 5D Tensor Fix for {quant_type}"):
                loop_output, all_created_files = fixed, all_created_files + [fixed]
        final_model_files.append(loop_output)
    
    logging.info("\nDetermining which files to clean up...")
    files_to_delete = set()
    dequantized_path = os.path.join(output_dir, f"{base_name}-dequantized.safetensors")
    if dequantized_path in all_created_files: files_to_delete.add(dequantized_path)
    is_f16_kept = 'F16' in selected_quant_types or 'BF16' in selected_quant_types
    if 'converted' not in files_to_keep and not is_f16_kept and quantization_source_file.endswith("-CONVERT.gguf"):
        files_to_delete.add(quantization_source_file)
    if fix_tensor_file and 'fix_tensor' not in files_to_keep: files_to_delete.add(fix_tensor_file)
    for f in final_model_files:
        if f.endswith("-FIXED.gguf") and 'unfixed' not in files_to_keep:
            unfixed_sibling = f.replace("-FIXED.gguf", "-UnFixed.gguf")
            if unfixed_sibling in all_created_files: files_to_delete.add(unfixed_sibling)
    if files_to_delete:
        logging.info("Cleaning up intermediate files...")
        for f in sorted(list(files_to_delete)):
            if os.path.exists(f):
                try: os.remove(f); logging.info(f"  - Deleted {f}")
                except OSError as e: logging.error(f"  - Error deleting {f}: {e}")

    logging.info("\nRenaming final model files...")
    renamed_final_files, final_products = [], set(final_model_files) - files_to_delete
    for old_path in final_products:
        new_path, filename = old_path, os.path.basename(old_path)
        clean_path = old_path
        if filename.endswith(("-CONVERT.gguf", "-F16.gguf")):
            clean_path = os.path.join(output_dir, f"{base_name}-F16.gguf")
        elif filename.endswith("-FIXED.gguf") or filename.endswith("-UnFixed.gguf"):
            sibling = old_path.replace("-FIXED.gguf", "-UnFixed.gguf") if filename.endswith("-FIXED.gguf") else old_path.replace("-UnFixed.gguf", "-FIXED.gguf")
            if sibling not in final_products:
                quant_name = filename.split('-')[-2]
                clean_path = os.path.join(output_dir, f"{base_name}-{quant_name}.gguf")
            else: logging.info(f"  - Keeping original name for '{filename}' to avoid conflict.")
        if clean_path != old_path:
            try:
                if os.path.exists(clean_path): logging.warning(f"  - ⚠️  Warning: Target file '{os.path.basename(clean_path)}' already exists.")
                else: os.rename(old_path, clean_path); new_path = clean_path; logging.info(f"  - Renamed '{filename}' to '{os.path.basename(clean_path)}'")
            except OSError as e: logging.error(f"  - ❌ Error renaming file: {e}")
        renamed_final_files.append(new_path)
    
    logging.info(f"\n### Finished processing for: {os.path.basename(source_file)} ###")
    return renamed_final_files

def clean_model_name(filename):
    """Extracts a clean model name from a filename to use as a folder name."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Remove common suffixes to get the 'root' model name
    base = re.sub(r'-(f16|F16|BF16|CONVERT|UnFixed|FIXED)$', '', base, flags=re.IGNORECASE)
    return base.strip()

def main():
    """Main function to orchestrate the conversion process for multiple files."""
    log_file = setup_logging()
    logging.info(f"--- Model Conversion/Quantization Script ---")
    logging.info(f"Log file: {os.path.abspath(log_file)}")
    
    missing = []
    for package in required_packages:
        try: importlib.import_module(package.replace('-', '_'))
        except ImportError: missing.append(package)
    if missing: logging.error(f"Required libraries not installed: {', '.join(missing)}"); sys.exit(1)
    logging.info("All required libraries are installed.")

    try:
        source_files = get_and_validate_input_files()
        logging.info(f"Source files: {source_files}")
        if not source_files: logging.error("No valid input files. Exiting."); return
        logging.info("Models to process:" + "".join([f"\n  - {f}" for f in source_files]))
        
        output_dir_base_str = prompt("\nEnter BASE output directory (blank for source dir): ", completer=PathCompleter(only_directories=True)).strip() if prompt_toolkit_available else input("\nEnter BASE output directory (blank for source dir): ").strip()
        
        print("\nOutput Organization:")
        print("  1. Create separate folder per model (e.g. BaseDir/ModelName/...)")
        print("  2. Save all files in one folder (e.g. BaseDir/...)")
        org_choice = get_user_input("Select organization (1 or 2): ", lambda s: s in ['1', '2'], "Please enter 1 or 2.")
        group_by_model = (org_choice == '1')

        logging.info(f"Base Directory: '{output_dir_base_str}' | Group by model: {group_by_model}")

        logging.info("\nAvailable quants:" + "".join([f"\n  {i+1:2}. {q}" for i, q in enumerate(QUANTIZATION_OPTIONS)]))
        indices = get_user_input("Select quants (e.g., '9 18 20'): ", lambda s: all(c in '0123456789, ' for c in s.replace(",", " ")), "Invalid selection.")
        selected_quant_types = [QUANTIZATION_OPTIONS[int(i) - 1] for i in indices.replace(',', ' ').split()]
        logging.info(f"Quants selected: {selected_quant_types}")
        
        file_options = {'1': "converted", '2': "unfixed", '3': "fixed", '4': "fix_tensor"}
        logging.info("\nKeep intermediate files?" + "\n  1. ...-CONVERT.gguf" + "\n  2. ...-UnFixed.gguf" + "\n  3. ...-FIXED.gguf" + "\n  4. ...fix_tensor.safetensors" + "\n  5. All")
        keep_indices = get_user_input("Enter numbers to keep (e.g., '1 4'): ", lambda s: all(c in '12345, ' for c in s.replace(",", " ")), "Invalid selection.").split()
        files_to_keep = set(file_options.values()) if '5' in keep_indices else {file_options[k] for k in keep_indices}
        
        upload_enabled, hf_repo, hf_root_dest_folder, hf_token = False, "", "", None
        if uploader and input("\nUpload to Hugging Face Hub? (y/n): ").lower().strip() == 'y':
            logging.info("User chose to upload.")
            try:
                hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
                if hf_token: logging.info("Found HF_TOKEN.")
                login(token=hf_token, add_to_git_credential=False)
                username = whoami(token=hf_token)['name']; logging.info(f"Logged in as: {username}"); api = HfApi(token=hf_token)
                while not hf_repo:
                    logging.info("\nFetching repos..."); repos = sorted([r.id for r in api.list_models(author=username, token=hf_token)])
                    if repos: print("Select a repo:", *[f"\n  {i+1}. {r}" for i, r in enumerate(repos)])
                    print("\n  N. Create new repo")
                    choice = input("Enter number, 'N' to create, or 'Q' to quit upload: ").strip().upper()
                    if choice == 'Q': break
                    elif choice == 'N':
                        new_repo = input("Enter new repo name: ").strip()
                        if not new_repo: logging.warning("Repo name cannot be empty."); continue
                        if '/' not in new_repo: new_repo = f"{username}/{new_repo}"
                        is_private = input("Make private? [y/n]: ").strip().lower() == 'y'
                        try: hf_repo = create_repo(repo_id=new_repo, private=is_private, repo_type='model', token=hf_token).repo_id; logging.info(f"Created '{hf_repo}'")
                        except HfHubHTTPError as e: logging.error(f"Error creating repo: {e}")
                    else:
                        try: hf_repo = repos[int(choice) - 1]
                        except (ValueError, IndexError): logging.warning("Invalid selection.")
                if hf_repo: 
                    upload_enabled = True; logging.info(f"Repo selected: {hf_repo}")
                    hf_root_dest_folder = input(f"\nEnter destination root folder in '{hf_repo}' (blank for root): ").strip()
                    logging.info(f"Destination root: '{hf_root_dest_folder or 'Repo Root'}'")
            except Exception as e: logging.error(f"HF login/selection failed: {e}\nSkipping upload.")
        
        shutdown_when_done = input("\nShutdown when complete? (y/n): ").lower().strip() in ['y', 'yes']

        processed_batches = [] 

        for f in source_files:
            model_name = clean_model_name(f)
            base_dir = output_dir_base_str if output_dir_base_str else os.path.dirname(f)
            
            # Decision: Subfolder or Flat?
            if group_by_model:
                specific_out_dir = os.path.join(base_dir, model_name)
            else:
                specific_out_dir = base_dir
            
            os.makedirs(specific_out_dir, exist_ok=True)
            
            generated_files = process_single_model(f, specific_out_dir, selected_quant_types, files_to_keep)
            
            processed_batches.append({
                "name": model_name,
                "directory": specific_out_dir,
                "files": generated_files
            })
        
        logging.info("\n--- All models processed. ---")
        
        if upload_enabled and processed_batches:
            logging.info("\nStarting structured Hugging Face Upload...")
            for batch in processed_batches:
                if not batch["files"]: continue
                
                # Determine destination folder in HF
                if group_by_model:
                    # Upload to: RootDest/ModelName
                    if hf_root_dest_folder:
                        dest_subfolder = f"{hf_root_dest_folder}/{batch['name']}"
                    else:
                        dest_subfolder = batch["name"]
                else:
                    # Upload to: RootDest/ (All combined)
                    dest_subfolder = hf_root_dest_folder
                
                # Clean up double slashes just in case
                if dest_subfolder: dest_subfolder = dest_subfolder.replace("//", "/")
                
                logging.info(f"\n[UPLOAD] Uploading '{batch['name']}' to: {dest_subfolder if dest_subfolder else 'Repo Root'}")
                try:
                    uploader.main(token=hf_token, repo_id=hf_repo, local_paths_args=batch["files"], dest_folder=dest_subfolder, non_interactive=True)
                    logging.info(f"[SUCCESS] Upload for {batch['name']} complete.")
                except Exception as e: 
                    logging.error(f"[ERROR] Upload failed for {batch['name']}: {e}")
        
        if shutdown_when_done:
            logging.info("\nInitiating shutdown...")
            try:
                if platform.system() == "Windows": subprocess.run(["shutdown", "/s", "/t", "60"], check=True)
                else: subprocess.run(["sudo", "shutdown", "-h", "+1"], check=True)
            except Exception as e: logging.error(f"❌ Failed to initiate shutdown: {e}")
        else: logging.info("\nAll done!")

    except KeyboardInterrupt:
        logging.warning("\n\nScript interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        logging.exception("An unexpected top-level error occurred:")
        sys.exit(1)

if __name__ == "__main__":
    main()
