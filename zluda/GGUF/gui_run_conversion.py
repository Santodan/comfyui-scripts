import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, Menu
import os
import sys
import subprocess
import threading
import logging
import re
import glob
import json
from datetime import datetime
import math
import platform

# --- IMPORTS ---
try:
    import upload_to_hf_v4 as uploader
    UPLOADER_AVAILABLE = True
except ImportError:
    uploader = None
    UPLOADER_AVAILABLE = False

# --- CONFIGURATION ---
QUANTIZATION_OPTIONS = sorted([
    "F16", "BF16", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
    "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS",
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"
])

# --- DUAL OUTPUT CLASS ---
class DualOutput:
    """Captures standard 'print' to BOTH terminal and GUI."""
    def __init__(self, original_stream, text_widget):
        self.original_stream = original_stream
        self.text_widget = text_widget

    def write(self, message):
        self.original_stream.write(message)
        if message.strip() and not message.startswith('\r'): 
            def update():
                self.text_widget.configure(state='normal')
                self.text_widget.insert(tk.END, message)
                self.text_widget.see(tk.END)
                self.text_widget.configure(state='disabled')
            self.text_widget.after(0, update)

    def flush(self):
        self.original_stream.flush()

# --- UTILITIES ---
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        def append():
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, msg + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.configure(state='disabled')
        self.text_widget.after(0, append)

def get_quantize_command():
    system = platform.system()
    if system == "Windows":
        return "llama-quantize.exe"
    else:
        if os.path.exists("./llama-quantize"): return "./llama-quantize"
        return "llama-quantize"

def run_command(command, step_name=""):
    logging.info(f"--- [STEP] {step_name} ---\nCMD: {' '.join(command)}")
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace',
            bufsize=1, universal_newlines=True
        )
        for line in iter(process.stdout.readline, ''):
            logging.info(line.strip())
        process.stdout.close()
        return_code = process.wait()
        if return_code == 0:
            logging.info(f"✅ {step_name} Success.")
            return True
        else:
            logging.error(f"❌ {step_name} Failed (Code {return_code}).")
            return False
    except Exception as e:
        logging.error(f"❌ Error in {step_name}: {e}")
        return False

def clean_model_name(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r'-(f16|F16|BF16|CONVERT|UnFixed|FIXED)$', '', base, flags=re.IGNORECASE)
    return base.strip()

# --- MAIN APP CLASS ---
class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUF Conversion & Upload Manager v10")
        self.root.geometry("1100x950")
        
        self.source_files = []
        self.is_running = False
        self.quant_vars_gen = {}
        self.quant_vars_keep = {} 
        
        self.quant_cmd = get_quantize_command()
        logging.info(f"OS: {platform.system()} | Binary: {self.quant_cmd}")

        self._setup_menu()
        self._setup_ui()
        
        # Logging Setup
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        gui_handler = TextHandler(self.log_display)
        gui_handler.setFormatter(formatter)
        self.logger.addHandler(gui_handler)

        self.load_settings("last_run_settings.json", silent=True)

    def _setup_menu(self):
        menubar = Menu(self.root)
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Preset...", command=self.load_preset_dialog)
        file_menu.add_command(label="Save Preset...", command=self.save_preset_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_ui(self):
        # 0. Environment
        frame_env = tk.LabelFrame(self.root, text="0. Python Environment", padx=10, pady=5, fg="blue")
        frame_env.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_env, text="Interpreter:").pack(side="left")
        self.python_path_var = tk.StringVar(value=sys.executable)
        tk.Entry(frame_env, textvariable=self.python_path_var, width=60).pack(side="left", padx=5)
        tk.Button(frame_env, text="Browse...", command=self.browse_python).pack(side="left")
        tk.Button(frame_env, text="Restart", command=self.restart_in_venv, bg="#ffdddd").pack(side="left", padx=10)

        # 1. Files
        frame_files = tk.LabelFrame(self.root, text="1. Input Models (.safetensors / .gguf)", padx=10, pady=5)
        frame_files.pack(fill="x", padx=10, pady=5)
        self.file_listbox = tk.Listbox(frame_files, height=4, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side="left", fill="x", expand=True, padx=(0, 5))
        btn_frame = tk.Frame(frame_files)
        btn_frame.pack(side="right", fill="y")
        tk.Button(btn_frame, text="Add Files", command=self.add_files).pack(fill="x", pady=2)
        tk.Button(btn_frame, text="Clear", command=self.clear_files).pack(fill="x", pady=2)

        # 2. Output
        frame_out = tk.LabelFrame(self.root, text="2. Output Configuration", padx=10, pady=5)
        frame_out.pack(fill="x", padx=10, pady=5)
        tk.Label(frame_out, text="Base Output Dir:").grid(row=0, column=0, sticky="w")
        self.out_dir_var = tk.StringVar()
        tk.Entry(frame_out, textvariable=self.out_dir_var, width=50).grid(row=0, column=1, padx=5)
        tk.Button(frame_out, text="Browse", command=self.browse_output).grid(row=0, column=2)
        self.org_var = tk.StringVar(value="folder")
        tk.Radiobutton(frame_out, text="Create Folder per Model", variable=self.org_var, value="folder").grid(row=1, column=0, columnspan=3, sticky="w")
        tk.Radiobutton(frame_out, text="All in One Folder", variable=self.org_var, value="flat").grid(row=2, column=0, columnspan=3, sticky="w")

        # 3. Quants
        frame_quants = tk.LabelFrame(self.root, text="3. Quantization Selection", padx=10, pady=5)
        frame_quants.pack(fill="x", padx=10, pady=5)
        total_opts = len(QUANTIZATION_OPTIONS)
        chunk_size = math.ceil(total_opts / 3)
        chunks = [QUANTIZATION_OPTIONS[i:i + chunk_size] for i in range(0, total_opts, chunk_size)]

        def create_headers(parent, col_base):
            tk.Label(parent, text="Type", font=("Arial", 9, "bold")).grid(row=0, column=col_base, padx=5, sticky="w")
            tk.Label(parent, text="GEN", font=("Arial", 9, "bold"), fg="blue").grid(row=0, column=col_base+1, padx=2)
            tk.Label(parent, text="KEEP", font=("Arial", 9, "bold"), fg="green").grid(row=0, column=col_base+2, padx=2)
            if col_base < 8: tk.Label(parent, text=" | ", fg="#ccc").grid(row=0, column=col_base+3, rowspan=chunk_size+1)

        for col_idx, chunk in enumerate(chunks):
            base = col_idx * 4 
            create_headers(frame_quants, base)
            for i, q in enumerate(chunk):
                r = i + 1
                tk.Label(frame_quants, text=q).grid(row=r, column=base, sticky="w", padx=5)
                v_gen, v_keep = tk.BooleanVar(), tk.BooleanVar()
                self.quant_vars_gen[q], self.quant_vars_keep[q] = v_gen, v_keep
                def on_chg(vg=v_gen, vk=v_keep):
                    if vg.get(): vk.set(True)
                    else: vk.set(False)
                tk.Checkbutton(frame_quants, variable=v_gen, command=on_chg).grid(row=r, column=base+1)
                tk.Checkbutton(frame_quants, variable=v_keep).grid(row=r, column=base+2)

        # 4. Cleanup
        frame_clean = tk.LabelFrame(self.root, text="4. Intermediate Files to Keep", padx=10, pady=5)
        frame_clean.pack(fill="x", padx=10, pady=5)
        self.keep_vars = {"converted": tk.BooleanVar(), "unfixed": tk.BooleanVar(), "fixed": tk.BooleanVar(), "fix_tensor": tk.BooleanVar()}
        for k, v in self.keep_vars.items(): tk.Checkbutton(frame_clean, text=f"Keep {k}", variable=v).pack(side="left", padx=5)

        # 5. HF
        frame_hf = tk.LabelFrame(self.root, text="5. Hugging Face Upload", padx=10, pady=5)
        frame_hf.pack(fill="x", padx=10, pady=5)
        self.do_upload = tk.BooleanVar()
        chk = tk.Checkbutton(frame_hf, text="Enable Upload", variable=self.do_upload, command=self.toggle_hf_inputs)
        chk.grid(row=0, column=0, sticky="w")
        if not UPLOADER_AVAILABLE: chk.config(state="disabled", text="Upload Disabled (No Module)")
        
        tk.Label(frame_hf, text="Token:").grid(row=1, column=0, sticky="e")
        self.hf_token_var = tk.StringVar(value=os.getenv("HUGGING_FACE_HUB_TOKEN", ""))
        self.entry_token = tk.Entry(frame_hf, textvariable=self.hf_token_var, width=40, show="*")
        self.entry_token.grid(row=1, column=1, sticky="w")
        
        tk.Label(frame_hf, text="Repo ID:").grid(row=2, column=0, sticky="e")
        self.hf_repo_var = tk.StringVar()
        self.entry_repo = tk.Entry(frame_hf, textvariable=self.hf_repo_var, width=40)
        self.entry_repo.grid(row=2, column=1, sticky="w")
        
        tk.Label(frame_hf, text="Dest Folder:").grid(row=3, column=0, sticky="e")
        self.hf_dest_var = tk.StringVar()
        self.entry_dest = tk.Entry(frame_hf, textvariable=self.hf_dest_var, width=40)
        self.entry_dest.grid(row=3, column=1, sticky="w")
        self.toggle_hf_inputs()

        # 6. Run Actions
        frame_action = tk.Frame(self.root, pady=5)
        frame_action.pack(fill="both", expand=True, padx=10)
        
        # SHUTDOWN CHECKBOX
        self.shutdown_var = tk.BooleanVar()
        tk.Checkbutton(frame_action, text="Shutdown computer when complete", variable=self.shutdown_var, fg="red").pack(anchor="w", pady=2)

        self.btn_run = tk.Button(frame_action, text="START PROCESSING", bg="#ddffdd", font=("Arial", 10, "bold"), height=2, command=self.start_thread)
        self.btn_run.pack(fill="x")
        self.log_display = scrolledtext.ScrolledText(frame_action, state='disabled', height=15)
        self.log_display.pack(fill="both", expand=True, pady=5)

    def toggle_hf_inputs(self):
        state = "normal" if self.do_upload.get() else "disabled"
        self.entry_token.config(state=state)
        self.entry_repo.config(state=state)
        self.entry_dest.config(state=state)

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Model Files", "*.safetensors *.gguf")])
        for f in files:
            if f not in self.source_files:
                self.source_files.append(f)
                self.file_listbox.insert(tk.END, os.path.basename(f))

    def clear_files(self):
        self.source_files = []
        self.file_listbox.delete(0, tk.END)

    def browse_output(self):
        d = filedialog.askdirectory()
        if d: self.out_dir_var.set(d)

    def browse_python(self):
        f = filedialog.askopenfilename()
        if f: self.python_path_var.set(f)

    def restart_in_venv(self):
        target = self.python_path_var.get()
        if os.path.exists(target):
            self.save_settings("last_run_settings.json")
            subprocess.Popen([target] + sys.argv)
            self.root.destroy()

    def on_close(self):
        self.save_settings("last_run_settings.json")
        self.root.destroy()

    def save_preset_dialog(self):
        f = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if f: self.save_settings(f)

    def load_preset_dialog(self):
        f = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if f: self.load_settings(f)

    def save_settings(self, filename):
        data = {
            "python_path": self.python_path_var.get(),
            "out_dir": self.out_dir_var.get(),
            "org_mode": self.org_var.get(),
            "hf_enabled": self.do_upload.get(),
            "hf_token": self.hf_token_var.get(),
            "hf_repo": self.hf_repo_var.get(),
            "hf_dest": self.hf_dest_var.get(),
            "shutdown": self.shutdown_var.get(),  # Save Shutdown State
            "quants_gen": [k for k, v in self.quant_vars_gen.items() if v.get()],
            "quants_keep": [k for k, v in self.quant_vars_keep.items() if v.get()],
            "cleanup_opts": {k: v.get() for k, v in self.keep_vars.items()}
        }
        try:
            with open(filename, 'w') as f: json.dump(data, f, indent=4)
        except: pass

    def load_settings(self, filename, silent=False):
        if not os.path.exists(filename): return
        try:
            with open(filename, 'r') as f: data = json.load(f)
            if "python_path" in data: self.python_path_var.set(data["python_path"])
            if "out_dir" in data: self.out_dir_var.set(data["out_dir"])
            if "org_mode" in data: self.org_var.set(data["org_mode"])
            if "hf_enabled" in data: self.do_upload.set(data["hf_enabled"])
            if "hf_token" in data: self.hf_token_var.set(data["hf_token"])
            if "hf_repo" in data: self.hf_repo_var.set(data["hf_repo"])
            if "hf_dest" in data: self.hf_dest_var.set(data["hf_dest"])
            if "shutdown" in data: self.shutdown_var.set(data["shutdown"]) # Load Shutdown State
            for v in self.quant_vars_gen.values(): v.set(False)
            for v in self.quant_vars_keep.values(): v.set(False)
            for q in data.get("quants_gen", []): 
                if q in self.quant_vars_gen: self.quant_vars_gen[q].set(True)
            for q in data.get("quants_keep", []):
                if q in self.quant_vars_keep: self.quant_vars_keep[q].set(True)
            for k, v in data.get("cleanup_opts", {}).items():
                if k in self.keep_vars: self.keep_vars[k].set(v)
            self.toggle_hf_inputs()
        except: pass

    def start_thread(self):
        if self.is_running: return
        if not self.source_files: return messagebox.showerror("Error", "No files.")
        gen_list = [q for q, v in self.quant_vars_gen.items() if v.get()]
        keep_list = [q for q, v in self.quant_vars_keep.items() if v.get()]
        if not gen_list: return messagebox.showerror("Error", "Select GEN quants.")
        self.is_running = True
        self.btn_run.config(state="disabled", text="Running...")
        threading.Thread(target=self.run_logic, args=(gen_list, keep_list)).start()

    def run_logic(self, gen_list, keep_list):
        try:
            files_to_keep = {k for k, v in self.keep_vars.items() if v.get()}
            root = self.out_dir_var.get()
            group_by_model = (self.org_var.get() == "folder")
            
            batches = []
            logging.info("\n>>> PHASE 1: GENERATION <<<")
            for f in self.source_files:
                name = clean_model_name(f)
                base = root if root else os.path.dirname(f)
                out = os.path.join(base, name) if group_by_model else base
                if not os.path.exists(out): os.makedirs(out)
                logging.info(f"Processing: {name} -> {out}")
                generated = self.process_model(f, out, gen_list, files_to_keep)
                batches.append({"name": name, "files": generated})

            if self.do_upload.get() and UPLOADER_AVAILABLE:
                logging.info("\n>>> PHASE 2: UPLOAD <<<")
                from huggingface_hub import login
                login(token=self.hf_token_var.get(), add_to_git_credential=False)
                hf_root = self.hf_dest_var.get().strip()
                
                for b in batches:
                    if not b['files']: continue
                    dest = f"{hf_root}/{b['name']}" if group_by_model and hf_root else (b['name'] if group_by_model else hf_root)
                    if dest: dest = dest.replace("//", "/")
                    logging.info(f"Uploading {b['name']} to {dest}...")
                    
                    old_stdout = sys.stdout
                    sys.stdout = DualOutput(old_stdout, self.log_display)
                    try:
                        uploader.main(token=self.hf_token_var.get(), repo_id=self.hf_repo_var.get(), local_paths_args=b['files'], dest_folder=dest, non_interactive=True)
                    except Exception as e:
                        logging.error(f"Upload error: {e}")
                    finally:
                        sys.stdout = old_stdout

            logging.info("\n>>> PHASE 3: CLEANUP <<<")
            for b in batches:
                for path in b['files']:
                    if not os.path.exists(path): continue
                    fname = os.path.basename(path)
                    keep = False
                    if "F16" in fname or "BF16" in fname:
                        if "F16" in keep_list or "BF16" in keep_list or "converted" in files_to_keep: keep = True
                    else:
                        for q in keep_list:
                            if f"-{q}.gguf" in fname: keep = True; break
                    if not keep:
                        try: os.remove(path); logging.info(f"Deleted: {fname}")
                        except: pass
                    else: logging.info(f"Kept: {fname}")

            logging.info("DONE.")
            
            # --- SHUTDOWN LOGIC ---
            if self.shutdown_var.get():
                logging.info("⚠️ SHUTDOWN INITIATED (60s)...")
                if platform.system() == "Windows":
                    subprocess.run(["shutdown", "/s", "/t", "60"])
                else:
                    # Linux/Mac (Requires Sudo usually)
                    subprocess.run(["sudo", "shutdown", "-h", "+1"])
            else:
                messagebox.showinfo("Done", "Finished.")

        except Exception as e:
            logging.exception("Error")
            messagebox.showerror("Error", str(e))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.btn_run.config(state="normal", text="START PROCESSING"))

    def process_model(self, src, out, quants, keep_inter):
        created, final = [], []
        
        if src.lower().endswith('.safetensors'):
            base = os.path.splitext(os.path.basename(src))[0]
            curr = src
            dq = os.path.join(out, f"{base}-dequantized.safetensors")
            if os.path.exists("dequantize_fp8v2.py"):
                if run_command([sys.executable, "dequantize_fp8v2.py", "--src", src, "--dst", dq, "--strip-fp8", "--dtype", "fp16"], "Dequant"):
                    if os.path.exists(dq): curr, created = dq, created + [dq]
            
            gguf = os.path.join(out, f"{base}-CONVERT.gguf")
            if not run_command([sys.executable, "convert.py", "--src", curr, "--dst", gguf], "Convert"): return []
            created.append(gguf)
        elif src.lower().endswith('.gguf'):
            gguf = src
            base = re.sub(r'-(f16|F16|CONVERT)$', '', os.path.splitext(os.path.basename(src))[0], flags=re.IGNORECASE)
        else: return []

        if 'F16' in quants or 'BF16' in quants: final.append(gguf)

        fix_file = None
        fixes = glob.glob("fix_5d_tensors_*.safetensors")
        if fixes:
            fix_file = os.path.join(out, os.path.basename(fixes[0]))
            try: 
                import shutil; shutil.copy(fixes[0], fix_file); created.append(fix_file)
            except: pass

        for q in [x for x in quants if x not in ['F16','BF16']]:
            unfixed = os.path.join(out, f"{base}-{q}-UnFixed.gguf")
            if run_command([self.quant_cmd, gguf, unfixed, q], f"Quantize {q}"):
                created.append(unfixed)
                res = unfixed
                if fix_file:
                    fixed = os.path.join(out, f"{base}-{q}-FIXED.gguf")
                    if run_command([sys.executable, "fix_5d_tensors.py", "--src", unfixed, "--dst", fixed, "--fix", fix_file, "--overwrite"], "Fix"):
                        created.append(fixed); res = fixed
                final.append(res)

        for f in created:
            if f in final: continue
            k = False
            if "CONVERT" in f and "converted" in keep_inter: k = True
            if "UnFixed" in f and "unfixed" in keep_inter: k = True
            if "FIXED" in f and "fixed" in keep_inter: k = True
            if not k and os.path.exists(f): 
                try: os.remove(f)
                except: pass

        renamed = []
        for old in final:
            if not os.path.exists(old): continue
            fname = os.path.basename(old)
            new_name = fname
            if "CONVERT" in fname: new_name = f"{base}-F16.gguf"
            elif "-FIXED" in fname: new_name = f"{base}-{fname.split('-')[-2]}.gguf"
            elif "-UnFixed" in fname: new_name = f"{base}-{fname.split('-')[-2]}.gguf"
            new_path = os.path.join(out, new_name)
            if old != new_path:
                try: os.rename(old, new_path); renamed.append(new_path)
                except: renamed.append(old)
            else: renamed.append(old)
        return renamed

if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()