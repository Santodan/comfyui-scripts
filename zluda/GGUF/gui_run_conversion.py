import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk, Menu, Canvas, Toplevel
import os
import sys
import subprocess
import threading
import logging
import re
import glob
import json
import platform
import queue
import time
import shutil
from datetime import datetime
import math

# --- 0. AUTO-RESTART IN VENV ---
def check_and_restart_in_venv():
    possible_venvs = ["venv", ".venv", "env"]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_venv_path = None
    for venv_name in possible_venvs:
        full_path = os.path.join(script_dir, venv_name)
        if os.path.isdir(full_path):
            target_venv_path = full_path
            break
    if not target_venv_path: return
    if sys.platform == "win32":
        venv_python = os.path.join(target_venv_path, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(target_venv_path, "bin", "python")
    if not os.path.exists(venv_python): return
    current_exe = os.path.normpath(sys.executable).lower()
    target_exe = os.path.normpath(venv_python).lower()
    if current_exe != target_exe:
        print(f"[INFO] Auto-relaunching in venv: {venv_python}")
        try:
            subprocess.call([venv_python] + sys.argv)
            sys.exit()
        except Exception as e:
            print(f"[ERROR] Restart failed: {e}")

check_and_restart_in_venv()

# --- IMPORTS ---
try:
    import upload_to_hf as uploader
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

# --- CONFIGURATION GROUPS ---
QUANT_GROUPS = [
    ["F16", "BF16"],
    ["IQ2_XS", "IQ2_S"],
    ["IQ3_XXS", "IQ3_S", "IQ3_M"],
    ["IQ4_NL", "IQ4_XS"],
    ["Q2_K"],
    ["Q3_K_S", "Q3_K_M", "Q3_K_L"],
    ["Q4_0", "Q4_K_S", "Q4_K_M"],
    ["Q5_0", "Q5_K_S", "Q5_K_M"],
    ["Q6_K", "Q8_0"],
    ["FP8_E5M2", "FP8_E5M2 (All)"]
]
QUANTIZATION_OPTIONS = [item for sublist in QUANT_GROUPS for item in sublist]

# --- FP8 LOGIC ---
if TORCH_AVAILABLE:
    class FP8Quantizer:
        def __init__(self, quant_dtype: str = "float8_e5m2"):
            if not hasattr(torch, quant_dtype): raise ValueError(f"Unsupported: {quant_dtype}")
            self.quant_dtype = quant_dtype

        def quantize_weights(self, weight: torch.Tensor) -> torch.Tensor:
            if not weight.is_floating_point(): return weight
            target_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            weight_on_target = weight.to(target_device)
            max_val = torch.max(torch.abs(weight_on_target))
            if max_val == 0:
                target_torch_dtype = getattr(torch, self.quant_dtype)
                return torch.zeros_like(weight_on_target, dtype=target_torch_dtype)
            scale = max_val / 127.0 
            scale = torch.max(scale, torch.tensor(1e-12, device=target_device, dtype=weight_on_target.dtype))
            quantized = torch.round(weight_on_target / scale * 127.0) / 127.0 * scale
            return quantized.to(dtype=getattr(torch, self.quant_dtype))

        def apply_quantization_to_file(self, src_path, dst_path, unet_only=True, check_stop_func=None):
            if src_path.endswith(".safetensors"): state_dict = load_file(src_path)
            else: state_dict = torch.load(src_path, map_location="cpu")
            
            quantized_dict = {}
            total = len(state_dict)
            logging.info(f"[FP8] Loaded {total} tensors. Unet Only: {unet_only}")

            for i, (name, param) in enumerate(state_dict.items()):
                if check_stop_func and check_stop_func(): return False
                if unet_only and "model.diffusion_model" not in name: continue 
                if i % 100 == 0: logging.info(f"[FP8] Processing {i}/{total}...")
                
                if isinstance(param, torch.Tensor) and param.is_floating_point():
                    quantized_dict[name] = self.quantize_weights(param)
                else:
                    quantized_dict[name] = param
            
            if not quantized_dict: return False
            save_file(quantized_dict, dst_path)
            del state_dict; del quantized_dict
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return True
else:
    class FP8Quantizer:
        def __init__(self, *args, **kwargs): pass

# --- GUI UTILS ---
class DualOutput:
    def __init__(self, original_stream, text_widget):
        self.original_stream = original_stream
        self.text_widget = text_widget
    def write(self, message):
        self.original_stream.write(message)
        if message.strip() and not message.startswith('\r'): 
            def update():
                try:
                    self.text_widget.configure(state='normal')
                    if '\r' in message:
                        parts = message.split('\r')
                        content = parts[-1]
                        if content:
                            self.text_widget.delete("end-1c linestart", "end")
                            self.text_widget.insert("end", content)
                    else:
                        self.text_widget.insert("end", message)
                    self.text_widget.see("end")
                    self.text_widget.configure(state='disabled')
                except: pass
            self.text_widget.after(0, update)
    def flush(self): self.original_stream.flush()

class ProgressPopup(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Job Progress Status")
        self.geometry("700x500")
        self.protocol("WM_DELETE_WINDOW", self.hide_window)
        
        self.canvas = Canvas(self, bg="#f0f0f0")
        self.scroll_y = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scroll_x = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.inner = tk.Frame(self.canvas, bg="#f0f0f0")
        
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
        
        self.scroll_y.pack(side="right", fill="y")
        self.scroll_x.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.cells = {} 

    def hide_window(self):
        self.withdraw()

    def setup_grid(self, models, steps):
        for w in self.inner.winfo_children(): w.destroy()
        self.cells = {}
        tk.Label(self.inner, text="Model Name", font=("Arial", 9, "bold"), bg="#ddd", width=30, anchor="w").grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        for i, step in enumerate(steps):
            tk.Label(self.inner, text=step, font=("Arial", 8, "bold"), bg="#ddd", width=12).grid(row=0, column=i+1, sticky="nsew", padx=1, pady=1)
        for r, model in enumerate(models):
            row = r + 1
            tk.Label(self.inner, text=model[:40], anchor="w", bg="white").grid(row=row, column=0, sticky="nsew", padx=1, pady=1)
            for c, step in enumerate(steps):
                lbl = tk.Label(self.inner, text="...", bg="#cccccc", width=12)
                lbl.grid(row=row, column=c+1, sticky="nsew", padx=1, pady=1)
                self.cells[(model, step)] = lbl

    def update_status(self, model, step, status):
        if (model, step) not in self.cells: return
        lbl = self.cells[(model, step)]
        if status == "RUNNING": lbl.config(bg="#ffff99", text="Running")
        elif status == "DONE": lbl.config(bg="#99ff99", text="Done")
        elif status == "ERROR": lbl.config(bg="#ff9999", text="Error")
        elif status == "SKIP": lbl.config(bg="#eeeeee", text="-")
        elif status == "CANCEL": lbl.config(bg="#ffcc00", text="Cancel")
        else: lbl.config(bg="#cccccc", text="...")

# --- MAIN APP ---
class ConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GGUF & FP8 Manager")
        self.root.geometry("1600x950")
        
        # --- SETTINGS FILE LOCATION (Absolute Path) ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.settings_file = os.path.join(script_dir, "last_run_settings.json")
        
        self.msg_queue = queue.Queue()
        self.source_files = []
        self.custom_file_data = {} 
        
        self.is_running = False
        self.quant_vars_gen = {}
        self.quant_vars_up = {}
        self.quant_vars_keep = {}
        self.current_process = None
        self.stop_requested = False
        self.progress_window = None
        
        self.quant_cmd = self.get_quantize_command()

        # --- BIND EXIT EVENT ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._setup_ui()
        self._setup_logging()
        
        self.root.after(100, self.process_queue)
        
        # Load using absolute path
        self.load_settings(self.settings_file, silent=True)

    def get_quantize_command(self):
        system = platform.system()
        if system == "Windows": return "llama-quantize.exe"
        else: return "./llama-quantize" if os.path.exists("./llama-quantize") else "llama-quantize"

    def _setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers(): self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fh = logging.FileHandler(f"logs/log_{ts}.log", encoding='utf-8')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        class TextHandler(logging.Handler):
            def __init__(self, widget):
                super().__init__()
                self.widget = widget
            def emit(self, record):
                msg = self.format(record)
                def app():
                    self.widget.configure(state='normal')
                    self.widget.insert(tk.END, msg + '\n')
                    self.widget.see(tk.END)
                    self.widget.configure(state='disabled')
                self.widget.after(0, app)
        self.logger.addHandler(TextHandler(self.log_display))

    def _setup_ui(self):
        main_pane = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashwidth=5)
        main_pane.pack(fill="both", expand=True)
        config_frame = tk.Frame(main_pane)
        canvas = Canvas(config_frame)
        scrollbar = ttk.Scrollbar(config_frame, orient="vertical", command=canvas.yview)
        self.content_frame = tk.Frame(canvas)
        self.content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig("inner", width=e.width))
        def on_mousewheel(event):
            if platform.system() == 'Windows': canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            else: canvas.yview_scroll(int(-1*event.delta), "units")
        canvas.bind_all("<MouseWheel>", on_mousewheel)
        canvas.create_window((0, 0), window=self.content_frame, anchor="nw", tags="inner")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        main_pane.add(config_frame, height=750)
        log_container = tk.Frame(main_pane)
        main_pane.add(log_container, minsize=150)
        self.log_display = scrolledtext.ScrolledText(log_container, height=10)
        self.log_display.pack(side="left", fill="both", expand=True)
        btn_clear = tk.Button(log_container, text="CLEAR\nLOGS", bg="#eee", command=self.clear_logs)
        btn_clear.pack(side="right", fill="y", padx=2)

        # 1. Environment
        f_env = tk.LabelFrame(self.content_frame, text="1. Environment", padx=5, pady=5, fg="blue")
        f_env.pack(fill="x", padx=5, pady=5)
        self.python_path_var = tk.StringVar(value=os.path.normpath(sys.executable))
        tk.Entry(f_env, textvariable=self.python_path_var).pack(side="left", fill="x", expand=True, padx=(0,5))
        tk.Button(f_env, text="Browse...", command=self.browse_python).pack(side="left", padx=2)
        tk.Button(f_env, text="Restart", command=self.restart).pack(side="left", padx=2)

        # 2. Output
        f_mode = tk.LabelFrame(self.content_frame, text="2. Local Output Configuration", padx=5, pady=5)
        f_mode.pack(fill="x", padx=5, pady=5)
        self.out_mode_var = tk.StringVar(value="folder") 
        tk.Label(f_mode, text="Output Strategy:", fg="blue").pack(anchor="w")
        modes_frame = tk.Frame(f_mode)
        modes_frame.pack(fill="x")
        tk.Radiobutton(modes_frame, text="Folder per Model", variable=self.out_mode_var, value="folder", command=self.refresh_file_list_ui).pack(side="left")
        tk.Radiobutton(modes_frame, text="All in One Folder (Flat)", variable=self.out_mode_var, value="flat", command=self.refresh_file_list_ui).pack(side="left")
        tk.Radiobutton(modes_frame, text="Custom Output Path (Per File)", variable=self.out_mode_var, value="custom", command=self.refresh_file_list_ui).pack(side="left")
        self.global_out_frame = tk.Frame(self.content_frame)
        tk.Label(self.global_out_frame, text="Base Output Dir:").pack(side="left")
        self.out_dir_var = tk.StringVar()
        tk.Entry(self.global_out_frame, textvariable=self.out_dir_var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(self.global_out_frame, text="Browse", command=self.browse_out).pack(side="left")
        self.global_out_frame.pack(fill="x", padx=10, pady=5, after=f_mode) 

        # 3. Files
        self.f_files_container = tk.LabelFrame(self.content_frame, text="3. Input Files & Routing Table", padx=5, pady=5)
        self.f_files_container.pack(fill="x", padx=5, pady=5)
        btn_box = tk.Frame(self.f_files_container)
        btn_box.pack(fill="x", pady=2)
        tk.Button(btn_box, text="Add Files...", command=self.add_files, bg="#e6f2ff").pack(side="left", fill="x", expand=True)
        
        # --- NEW BUTTON HERE ---
        tk.Button(btn_box, text="Remove Selected", command=self.remove_selected_files, bg="#fff0f0").pack(side="left", padx=5)
        # -----------------------

        tk.Button(btn_box, text="Clear List", command=self.clear_files).pack(side="left", padx=5)
        self.simple_list_frame = tk.Frame(self.f_files_container)
        self.file_listbox = tk.Listbox(self.simple_list_frame, height=6, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side="left", fill="x", expand=True)
        self.simple_list_frame.pack(fill="x", expand=True) 
        self.local_custom_frame = tk.Frame(self.f_files_container)

        # 4. Quants
        f_quant = tk.LabelFrame(self.content_frame, text="4. Quantization", padx=5, pady=5)
        f_quant.pack(fill="x", padx=5, pady=5)
        if not TORCH_AVAILABLE: tk.Label(f_quant, text="⚠️ Torch missing. FP8 disabled.", fg="red").grid(row=0, column=0, columnspan=10)
        for col_idx, group in enumerate(QUANT_GROUPS):
            base_col = col_idx * 5  
            tk.Label(f_quant, text="Type", font="Arial 8 bold").grid(row=1, column=base_col, sticky="w")
            tk.Label(f_quant, text="G", font="Arial 8 bold", fg="blue", width=2).grid(row=1, column=base_col+1)
            tk.Label(f_quant, text="U", font="Arial 8 bold", fg="purple", width=2).grid(row=1, column=base_col+2)
            tk.Label(f_quant, text="K", font="Arial 8 bold", fg="green", width=2).grid(row=1, column=base_col+3)
            tk.Label(f_quant, text="|", fg="#ccc").grid(row=1, column=base_col+4, rowspan=10, sticky="ns")
            for i, q in enumerate(group):
                row = i + 2
                tk.Label(f_quant, text=q).grid(row=row, column=base_col, sticky="w")
                vg, vu, vk = tk.BooleanVar(), tk.BooleanVar(), tk.BooleanVar()
                self.quant_vars_gen[q] = vg
                self.quant_vars_up[q] = vu
                self.quant_vars_keep[q] = vk
                state = "normal"
                if "FP8" in q and not TORCH_AVAILABLE: state = "disabled"
                def sync(g=vg, u=vu, k=vk): 
                    if g.get(): 
                        u.set(True)
                        k.set(True)
                    else: 
                        u.set(False)
                        k.set(False)
                tk.Checkbutton(f_quant, variable=vg, command=sync, state=state).grid(row=row, column=base_col+1)
                tk.Checkbutton(f_quant, variable=vu, state=state).grid(row=row, column=base_col+2)
                tk.Checkbutton(f_quant, variable=vk, state=state).grid(row=row, column=base_col+3)

        # 5. Upload
        f_sets = tk.LabelFrame(self.content_frame, text="5. Global Settings & Upload", padx=5, pady=5)
        f_sets.pack(fill="x", padx=5, pady=5)
        self.do_upload = tk.BooleanVar()
        tk.Checkbutton(f_sets, text="Enable Upload", variable=self.do_upload).grid(row=0, column=0, sticky="w")
        tk.Label(f_sets, text="Token:").grid(row=0, column=1, sticky="e")
        self.hf_token = tk.StringVar(value=os.getenv("HUGGING_FACE_HUB_TOKEN",""))
        tk.Entry(f_sets, textvariable=self.hf_token, show="*").grid(row=0, column=2, columnspan=3, sticky="ew", padx=5)
        ttk.Separator(f_sets, orient="horizontal").grid(row=1, column=0, columnspan=6, sticky="ew", pady=5)
        tk.Label(f_sets, text="Upload Strategy:", fg="purple").grid(row=2, column=0, sticky="e")
        self.upload_mode_var = tk.StringVar(value="global")
        tk.Radiobutton(f_sets, text="Use Global Repos (Below)", variable=self.upload_mode_var, value="global", command=self.refresh_upload_ui).grid(row=2, column=1, columnspan=1, sticky="w")
        tk.Radiobutton(f_sets, text="Custom Repos per File (Table)", variable=self.upload_mode_var, value="custom", command=self.refresh_upload_ui).grid(row=2, column=2, columnspan=2, sticky="w")
        
        self.global_upload_frame = tk.Frame(f_sets)
        self.global_upload_frame.grid(row=3, column=0, columnspan=5, sticky="ew")
        tk.Label(self.global_upload_frame, text="GGUF Repo:", fg="blue").grid(row=0, column=0, sticky="e")
        self.hf_repo_gguf = tk.StringVar()
        tk.Entry(self.global_upload_frame, textvariable=self.hf_repo_gguf).grid(row=0, column=1, sticky="ew", padx=5)
        tk.Label(self.global_upload_frame, text="FP8 Repo:", fg="green").grid(row=0, column=2, sticky="e")
        self.hf_repo_fp8 = tk.StringVar()
        tk.Entry(self.global_upload_frame, textvariable=self.hf_repo_fp8).grid(row=0, column=3, sticky="ew", padx=5)
        tk.Label(self.global_upload_frame, text="GGUF Folder:").grid(row=1, column=0, sticky="e")
        self.hf_dest_gguf = tk.StringVar()
        tk.Entry(self.global_upload_frame, textvariable=self.hf_dest_gguf).grid(row=1, column=1, sticky="ew", padx=5)
        tk.Label(self.global_upload_frame, text="FP8 Folder:").grid(row=1, column=2, sticky="e")
        self.hf_dest_fp8 = tk.StringVar()
        tk.Entry(self.global_upload_frame, textvariable=self.hf_dest_fp8).grid(row=1, column=3, sticky="ew", padx=5)
        self.global_upload_frame.columnconfigure(1, weight=1)
        self.global_upload_frame.columnconfigure(3, weight=1)
        self.custom_upload_frame = tk.Frame(f_sets)
        
        self.footer_frame = tk.Frame(f_sets)
        self.footer_frame.grid(row=5, column=0, columnspan=5, sticky="ew", pady=5)
        ttk.Separator(self.footer_frame, orient="horizontal").pack(fill="x", pady=5)
        f_c = tk.Frame(self.footer_frame)
        f_c.pack()
        tk.Label(f_c, text="Cleanup Strategy:").pack(side="left")
        self.cleanup_mode = tk.StringVar(value="per_model")
        tk.Radiobutton(f_c, text="After Each Model", variable=self.cleanup_mode, value="per_model").pack(side="left")
        tk.Radiobutton(f_c, text="After All Complete", variable=self.cleanup_mode, value="all_end").pack(side="left")
        
        self.keep_dequant_var = tk.BooleanVar(value=False)
        self.keep_convert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(f_c, text="Keep Dequant Source", variable=self.keep_dequant_var, fg="orange").pack(side="left", padx=10)
        tk.Checkbutton(f_c, text="Keep GGUF Source (CONVERT)", variable=self.keep_convert_var, fg="orange").pack(side="left")
        f_sets.columnconfigure(2, weight=1)

        # 6. Actions
        f_act = tk.Frame(self.content_frame)
        f_act.pack(fill="x", padx=5, pady=10)
        self.shutdown_var = tk.BooleanVar()
        tk.Checkbutton(f_act, text="Shutdown when done", variable=self.shutdown_var, fg="red").pack(side="left")
        tk.Button(f_act, text="SHOW STATUS", command=self.show_progress_popup).pack(side="left", padx=20)
        tk.Button(f_act, text="CANCEL", bg="#ffcccc", command=self.cancel_processing).pack(side="right")
        self.btn_run = tk.Button(f_act, text="START PROCESSING", bg="#ddffdd", height=2, command=self.start_thread)
        self.btn_run.pack(side="right", fill="x", expand=True, padx=5)

    def clear_logs(self):
        self.log_display.configure(state='normal')
        self.log_display.delete("1.0", tk.END)
        self.log_display.configure(state='disabled')

    def refresh_file_list_ui(self):
        out_mode = self.out_mode_var.get()
        if out_mode == "custom": self.global_out_frame.pack_forget()
        else: self.global_out_frame.pack(fill="x", padx=10, pady=5, before=self.f_files_container)
        if out_mode == "custom":
            self.simple_list_frame.pack_forget()
            self.build_local_table()
            self.local_custom_frame.pack(fill="x", expand=True)
        else:
            self.local_custom_frame.pack_forget()
            self.simple_list_frame.pack(fill="x", expand=True)
            self.file_listbox.delete(0, tk.END)
            for f in self.source_files: self.file_listbox.insert(tk.END, os.path.basename(f))

    def refresh_upload_ui(self):
        mode = self.upload_mode_var.get()
        if mode == "custom":
            self.global_upload_frame.grid_remove()
            self.build_upload_table()
            self.custom_upload_frame.grid(row=3, column=0, columnspan=5, sticky="ew")
        else:
            self.custom_upload_frame.grid_remove()
            self.global_upload_frame.grid()

    def build_local_table(self):
        for w in self.local_custom_frame.winfo_children(): w.destroy()
        
        # Added "Remove" to headers
        headers = ["File Name", "Output Path", "Remove"]
        for c, h in enumerate(headers): 
            tk.Label(self.local_custom_frame, text=h, font="Arial 8 bold", bg="#ddd").grid(row=0, column=c, sticky="ew", padx=1)
        
        self.local_custom_frame.columnconfigure(1, weight=1)
        
        for i, fpath in enumerate(self.source_files):
            row = i + 1
            self._ensure_file_data(fpath)
            dat = self.custom_file_data[fpath]
            
            # File Name
            tk.Label(self.local_custom_frame, text=os.path.basename(fpath), width=50, anchor="w").grid(row=row, column=0, sticky="w", padx=2)
            
            # Path Entry + Browse Button
            fr = tk.Frame(self.local_custom_frame)
            fr.grid(row=row, column=1, sticky="ew", padx=2)
            tk.Entry(fr, textvariable=dat["out"]).pack(side="left", fill="x", expand=True)
            tk.Button(fr, text="..", width=3, command=lambda d=dat: self.browse_file_out(d)).pack(side="right")
            
            # NEW: Remove Button (Red X)
            # We use lambda f=fpath: to capture the specific file for this row
            btn_rem = tk.Button(self.local_custom_frame, text="X", bg="#ffcccc", fg="red", font="Arial 8 bold",
                                command=lambda f=fpath: self.remove_single_file(f))
            btn_rem.grid(row=row, column=2, padx=2, pady=1)

    def build_upload_table(self):
        for w in self.custom_upload_frame.winfo_children(): w.destroy()
        
        # Added "Remove" to headers
        headers = ["File", "GGUF Repo", "GGUF Folder", "FP8 Repo", "FP8 Folder", "Remove"]
        for c, h in enumerate(headers): 
            tk.Label(self.custom_upload_frame, text=h, font="Arial 8 bold", bg="#ddd").grid(row=0, column=c, sticky="ew", padx=1)
        
        # Configure column weights
        for c in range(5): self.custom_upload_frame.columnconfigure(c, weight=1)
        
        for i, fpath in enumerate(self.source_files):
            row = i + 1
            self._ensure_file_data(fpath)
            dat = self.custom_file_data[fpath]
            
            # Fields
            tk.Label(self.custom_upload_frame, text=os.path.basename(fpath), width=30, anchor="w").grid(row=row, column=0, sticky="w", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["gguf_r"]).grid(row=row, column=1, sticky="ew", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["gguf_d"]).grid(row=row, column=2, sticky="ew", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["fp8_r"]).grid(row=row, column=3, sticky="ew", padx=1)
            tk.Entry(self.custom_upload_frame, textvariable=dat["fp8_d"]).grid(row=row, column=4, sticky="ew", padx=1)
            
            # NEW: Remove Button (Red X)
            btn_rem = tk.Button(self.custom_upload_frame, text="X", bg="#ffcccc", fg="red", font="Arial 8 bold",
                                command=lambda f=fpath: self.remove_single_file(f))
            btn_rem.grid(row=row, column=5, padx=2, pady=1)

    def _ensure_file_data(self, fpath):
        if fpath not in self.custom_file_data:
            self.custom_file_data[fpath] = {
                "out": tk.StringVar(value=os.path.dirname(fpath)),
                "gguf_r": tk.StringVar(value=self.hf_repo_gguf.get()),
                "gguf_d": tk.StringVar(value=self.hf_dest_gguf.get()),
                "fp8_r": tk.StringVar(value=self.hf_repo_fp8.get()),
                "fp8_d": tk.StringVar(value=self.hf_dest_fp8.get())
            }

    def browse_file_out(self, dat_dict):
        d = filedialog.askdirectory()
        if d: dat_dict["out"].set(d)

    def add_files(self):
        fs = filedialog.askopenfilenames()
        for f in fs: 
            norm = os.path.normpath(f)
            if norm not in self.source_files:
                self.source_files.append(norm)
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom": self.refresh_upload_ui()

    def remove_single_file(self, fpath):
        if fpath in self.source_files:
            self.source_files.remove(fpath)
            
        # Clean up data dictionary (optional but good for memory)
        if fpath in self.custom_file_data:
            del self.custom_file_data[fpath]
            
        # Refresh both views so the file disappears everywhere
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom":
            self.refresh_upload_ui()

    def remove_selected_files(self):
        # Get selected indices from listbox (returns a tuple of integers)
        selection = self.file_listbox.curselection()
        if not selection: return

        # Delete from source_files. 
        # Must delete from highest index to lowest to avoid shifting issues.
        for index in reversed(selection):
            if index < len(self.source_files):
                del self.source_files[index]
        
        # Refresh the displays
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom":
            self.refresh_upload_ui()

    def clear_files(self):
        self.source_files = []
        self.custom_file_data = {}
        self.refresh_file_list_ui()
        if self.upload_mode_var.get() == "custom": self.refresh_upload_ui()

    def browse_out(self): self.out_dir_var.set(filedialog.askdirectory())
    def browse_python(self): 
        # Detect OS to determine what files to look for
        if platform.system() == "Windows":
            ftypes = [("Python Executable", "python.exe"), ("All Files", "*.*")]
        else:
            # On Linux/Mac, python binaries often have no extension
            ftypes = [("Python Executable", "python3"), ("Python Executable", "python"), ("All Files", "*")]
            
        f = filedialog.askopenfilename(filetypes=ftypes)
        if f: self.python_path_var.set(os.path.normpath(f))
    
    def restart(self):
        target = self.python_path_var.get()
        if not os.path.exists(target): return messagebox.showerror("Error", "Python not found")
        self.save_settings(self.settings_file)
        subprocess.Popen([target] + sys.argv)
        self.root.destroy()

    def on_close(self):
        self.save_settings(self.settings_file)
        self.root.destroy()

    def show_progress_popup(self):
        if self.progress_window is None or not self.progress_window.winfo_exists():
            self.progress_window = ProgressPopup(self.root)
        self.progress_window.deiconify()
        self.progress_window.lift()

    def cancel_processing(self):
        if not self.is_running: return
        if messagebox.askyesno("Cancel", "Stop processing?"):
            self.stop_requested = True
            logging.warning("STOP REQUESTED")
            if self.current_process:
                try: self.current_process.kill()
                except: pass

    def process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                if msg[0] == "UPDATE_GRID":
                    if self.progress_window is None or not self.progress_window.winfo_exists():
                        self.show_progress_popup()
                    self.progress_window.update_status(msg[1], msg[2], msg[3])
        except queue.Empty: pass
        self.root.after(100, self.process_queue)

    def start_thread(self):
        if self.is_running: return
        if not self.source_files: return messagebox.showerror("Error", "No files")
        
        gen = [q for q, v in self.quant_vars_gen.items() if v.get()]
        up_only = [q for q, v in self.quant_vars_up.items() if v.get()]
        if not gen and not up_only: return messagebox.showerror("Error", "Select at least one Generate or Upload option.")
        
        self.stop_requested = False
        steps = []
        SORT_ORDER = ["IQ2_XS", "IQ2_S", "Q2_K", "IQ3_XXS", "IQ3_S", "IQ3_M", "Q3_K_S", "Q3_K_M", "Q3_K_L",
                      "IQ4_NL", "IQ4_XS", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
                      "BF16", "F16", "FP8_E4M3FN", "FP8_E4M3FN (All)", "FP8_E5M2", "FP8_E5M2 (All)"]
        
        active_quants = list(set(gen + up_only))
        active_quants.sort(key=lambda x: SORT_ORDER.index(x) if x in SORT_ORDER else 999)

        fp8_variants = ["FP8_E5M2", "FP8_E5M2 (All)", "FP8_E4M3FN", "FP8_E4M3FN (All)"]
        gguf_gen_needed = [q for q in gen if q not in fp8_variants]
        if gguf_gen_needed: steps.append("GGUF Prep")
        
        for q in active_quants: steps.append(q)
        if self.do_upload.get(): steps.append("Upload")
        steps.append("Cleanup")

        self.show_progress_popup()
        model_names = [os.path.basename(f) for f in self.source_files]
        self.progress_window.setup_grid(model_names, steps)

        self.is_running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self.run_main_logic, args=(gen, up_only)).start()

    def run_main_logic(self, gen_list, up_list):
        try:
            strategy = self.cleanup_mode.get()
            keep_list = [q for q, v in self.quant_vars_keep.items() if v.get()]
            keep_dequant = self.keep_dequant_var.get()
            keep_convert = self.keep_convert_var.get()
            out_mode = self.out_mode_var.get()
            up_mode = self.upload_mode_var.get()
            
            if self.do_upload.get() and UPLOADER_AVAILABLE:
                from huggingface_hub import login
                login(token=self.hf_token.get(), add_to_git_credential=False)

            batch_results = []

            for f in self.source_files:
                if self.stop_requested: break
                
                # --- FIX: CLEANUP BEFORE EVERY MODEL ---
                # We must delete the fix file before starting a new model, 
                # otherwise convert.py crashes on the second model.
                fix_file = "fix_5d_tensors_wan.safetensors"
                if os.path.exists(fix_file):
                    try:
                        os.remove(fix_file)
                        logging.info(f"Loop Cleanup: Removed old '{fix_file}'")
                    except Exception as e:
                        logging.warning(f"Could not remove fix file: {e}")
                # ---------------------------------------

                model_base = os.path.basename(f)
                name = re.sub(r'-(f16|F16|BF16|CONVERT|UnFixed|FIXED)$', '', os.path.splitext(model_base)[0], flags=re.IGNORECASE)
                
                if out_mode == "custom":
                    dat = self.custom_file_data.get(f, {})
                    out_dir = dat["out"].get() if "out" in dat else os.path.dirname(f)
                else:
                    base = self.out_dir_var.get() if self.out_dir_var.get() else os.path.dirname(f)
                    out_dir = os.path.join(base, name) if out_mode == "folder" else base
                
                os.makedirs(out_dir, exist_ok=True)
                generated_files = []

                # --- FP8 Logic ---
                fp8_targets = ["FP8_E5M2", "FP8_E5M2 (All)", "FP8_E4M3FN", "FP8_E4M3FN (All)"]
                for q in fp8_targets:
                    if q in gen_list or q in up_list:
                        if self.stop_requested: break
                        self.msg_queue.put(("UPDATE_GRID", model_base, q, "RUNNING"))
                        
                        suffix = "_All" if "All" in q else ""
                        base_q_name = q.split(" ")[0]
                        expected_path = os.path.join(out_dir, f"{name}-{base_q_name}{suffix}.safetensors")
                        
                        if q in gen_list:
                            try:
                                if TORCH_AVAILABLE:
                                    dtype_str = "float8_e5m2" if "E5M2" in q else "float8_e4m3fn"
                                    qzer = FP8Quantizer(dtype_str)
                                    ok = qzer.apply_quantization_to_file(f, expected_path, unet_only=("All" not in q), check_stop_func=lambda: self.stop_requested)
                                    if ok: 
                                        generated_files.append(expected_path)
                                        self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                                    else: self.msg_queue.put(("UPDATE_GRID", model_base, q, "CANCEL"))
                                else: self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                            except Exception as e:
                                logging.error(f"FP8 Err: {e}")
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                        
                        elif q in up_list:
                            if os.path.exists(expected_path):
                                generated_files.append(expected_path)
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                            else:
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "SKIP"))

                # --- GGUF Logic ---
                # Deduplicate tasks
                raw_combined = gen_list + up_list
                unique_tasks = list(set(raw_combined))
                
                all_gguf_active = [q for q in unique_tasks if "FP8" not in q]
                
                if all_gguf_active:
                    gguf_gen_needed = [q for q in gen_list if "FP8" not in q]
                    gguf_src = None
                    
                    if gguf_gen_needed:
                        if self.stop_requested: break
                        self.msg_queue.put(("UPDATE_GRID", model_base, "GGUF Prep", "RUNNING"))
                        
                        if f.lower().endswith(".safetensors"):
                            curr = f
                            dq = os.path.join(out_dir, f"{name}-dequant.safetensors")
                            if os.path.exists("dequantize_fp8v2.py"):
                                # Ensure -u is passed
                                self.run_cmd([sys.executable, "-u", "dequantize_fp8v2.py", "--src", f, "--dst", dq, "--strip-fp8", "--dtype", "fp16"])
                                if os.path.exists(dq): 
                                    curr = dq; generated_files.append(dq)
                            
                            conv = os.path.join(out_dir, f"{name}-CONVERT.gguf")
                            # Ensure -u is passed
                            self.run_cmd([sys.executable, "-u", "convert.py", "--src", curr, "--dst", conv])
                            if os.path.exists(conv): gguf_src = conv; generated_files.append(conv)
                        elif f.lower().endswith(".gguf"):
                            gguf_src = f

                        if gguf_src: self.msg_queue.put(("UPDATE_GRID", model_base, "GGUF Prep", "DONE"))
                        else: self.msg_queue.put(("UPDATE_GRID", model_base, "GGUF Prep", "ERROR"))
                    
                    for q in all_gguf_active:
                        if self.stop_requested: break
                        self.msg_queue.put(("UPDATE_GRID", model_base, q, "RUNNING"))
                        expected_path = os.path.join(out_dir, f"{name}-{q}.gguf")

                        if q in gen_list:
                            if not gguf_src: 
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "SKIP"))
                                continue

                            if q in ["F16", "BF16"]:
                                try:
                                    shutil.copy(gguf_src, expected_path)
                                    generated_files.append(expected_path)
                                    self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                                except: self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                                continue

                            unfixed = os.path.join(out_dir, f"{name}-{q}-UnFixed.gguf")
                            if self.run_cmd([self.quant_cmd, gguf_src, unfixed, q]):
                                final = unfixed
                                fixes = glob.glob("fix_5d_tensors_*.safetensors")
                                if fixes:
                                    fixed = os.path.join(out_dir, f"{name}-{q}-FIXED.gguf")
                                    # Ensure -u is passed
                                    self.run_cmd([sys.executable, "-u", "fix_5d_tensors.py", "--src", unfixed, "--dst", fixed, "--fix", fixes[0], "--overwrite"])
                                    if os.path.exists(fixed): final = fixed
                                
                                try: os.rename(final, expected_path); generated_files.append(expected_path)
                                except: generated_files.append(final)
                                
                                if os.path.exists(unfixed) and os.path.abspath(unfixed) != os.path.abspath(expected_path):
                                    try: os.remove(unfixed)
                                    except: pass

                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                            else:
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "ERROR"))
                        
                        elif q in up_list:
                            if os.path.exists(expected_path):
                                generated_files.append(expected_path)
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "DONE"))
                            else:
                                self.msg_queue.put(("UPDATE_GRID", model_base, q, "SKIP"))

                # Dedupe generated files list
                generated_files = list(set(generated_files))

                res_obj = { "name": name, "files": generated_files, "model_display": model_base, "src_path": f }
                batch_results.append(res_obj)

                if strategy == "per_model" and not self.stop_requested:
                    self.handle_upload_cleanup(res_obj, keep_list, up_list, up_mode, out_mode, keep_dequant, keep_convert)

            if strategy == "all_end" and not self.stop_requested:
                logging.info("Batch Cleanup...")
                for item in batch_results:
                    if self.stop_requested: break
                    self.handle_upload_cleanup(item, keep_list, up_list, up_mode, out_mode, keep_dequant, keep_convert)

            if self.shutdown_var.get() and not self.stop_requested:
                if platform.system() == "Windows": subprocess.run(["shutdown", "/s", "/t", "60"])
                else: subprocess.run(["sudo", "shutdown", "-h", "+1"])
            
            if not self.stop_requested: messagebox.showinfo("Done", "Finished")

        except Exception as e:
            logging.exception("Error")
            messagebox.showerror("Error", str(e))
        finally:
            self.is_running = False
            self.btn_run.config(state="normal")

    def _check_file_match_quant(self, fname, q):
        if "FP8" in q:
            base_q = q.split(" ")[0] 
            is_all_q = "(All)" in q
            if is_all_q: return (base_q in fname and "_All" in fname)
            else: return (base_q in fname and "_All" not in fname)
        if q in ["F16", "BF16"]: return f"-{q}.gguf" in fname
        return f"-{q}.gguf" in fname

    def handle_upload_cleanup(self, item, keep_list, up_list, up_mode, out_mode, keep_dequant, keep_convert):
        if self.stop_requested: return
        name = item['name']
        files = item['files']
        disp = item['model_display']
        src = item['src_path']
        
        r_gguf = self.hf_repo_gguf.get()
        d_gguf = self.hf_dest_gguf.get()
        r_fp8 = self.hf_repo_fp8.get()
        d_fp8 = self.hf_dest_fp8.get()

        if up_mode == "custom":
            dat = self.custom_file_data.get(src, {})
            if "gguf_r" in dat and dat["gguf_r"].get(): r_gguf = dat["gguf_r"].get()
            if "gguf_d" in dat and dat["gguf_d"].get(): d_gguf = dat["gguf_d"].get()
            if "fp8_r" in dat and dat["fp8_r"].get(): r_fp8 = dat["fp8_r"].get()
            if "fp8_d" in dat and dat["fp8_d"].get(): d_fp8 = dat["fp8_d"].get()

        if out_mode == "folder" and up_mode == "global":
            d_gguf = f"{d_gguf}/{name}" if d_gguf else name
            d_fp8 = f"{d_fp8}/{name}" if d_fp8 else name

        if self.do_upload.get() and UPLOADER_AVAILABLE:
            self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "RUNNING"))
            
            files_to_upload = []
            for f in files:
                if f.endswith("-CONVERT.gguf") or f.endswith("-UnFixed.gguf") or f.endswith("-dequant.safetensors"): continue
                fname = os.path.basename(f)
                should_upload = False
                for q in up_list:
                    if self._check_file_match_quant(fname, q):
                        should_upload = True
                        break
                if should_upload: files_to_upload.append(f)

            # FIX 2: Deduplicate the upload list.
            # If the same file was added multiple times, it crashes the uploader.
            files_to_upload = list(set(files_to_upload))

            fp8s = [f for f in files_to_upload if "FP8" in f]
            ggufs = [f for f in files_to_upload if "FP8" not in f]
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = DualOutput(old_stdout, self.log_display)
            sys.stderr = DualOutput(old_stderr, self.log_display)
            
            try:
                # Log what we are trying to upload
                if fp8s: logging.info(f"Uploading FP8: {len(fp8s)} files to {r_fp8}")
                if ggufs: logging.info(f"Uploading GGUF: {len(ggufs)} files to {r_gguf}")

                if fp8s and r_fp8: uploader.main(token=self.hf_token.get(), repo_id=r_fp8, local_paths_args=fp8s, dest_folder=d_fp8, non_interactive=True)
                if ggufs and r_gguf: uploader.main(token=self.hf_token.get(), repo_id=r_gguf, local_paths_args=ggufs, dest_folder=d_gguf, non_interactive=True)
                self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "DONE"))
            except Exception as e:
                logging.error(f"Up Err: {e}")
                self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "ERROR"))
            finally: 
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        else:
            self.msg_queue.put(("UPDATE_GRID", disp, "Upload", "SKIP"))

        self.msg_queue.put(("UPDATE_GRID", disp, "Cleanup", "RUNNING"))
        for p in files:
            if not os.path.exists(p): continue
            fname = os.path.basename(p)
            should_keep = False
            
            # Keep logic
            for q in keep_list:
                if self._check_file_match_quant(fname, q):
                    should_keep = True
                    break
            
            if keep_dequant and "-dequant.safetensors" in fname: should_keep = True
            if keep_convert and "-CONVERT.gguf" in fname: should_keep = True

            if not should_keep: 
                try: os.remove(p); logging.info(f"Deleted {fname}")
                except: pass
        self.msg_queue.put(("UPDATE_GRID", disp, "Cleanup", "DONE"))

    def run_cmd(self, cmd, step=""):
        logging.info(f"CMD: {' '.join(cmd)}")
        try:
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for l in iter(self.current_process.stdout.readline, ''):
                logging.info(l.strip())
                if self.stop_requested: 
                    self.current_process.kill(); return False
            self.current_process.stdout.close()
            return (self.current_process.wait() == 0)
        except: return False

    def save_settings(self, f):
        d = {
            "python": self.python_path_var.get(),
            "out": self.out_dir_var.get(),
            "out_mode": self.out_mode_var.get(),
            "up_mode": self.upload_mode_var.get(),
            "token": self.hf_token.get(),
            "r_gguf": self.hf_repo_gguf.get(), "d_gguf": self.hf_dest_gguf.get(),
            "r_fp8": self.hf_repo_fp8.get(), "d_fp8": self.hf_dest_fp8.get(),
            "clean": self.cleanup_mode.get(),
            "shut": self.shutdown_var.get(),
            "q_gen": [k for k,v in self.quant_vars_gen.items() if v.get()],
            "q_up": [k for k,v in self.quant_vars_up.items() if v.get()],
            "q_keep": [k for k,v in self.quant_vars_keep.items() if v.get()],
            "k_dequant": self.keep_dequant_var.get(),
            "k_convert": self.keep_convert_var.get()
        }
        try: json.dump(d, open(f, 'w'), indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def load_settings(self, f, silent=False):
        if not os.path.exists(f): return
        try:
            d = json.load(open(f))
            if "python" in d: self.python_path_var.set(d["python"])
            if "out" in d: self.out_dir_var.set(d["out"])
            if "token" in d: self.hf_token.set(d["token"])
            if "r_gguf" in d: self.hf_repo_gguf.set(d["r_gguf"])
            if "d_gguf" in d: self.hf_dest_gguf.set(d["d_gguf"])
            if "r_fp8" in d: self.hf_repo_fp8.set(d["r_fp8"])
            if "d_fp8" in d: self.hf_dest_fp8.set(d["d_fp8"])
            if "out_mode" in d: self.out_mode_var.set(d["out_mode"])
            if "up_mode" in d: self.upload_mode_var.set(d["up_mode"])
            if "clean" in d: self.cleanup_mode.set(d["clean"])
            if "shut" in d: self.shutdown_var.set(d["shut"])
            if "k_dequant" in d: self.keep_dequant_var.set(d["k_dequant"])
            if "k_convert" in d: self.keep_convert_var.set(d["k_convert"])
            
            for v in self.quant_vars_gen.values(): v.set(False)
            for v in self.quant_vars_up.values(): v.set(False)
            for v in self.quant_vars_keep.values(): v.set(False)
            
            for q in d.get("q_gen", []): 
                if q in self.quant_vars_gen: self.quant_vars_gen[q].set(True)
            for q in d.get("q_up", []): 
                if q in self.quant_vars_up: self.quant_vars_up[q].set(True)
            for q in d.get("q_keep", []): 
                if q in self.quant_vars_keep: self.quant_vars_keep[q].set(True)
            
            self.refresh_file_list_ui()
            self.refresh_upload_ui()
        except: pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ConverterApp(root)
    root.mainloop()

