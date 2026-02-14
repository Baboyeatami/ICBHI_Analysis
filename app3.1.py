import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn as nn
import torchvision.models as models
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import os
import csv
from datetime import datetime
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# --- CONFIGURATION ---
CLASSES = ['Crackle', 'Normal', 'Wheeze']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CLINICAL-STYLE GUIDANCE COPY ---
ADVICE = {
    "Normal": (
        "Professional Advice (Normal)\n"
        "Your recording is most consistent with normal, healthy breath sounds.\n\n"
        "Lungs that are functioning normally create a smooth, soft sound you can hear "
        "when you breathe in and out. They are called vesicular lung sounds. Vesicular "
        "lung sounds usually mean nothing is blocking your airways, and they are fully "
        "open (not narrowed or swollen).\n\n"
        "If you are feeling well and have no new symptoms, routine care and healthy "
        "habits are typically all that is needed. If you notice new or worsening symptoms, "
        "consult a healthcare professional."
    ),
    "Wheeze": (
        "Professional Advice (Wheeze)\n"
        "Wheezing is the shrill whistle or coarse rattle you hear when your airway is "
        "partially blocked. It might be blocked because of an allergic reaction, a cold, "
        "bronchitis or allergies. Wheezing is also a symptom of asthma, pneumonia, heart "
        "failure and more. It could go away on its own, or it could be a sign of a "
        "serious condition.\n\n"
        "Common Symptoms\n"
        "- Shortness of breath or rapid breathing\n"
        "- Chest tightness or a feeling of constriction\n"
        "- Coughing\n\n"
        "When to Seek Emergency Care\n"
        "- Severe difficulty breathing or gasping for air\n"
        "- Bluish tinge around lips, face, or nails\n"
        "- Confusion, dizziness, or altered mental state\n"
        "- Rapidly worsening shortness of breath\n"
        "- Swelling of the face, lips, or tongue\n"
        "- Wheezing that begins suddenly after an insect sting, new medication, or new food\n\n"
        "Other Associated Symptoms\n"
        "- Asthma (airway inflammation and spasms)\n"
        "- Allergies or allergic reactions\n"
        "- Bronchitis or respiratory infections\n"
        "- COPD or emphysema\n"
        "- GERD (acid reflux)\n"
        "- Heart failure\n"
        "- Inhaling a foreign object"
    ),
    "Crackle": (
        "Professional Advice (Crackle)\n"
        "Rales, or crackles, are discontinuous, interrupted, or explosive lung sounds. "
        "They may sound like pulling velcro open. The sounds can be short and high-pitched, "
        "or they may last a bit longer and be lower-pitched. Your doctor is more likely "
        "to hear crackles when you are breathing in, but they may happen when you breathe "
        "out, too. Rales happen when your airway snaps open as you breathe in.\n\n"
        "Common Symptoms\n"
        "- Shortness of breath or difficulty in breathing\n"
        "- Cough\n"
        "- Chest discomfort\n"
        "- Rapid or shallow breathing\n"
        "- Fatigue or weakness\n"
        "- Fever or chills\n"
        "- Wheezing\n"
        "- Swelling in the legs or ankles\n\n"
        "When to Seek Emergency Care\n"
        "- Severe or rapidly worsening shortness of breath\n"
        "- Chest pain or pressure\n"
        "- Coughing up blood or pink, frothy sputum (may indicate fluid in the lungs)\n"
        "- Bluish or gray lips, face, or fingertips (cyanosis)\n"
        "- Confusion, dizziness, or fainting\n"
        "- Very high fever with chills\n\n"
        "Other Associated Symptoms\n"
        "- Pneumonia\n"
        "- Congestive heart failure / pulmonary edema\n"
        "- COPD or bronchitis\n"
        "- Interstitial lung disease"
    ),
}

ADVICE_DISCLAIMER = (
    "Disclaimer: This guidance is informational only and does not replace a medical "
    "evaluation. If you are concerned about symptoms, seek care from a qualified "
    "healthcare professional."
)

LOG_COLUMNS = [
    "Time Stamp",
    "Filename",
    "Resnet18 Classification",
    "Resnet18 Confidence",
    "Resnet34 Classification",
    "Resnet34 Confidence",
    "Efficient Classification",
    "Efficient Confidence",
    "Best Model",
    "Best Classification",
    "Best Confidence",
]

# --- 1. THE UNIVERSAL MODEL CLASS ---
class SOTADualModel(nn.Module):
    def __init__(self, num_classes=3, architecture='resnet18'):
        super(SOTADualModel, self).__init__()
        self.architecture = architecture.lower()
        
        # --- BRANCH 1: CNN (The Eye) ---
        if self.architecture == 'resnet18':
            self.cnn = models.resnet18(pretrained=False)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.fc = nn.Identity()
            cnn_out_size = 512
            
        elif self.architecture == 'resnet34':
            self.cnn = models.resnet34(pretrained=False)
            self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.cnn.fc = nn.Identity()
            cnn_out_size = 512
            
        elif self.architecture == 'efficientnet_b0':
            self.cnn = models.efficientnet_b0(pretrained=False)
            self.cnn.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            self.cnn.classifier = nn.Identity()
            cnn_out_size = 1280
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # --- BRANCH 2: LSTM (The Ear) ---
        self.lstm = nn.LSTM(input_size=13, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        lstm_out_size = 256 

        # --- FUSION LAYER ---
        total_input = cnn_out_size + lstm_out_size
        
        self.fusion = nn.Sequential(
            nn.Linear(total_input, 512 if self.architecture == 'efficientnet_b0' else 256),
            nn.BatchNorm1d(512 if self.architecture == 'efficientnet_b0' else 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512 if self.architecture == 'efficientnet_b0' else 256, num_classes)
        )

    def forward(self, x_cnn, x_lstm):
        target_size = (224, 224)
        x_cnn = nn.functional.interpolate(x_cnn, size=target_size, mode='bilinear', align_corners=False)
        features_cnn = self.cnn(x_cnn)
        _, (h_n, _) = self.lstm(x_lstm)
        features_lstm = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        combined = torch.cat((features_cnn, features_lstm), dim=1)
        return self.fusion(combined)

# --- 2. GRAD-CAM UTILITY ---
class DualGradCAM:
    def __init__(self, model):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        
        # Hook correct layer based on architecture
        if 'resnet' in model.architecture:
            target_layer = model.cnn.layer4[-1]
        elif 'efficientnet' in model.architecture:
            target_layer = model.cnn.features[-1]
        
        target_layer.register_forward_hook(self.save_feature_maps)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, xc, xl, target_class_index):
        self.model.train() # Switch to train mode for gradients
        # Freeze BatchNorm to prevent stats update
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.Dropout)):
                module.eval()

        xc.requires_grad = True
        self.model.zero_grad()
        output = self.model(xc, xl)
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class_index] = 1
        output.backward(gradient=one_hot)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.feature_maps.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (xc.shape[3], xc.shape[2])) 
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-7)
        
        self.model.eval() 
        return cam

# --- 3. PREPROCESSING UTILS ---
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        max_len = 16000 * 5
        if len(y) > max_len: y = y[:max_len]
        else: y = np.pad(y, (0, max_len - len(y)))
        y = np.append(y[0], y[1:] - 0.97 * y[:-1])

        # Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=512)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
        x_cnn = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        x_lstm = torch.tensor(mfcc.T, dtype=torch.float32).unsqueeze(0)

        return x_cnn, x_lstm, mel_db
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None, None

# --- 4. THE MULTI-MODEL GUI APP ---
class LungApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Respiratech")
        self.root.geometry("1480x900")
        self.logo_img = None
        self.title_img = None
        self.log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.csv")
        self.colors = self._load_branding()
        self.root.configure(bg=self.colors["bg"])
        self._build_menu()
        
        # Store loaded models: {'resnet18': model_obj, ...}
        self.models = {} 
        self.status_labels = {}
        self.architectures = ['resnet18', 'resnet34', 'efficientnet_b0']

        # --- UI LAYOUT ---
        header = tk.Frame(root, bg="#ffffff", height=80)
        header.pack(fill="x")
        header_content = tk.Frame(header, bg="#ffffff")
        header_content.pack(pady=14)
        if self.logo_img is not None:
            tk.Label(
                header_content,
                image=self.logo_img,
                bg="#ffffff",
            ).pack(side="left", padx=(0, 12))
        if self.title_img is not None:
            tk.Label(
                header_content,
                image=self.title_img,
                bg="#ffffff",
            ).pack(side="left")
        else:
            tk.Label(
                header_content,
                text="Respiratech",
                font=("Segoe UI", 24, "bold"),
                bg="#ffffff",
                fg=self.colors["header_text"],
            ).pack(side="left")

        body = tk.Frame(root, bg=self.colors["bg"])
        body.pack(fill="both", expand=True, padx=16, pady=12)

        left_panel = tk.Frame(body, bg=self.colors["bg"])
        left_panel.pack(side="left", fill="y", padx=(0, 12))

        right_panel = tk.Frame(body, bg=self.colors["card"], bd=1, relief="solid", highlightbackground=self.colors["border"])
        right_panel.pack(side="right", fill="both", expand=True)

        # A. Control Panel (Load Models)
        control_frame = tk.LabelFrame(
            left_panel,
            text="Model Configuration",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=10,
            bg=self.colors["bg"],
            fg=self.colors["text"],
        )
        control_frame.pack(pady=6, fill="x")

        for idx, arch in enumerate(self.architectures):
            frame = tk.Frame(control_frame, bg=self.colors["bg"])
            frame.pack(side="left", expand=True, fill="x", padx=10)
            
            tk.Label(
                frame,
                text=f"{arch.upper()}",
                font=("Segoe UI", 9, "bold"),
                bg=self.colors["bg"],
                fg=self.colors["text"],
            ).pack(anchor="w")
            
            btn = tk.Button(
                frame,
                text=f"Load {arch}",
                command=lambda a=arch: self.load_specific_model(a),
                bg=self.colors["primary"],
                fg="white",
                relief="flat",
                activebackground=self.colors["primary_dark"],
            )
            btn.pack(fill="x", pady=2)
            
            lbl = tk.Label(frame, text="Not Loaded", fg=self.colors["muted"], font=("Segoe UI", 9), bg=self.colors["bg"])
            lbl.pack(anchor="w")
            self.status_labels[arch] = lbl

        # B. Prediction Controls
        pred_controls = tk.Frame(left_panel, bg=self.colors["bg"])
        pred_controls.pack(pady=8, fill="x")
        
        self.btn_predict = tk.Button(
            pred_controls,
            text="Select Audio & Compare Models",
            command=self.predict,
            font=("Segoe UI", 12, "bold"),
            bg=self.colors["accent"],
            fg="white",
            relief="flat",
            activebackground=self.colors["accent_dark"],
            state="disabled",
        )
        self.btn_predict.pack(fill="x")

        # C. Clinical Guidance Panel
        advice_frame = tk.LabelFrame(
            left_panel,
            text="Clinical Guidance",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=10,
            bg="#dbeafe",
            fg=self.colors["text"],
        )
        advice_frame.pack(pady=6, fill="both", expand=True)

        self.advice_text = tk.Text(
            advice_frame,
            wrap="word",
            font=("Georgia", 11),
            height=22,
            bg="#dbeafe",
            fg=self.colors["text"],
            relief="solid",
            bd=1,
            padx=10,
            pady=8,
            spacing1=2,
            spacing3=2,
        )
        self.advice_text.pack(fill="both", expand=True)
        self._set_advice("Select an audio file to generate professional guidance.\n\n" + ADVICE_DISCLAIMER)

        # C. Visualization Area
        self.canvas_frame = tk.Frame(right_panel, bg=self.colors["card"])
        self.canvas_frame.pack(fill="both", expand=True, padx=12, pady=12)
        
        # Matplotlib Figure: Top (Input), Bottom (3 Models)
        self.fig = plt.figure(figsize=(14, 6))
        self.gs = self.fig.add_gridspec(2, 3) # 2 rows, 3 columns
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._auto_load_models()
        self._ensure_log_header()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        logs_menu = tk.Menu(menubar, tearoff=0)
        logs_menu.add_command(label="View Logs", command=self._show_logs)
        menubar.add_cascade(label="Logs", menu=logs_menu)
        about_menu = tk.Menu(menubar, tearoff=0)
        about_menu.add_command(label="About Respiratech", command=self._show_about)
        menubar.add_cascade(label="About", menu=about_menu)
        self.root.config(menu=menubar)

    def _show_logs(self):
        log_path = self.log_path
        if not os.path.exists(log_path):
            log_path = filedialog.askopenfilename(
                title="Select Logs CSV",
                filetypes=[("CSV Files", "*.csv")],
            )
            if not log_path:
                return

        rows = []
        try:
            with open(log_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read logs:\n{str(e)}")
            return

        if not rows:
            messagebox.showinfo("Logs", "Log file is empty.")
            return

        logs = tk.Toplevel(self.root)
        logs.title("Logs")
        logs.geometry("900x500")
        logs.configure(bg=self.colors["bg"])
        logs.transient(self.root)
        logs.grab_set()

        container = tk.Frame(logs, bg=self.colors["bg"])
        container.pack(fill="both", expand=True, padx=12, pady=12)

        columns = LOG_COLUMNS

        tree = ttk.Treeview(container, columns=columns, show="headings")
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="w", width=170)

        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)

        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # Use header row if it matches expected columns, otherwise treat all rows as data.
        data_rows = rows
        header = [c.strip().lower() for c in rows[0]]
        expected = [c.strip().lower() for c in columns]
        if header == expected:
            data_rows = rows[1:]

        for row in data_rows:
            padded = row + [""] * (len(columns) - len(row))
            tree.insert("", "end", values=padded[: len(columns)])

    def _show_about(self):
        about = tk.Toplevel(self.root)
        about.title("About Respiratech")
        about.geometry("700x600")
        about.configure(bg=self.colors["bg"])
        about.transient(self.root)
        about.grab_set()

        container = tk.Frame(about, bg=self.colors["bg"])
        container.pack(fill="both", expand=True, padx=16, pady=16)

        title = tk.Label(
            container,
            text="Respiratech - About Us",
            font=("Segoe UI", 16, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"],
        )
        title.pack(anchor="w", pady=(0, 8))

        text = tk.Text(
            container,
            wrap="word",
            font=("Georgia", 11),
            bg=self.colors["card"],
            fg=self.colors["text"],
            relief="solid",
            bd=1,
            padx=12,
            pady=10,
            spacing1=2,
            spacing3=2,
        )
        text.pack(fill="both", expand=True)

        about_body = (
            "Description of Respiratech\n"
            "This web application is designed to analyze and classify lung sounds based on their "
            "pathological characteristics using advanced artificial intelligence. By leveraging "
            "state-of-the-art deep learning models, the system identifies abnormal respiratory "
            "patterns such as wheezes and crackles from recorded or uploaded lung sound audio.\n\n"
            "The application utilizes modern neural network architectures, including convolutional "
            "and sequence-based models, to accurately capture both the acoustic features and temporal "
            "patterns of respiratory sounds. To promote transparency and trust, visual explanations "
            "are provided through Grad-CAM, allowing users to see which sound patterns influenced the "
            "model's predictions.\n\n"
            "Users can easily upload or record lung sounds, view classification results with confidence "
            "scores, and explore visual representations of the analysis. This platform is intended as a "
            "supportive and educational tool for students, researchers, and healthcare professionals, "
            "and is not a substitute for professional medical diagnosis.\n\n"
            "FAQs\n"
            "How do I record properly?\n"
            "Use a quiet environment with minimal background noise. Place the stethoscope or microphone "
            "firmly on the chest area as instructed. Record for at least 10-20 seconds per lung area for "
            "better accuracy. Avoid speaking or moving during recording.\n\n"
            "How do I use the application?\n"
            "Upload a lung sound recording or use the built-in recording feature. Once uploaded, select "
            "the appropriate options and click Analyze to view the classification results and visual "
            "explanations.\n\n"
            "What audio formats are supported?\n"
            "The application supports common audio formats such as WAV and MP3. For best performance, "
            "WAV format is recommended.\n\n"
            "Does background noise affect the results?\n"
            "Yes. Excessive noise such as talking, movement, or environmental sounds may reduce accuracy.\n\n"
            "What do the classifications mean?\n"
            "The system identifies lung sound patterns such as normal breath sounds, wheezes, crackles, "
            "and other abnormal respiratory indicators. These classifications describe acoustic patterns, "
            "not medical diagnoses.\n\n"
            "What is the visual explanation shown in the results?\n"
            "Visual highlights indicate the regions of the sound representation that influenced the model's "
            "prediction, helping users understand how the result was generated.\n\n"
            "The analysis takes too long or fails.\n"
            "Check your internet connection and ensure the uploaded file meets the supported format and "
            "size requirements.\n\n"
            "Who can use this system?\n"
            "The platform is intended for students, researchers, and healthcare professionals interested "
            "in lung sound analysis.\n\n"
            "Contact Information\n"
            "For questions, feedback, or technical concerns, please contact the development team through "
            "the provided communication channels on this website.\n"
            "Mail: respiratech.mhpnhs@gmail.com"
        )

        text.insert(tk.END, about_body)
        text.config(state="disabled")

        close_btn = tk.Button(
            container,
            text="Close",
            command=about.destroy,
            bg=self.colors["primary"],
            fg="white",
            relief="flat",
            activebackground=self.colors["primary_dark"],
        )
        close_btn.pack(anchor="e", pady=(10, 0))

    def _load_branding(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(base_dir, "logo.png")
        if not os.path.exists(logo_path):
            logo_path = os.path.join(base_dir, "Logo.png")
        title_path = os.path.join(base_dir, "Title.png")

        def _clamp(v):
            return max(0, min(255, int(v)))

        def _rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

        def _blend(c1, c2, alpha):
            return (
                _clamp(c1[0] * alpha + c2[0] * (1 - alpha)),
                _clamp(c1[1] * alpha + c2[1] * (1 - alpha)),
                _clamp(c1[2] * alpha + c2[2] * (1 - alpha)),
            )

        colors = {
            "header": "#1f2937",
            "header_text": "#ffffff",
            "primary": "#2563eb",
            "primary_dark": "#1d4ed8",
            "accent": "#16a34a",
            "accent_dark": "#15803d",
            "bg": "#f4f6f8",
            "card": "#ffffff",
            "text": "#111827",
            "muted": "#6b7280",
            "border": "#e5e7eb",
        }

        if PIL_AVAILABLE and os.path.exists(title_path):
            try:
                title_img = Image.open(title_path).convert("RGBA")
                # Keep aspect ratio; target height 40px
                target_h = 40
                scale = target_h / max(1, title_img.size[1])
                target_w = int(title_img.size[0] * scale)
                title_img = title_img.resize((target_w, target_h), Image.LANCZOS)
                self.title_img = ImageTk.PhotoImage(title_img)
            except Exception:
                self.title_img = None
        elif os.path.exists(title_path):
            try:
                self.title_img = tk.PhotoImage(file=title_path)
            except Exception:
                self.title_img = None

        if PIL_AVAILABLE and os.path.exists(logo_path):
            try:
                img = Image.open(logo_path).convert("RGBA")
                logo_base = img.resize((56, 56), Image.LANCZOS)
                self.logo_img = ImageTk.PhotoImage(logo_base)

                sample = img.resize((48, 48), Image.LANCZOS).convert("RGB")
                avg = np.array(sample).reshape(-1, 3).mean(axis=0)
                avg = (int(avg[0]), int(avg[1]), int(avg[2]))

                primary = _blend(avg, (0, 0, 0), 0.6)
                accent = _blend(avg, (255, 255, 255), 0.65)
                bg = _blend(avg, (255, 255, 255), 0.15)
                border = _blend(avg, (255, 255, 255), 0.35)

                colors.update({
                    "header": _rgb_to_hex(primary),
                    "header_text": "#ffffff",
                    "primary": _rgb_to_hex(_blend(primary, (255, 255, 255), 0.15)),
                    "primary_dark": _rgb_to_hex(_blend(primary, (0, 0, 0), 0.75)),
                    "accent": _rgb_to_hex(accent),
                    "accent_dark": _rgb_to_hex(_blend(accent, (0, 0, 0), 0.8)),
                    "bg": _rgb_to_hex(bg),
                    "card": "#ffffff",
                    "text": "#111827",
                    "muted": "#6b7280",
                    "border": _rgb_to_hex(border),
                })
            except Exception:
                pass
        elif os.path.exists(logo_path):
            try:
                self.logo_img = tk.PhotoImage(file=logo_path)
            except Exception:
                self.logo_img = None

        return colors

    def _ensure_log_header(self):
        try:
            needs_header = True
            if os.path.exists(self.log_path):
                needs_header = os.path.getsize(self.log_path) == 0
            if needs_header:
                with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(LOG_COLUMNS)
        except Exception:
            pass

    def _append_log_row(self, file_path, results):
        try:
            self._ensure_log_header()
            labels = {arch: res["label"] for arch, res in results.items()}
            confs = {arch: res["conf"] for arch, res in results.items()}
            best_arch = max(results.keys(), key=lambda a: results[a]["conf"])
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                os.path.basename(file_path),
                labels.get("resnet18", ""),
                f'{confs.get("resnet18", ""):.1f}%' if "resnet18" in confs else "",
                labels.get("resnet34", ""),
                f'{confs.get("resnet34", ""):.1f}%' if "resnet34" in confs else "",
                labels.get("efficientnet_b0", ""),
                f'{confs.get("efficientnet_b0", ""):.1f}%' if "efficientnet_b0" in confs else "",
                best_arch.upper(),
                labels.get(best_arch, ""),
                f'{confs.get(best_arch, 0):.1f}%' if best_arch in confs else "",
            ]
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception:
            pass

    def load_specific_model(self, arch):
        path = filedialog.askopenfilename(title=f"Load {arch} Weights", filetypes=[("Model Files", "*.pth")])
        if not path: return
        self._load_model_from_path(arch, path, show_errors=True)

    def _auto_load_models(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "saved_models")

        for arch in self.architectures:
            self._set_status(arch, "Checking...", "gray")

        def _loader():
            for arch in self.architectures:
                path = os.path.join(models_dir, f"{arch}.pth")
                if os.path.exists(path):
                    self._load_model_from_path(arch, path, show_errors=False)
                else:
                    self._set_status(arch, "Missing file", "gray")

        threading.Thread(target=_loader, daemon=True).start()

    def _load_model_from_path(self, arch, path, show_errors=False):
        try:
            self._set_status(arch, "Loading...", "orange")

            # Initialize specific model
            model = SOTADualModel(num_classes=len(CLASSES), architecture=arch).to(DEVICE)
            
            # Smart Loader (Fix Key Names)
            state_dict = torch.load(path, map_location=DEVICE)
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("resnet."):
                    new_key = key.replace("resnet.", "cnn.")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            model.load_state_dict(new_state_dict)
            model.eval()
            
            self.models[arch] = model
            self._set_status(arch, "Ready ✅", "green")
            self.btn_predict.config(state="normal")
            return True
        except Exception as e:
            self._set_status(arch, "Error ❌", "red")
            if show_errors:
                messagebox.showerror("Error", f"Failed to load {arch}:\n{str(e)}")
            return False

    def _set_status(self, arch, text, color):
        def _update():
            lbl = self.status_labels.get(arch)
            if lbl:
                lbl.config(text=text, fg=color)
        self.root.after(0, _update)

    def predict(self):
        if not self.models:
            messagebox.showwarning("Warning", "No models loaded! Please load at least one model.")
            return
            
        file_path = filedialog.askopenfilename(title="Select Audio", filetypes=[("Audio Files", "*.wav")])
        if not file_path: return
        
        threading.Thread(target=self._run_inference_all, args=(file_path,)).start()

    def _run_inference_all(self, file_path):
        xc, xl, spec_img = preprocess_audio(file_path)
        if xc is None: return
        xc, xl = xc.to(DEVICE), xl.to(DEVICE)

        results = {}

        # Loop through all loaded models
        for arch, model in self.models.items():
            try:
                # 1. Prediction
                with torch.no_grad():
                    output = model(xc, xl)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    conf, pred_idx = torch.max(probs, 0)
                    
                # 2. Grad-CAM
                grad_cam = DualGradCAM(model)
                heatmap = grad_cam(xc, xl, pred_idx.item())
                
                results[arch] = {
                    "label": CLASSES[pred_idx.item()],
                    "conf": conf.item() * 100,
                    "probs": probs.cpu().numpy(),
                    "heatmap": heatmap
                }
            except Exception as e:
                print(f"Error inferencing {arch}: {e}")

        if results:
            self._append_log_row(file_path, results)

        self.root.after(0, lambda: self._update_ui(results, spec_img))

    def _update_ui(self, results, spec_img):
        self.fig.clf()
        
        # --- ROW 1: INPUT SPECTROGRAM (Centered) ---
        ax_main = self.fig.add_subplot(2, 1, 1) # Top row spanning all cols (simulated)
        # Actually let's use gridspec for better layout
        gs = self.fig.add_gridspec(2, 3)
        
        ax_input = self.fig.add_subplot(gs[0, :]) # Span full top row
        ax_input.imshow(spec_img, aspect='auto', origin='lower', cmap='magma')
        ax_input.set_title("Original Input Audio Spectrogram")
        ax_input.axis('off')

        # --- ROW 2: MODEL RESULTS ---
        for i, arch in enumerate(self.architectures):
            ax = self.fig.add_subplot(gs[1, i])
            
            if arch in results:
                res = results[arch]
                label = res['label']
                conf = res['conf']
                heatmap = res['heatmap']
                
                # Overlay Logic
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                
                spec_norm = (spec_img - spec_img.min()) / (spec_img.max() - spec_img.min())
                spec_rgb = (spec_norm * 255).astype(np.uint8)
                spec_rgb = cv2.cvtColor(spec_rgb, cv2.COLOR_GRAY2RGB)
                spec_rgb = cv2.resize(spec_rgb, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
                
                overlay = cv2.addWeighted(spec_rgb, 0.6, heatmap_colored, 0.4, 0)
                
                ax.imshow(overlay, aspect='auto', origin='lower')
                
                # Title with Color coding
                color = "green" if label == "Normal" else "red"
                title_text = f"{arch.upper()}\n{label}: {conf:.1f}%"
                ax.set_title(title_text, color=color, fontweight="bold")
                
                # Show full probabilities in xlabel
                probs_text = "\n".join([f"{c}: {p*100:.1f}%" for c, p in zip(CLASSES, res['probs'])])
                ax.set_xlabel(probs_text, fontsize=8)
                
            else:
                ax.text(0.5, 0.5, "Not Loaded", ha='center', va='center', color='gray')
                ax.set_title(arch.upper())
            
            ax.set_xticks([])
            ax.set_yticks([])

        self.fig.tight_layout()
        self.canvas.draw()

        # Update guidance based on most confident result
        if results:
            best_arch = max(results.keys(), key=lambda a: results[a]["conf"])
            best_label = results[best_arch]["label"]
            advice = ADVICE.get(best_label, "")
            if advice:
                note = ""
                labels = {arch: res["label"] for arch, res in results.items()}
                unique_labels = sorted(set(labels.values()))
                if len(unique_labels) > 1:
                    counts = {}
                    for lbl in labels.values():
                        counts[lbl] = counts.get(lbl, 0) + 1
                    counts_text = ", ".join([f"{k}: {v}" for k, v in counts.items()])
                    differing = ", ".join([f"{a.upper()}={l}" for a, l in labels.items()])
                    note = (
                        "\n\nModel Agreement Notice\n"
                        f"Results are not fully consistent across models ({counts_text}). "
                        f"Model outputs: {differing}.\n"
                        "Doctor verification is needed to confirm the final classification."
                    )

                advice = f"{advice}{note}\n\n{ADVICE_DISCLAIMER}"
                self._set_advice(advice)

    def _set_advice(self, text):
        self.advice_text.config(state="normal")
        self.advice_text.delete("1.0", tk.END)
        self.advice_text.insert(tk.END, text)
        self.advice_text.config(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = LungApp(root)
    root.mainloop()
