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

# --- CONFIGURATION ---
CLASSES = ['Crackle', 'Normal', 'Wheeze'] 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.root.title("Lung Sound Arena: Multi-Model Comparison")
        self.root.geometry("1400x900") # Wider window for side-by-side
        
        # Store loaded models: {'resnet18': model_obj, ...}
        self.models = {} 
        self.status_labels = {}

        # --- UI LAYOUT ---
        header = tk.Frame(root, bg="#2c3e50", height=80)
        header.pack(fill="x")
        tk.Label(header, text="🫁 AI Model Comparison Arena", font=("Segoe UI", 24, "bold"), bg="#2c3e50", fg="white").pack(pady=20)

        # A. Control Panel (Load Models)
        control_frame = tk.LabelFrame(root, text="Model Configuration", font=("Arial", 11, "bold"), padx=10, pady=10)
        control_frame.pack(pady=10, fill="x", padx=20)

        architectures = ['resnet18', 'resnet34', 'efficientnet_b0']
        
        for idx, arch in enumerate(architectures):
            frame = tk.Frame(control_frame)
            frame.pack(side="left", expand=True, fill="x", padx=10)
            
            tk.Label(frame, text=f"{arch.upper()}", font=("Arial", 10, "bold")).pack(anchor="w")
            
            btn = tk.Button(frame, text=f"📂 Load {arch}", command=lambda a=arch: self.load_specific_model(a), bg="#3498db", fg="white")
            btn.pack(fill="x", pady=2)
            
            lbl = tk.Label(frame, text="Not Loaded", fg="gray", font=("Arial", 9))
            lbl.pack(anchor="w")
            self.status_labels[arch] = lbl

        # B. Prediction Controls
        pred_controls = tk.Frame(root)
        pred_controls.pack(pady=10)
        
        self.btn_predict = tk.Button(pred_controls, text="🔍 Select Audio & Compare All Models", command=self.predict, font=("Arial", 14), bg="#27ae60", fg="white", state="disabled")
        self.btn_predict.pack()

        # C. Visualization Area
        self.canvas_frame = tk.Frame(root, bg="white")
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Matplotlib Figure: Top (Input), Bottom (3 Models)
        self.fig = plt.figure(figsize=(14, 6))
        self.gs = self.fig.add_gridspec(2, 3) # 2 rows, 3 columns
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_specific_model(self, arch):
        path = filedialog.askopenfilename(title=f"Load {arch} Weights", filetypes=[("Model Files", "*.pth")])
        if not path: return

        try:
            self.status_labels[arch].config(text="Loading...", fg="orange")
            self.root.update()

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
            self.status_labels[arch].config(text="Ready ✅", fg="green")
            self.btn_predict.config(state="normal")
            
        except Exception as e:
            self.status_labels[arch].config(text="Error ❌", fg="red")
            messagebox.showerror("Error", f"Failed to load {arch}:\n{str(e)}")

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
        architectures = ['resnet18', 'resnet34', 'efficientnet_b0']
        
        for i, arch in enumerate(architectures):
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

if __name__ == "__main__":
    root = tk.Tk()
    app = LungApp(root)
    root.mainloop()