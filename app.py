import os
import time
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import spaces
    gpu_decorator = spaces.GPU(duration=15)
except Exception:
    def gpu_decorator(fn):
        return fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import gradio as gr

# -----------------------------------------------------------------------------
# 1. CONSTANTS & METADATA (ENGLISH ONLY)
# -----------------------------------------------------------------------------
CLASS_NAMES = [
    'Agricultural_Land',
    'Beaches_Coastline',
    'Built_up_Areas',
    'Grasslands_Open_Land',
    'Hills_Mountains',
    'Trees_Forests',
    'Water_Bodies'
]

CLASS_ICONS = {
    'Agricultural_Land': '🌾',
    'Beaches_Coastline': '🏖️',
    'Built_up_Areas': '🏙️',
    'Grasslands_Open_Land': '🌿',
    'Hills_Mountains': '⛰️',
    'Trees_Forests': '🌲',
    'Water_Bodies': '🌊'
}

CLASS_DESCRIPTIONS = {
    'Agricultural_Land': 'Cultivated cropland, dynamic paddy fields, and alluvial agricultural terrain.',
    'Beaches_Coastline': 'Sandy shorelines, intertidal coastal sand, and tidal mudflats.',
    'Built_up_Areas': 'Urban settlements, residential structures, transportation, and impervious infrastructure.',
    'Grasslands_Open_Land': 'Open grazing fields, riverine sandbars (chars), and uncultivated grassland.',
    'Hills_Mountains': 'Elevated hilly topography, forested slopes, and undulating terrain.',
    'Trees_Forests': 'Dense natural canopy, mangrove belts, protected forests, and heavy vegetation.',
    'Water_Bodies': 'Rivers, lakes, natural water channels, canals, and seasonal wetlands.'
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# -----------------------------------------------------------------------------
# 2. LITERS-MAMBA ARCHITECTURE
# -----------------------------------------------------------------------------
@torch.jit.script
def fast_selective_scan(u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B_mat: torch.Tensor, C_mat: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    batch, dim, seq_len = u.shape
    d_state = A.shape[-1]
    A_exp = A.unsqueeze(0)
    y_list = []
    h = torch.zeros((batch, dim, d_state), device=u.device, dtype=u.dtype)
    for t in range(seq_len):
        dt_t = dt[:, t, :]
        u_t = u[:, :, t]
        B_t = B_mat[:, t, :]
        C_t = C_mat[:, t, :]
        dA_t = torch.exp(dt_t.unsqueeze(-1) * A_exp)
        dB_t = (dt_t * u_t).unsqueeze(-1) * B_t.unsqueeze(1)
        h = h * dA_t + dB_t
        y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1) + u_t * D
        y_list.append(y_t)
    return torch.stack(y_list, dim=-1)

class MultiScaleStem(nn.Module):
    def __init__(self, in_chans=3, out_dim=192):
        super().__init__()
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, out_dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )
    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        return self.fuse(torch.cat([f3, f5], dim=1))

class FastOmniSSMBlock(nn.Module):
    def __init__(self, d_model=192, d_state=16, expand=2, dt_rank=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dir_weights = nn.Parameter(torch.ones(4) / 4.0)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def parallel_ssm_scan(self, u_4b):
        batch_4, dim, seq_len = u_4b.shape
        x_dbl = self.x_proj(u_4b.transpose(1, 2))
        dt, B_mat, C_mat = torch.split(x_dbl, [16, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        return fast_selective_scan(u_4b, dt, A, B_mat, C_mat, self.D)

    def forward(self, x, H=14, W=14):
        residual = x
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_proj, z = xz.chunk(2, dim=-1)
        x_conv = self.act(self.conv1d(x_proj.transpose(1, 2)))
        B, D_in, L = x_conv.shape
        x_2d = x_conv.view(B, D_in, H, W)
        u1 = x_conv
        u2 = torch.flip(x_conv, dims=[-1])
        u3 = x_2d.transpose(2, 3).contiguous().view(B, D_in, L)
        u4 = torch.flip(u3, dims=[-1])
        u_batched = torch.cat([u1, u2, u3, u4], dim=0)
        y_batched = self.parallel_ssm_scan(u_batched)
        y1, y2_raw, y3_raw, y4_raw = y_batched.chunk(4, dim=0)
        y2 = torch.flip(y2_raw, dims=[-1])
        y3 = y3_raw.view(B, D_in, W, H).transpose(2, 3).contiguous().view(B, D_in, L)
        y4 = torch.flip(y4_raw, dims=[-1]).view(B, D_in, W, H).transpose(2, 3).contiguous().view(B, D_in, L)
        w = F.softmax(self.dir_weights, dim=0)
        y = w[0]*y1 + w[1]*y2 + w[2]*y3 + w[3]*y4
        y_gated = y.transpose(1, 2) * F.silu(z)
        return self.out_proj(y_gated) + residual

class LiteRSMamba(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=7, embed_dim=192, depth=6):
        super().__init__()
        self.stem = MultiScaleStem(in_chans=in_chans, out_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, embed_dim))
        self.blocks = nn.ModuleList([FastOmniSSMBlock(d_model=embed_dim, d_state=16, expand=2) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        feat = self.stem(x)
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens, 14, 14)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        out = self.head(pooled)
        return out

# -----------------------------------------------------------------------------
# 3. MODEL INITIALIZATION
# -----------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).parent.resolve()
CKPT_PATH = BASE_DIR / 'checkpoints' / 'best_liters_mamba_ArcGIS_Map.pth'

model = LiteRSMamba(num_classes=7)
if CKPT_PATH.exists():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
model.to(DEVICE)
model.eval()

# -----------------------------------------------------------------------------
# 4. PREDICTION HANDLER (HIGH-END MODERN HTML OUTPUT)
# -----------------------------------------------------------------------------
EMPTY_STATE_HTML = """
<div style="background: #ffffff; border: 1.5px dashed #cbd5e1; border-radius: 16px; padding: 48px 24px; text-align: center; box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03);">
    <div style="width: 56px; height: 56px; margin: 0 auto 16px auto; background: #eff6ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 26px; border: 1px solid #bfdbfe;">
        🛰️
    </div>
    <div style="font-size: 17px; font-weight: 700; color: #1e293b; margin-bottom: 6px;">Ready for Satellite Scene</div>
    <div style="font-size: 13.5px; color: #64748b; max-width: 380px; margin: 0 auto 20px auto; line-height: 1.5;">
        Upload or drop an aerial or satellite tile on the left and click Classify to inspect land cover predictions.
    </div>
    <div style="display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;">
        <span style="background: #f1f5f9; color: #475569; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 9999px;">⚡ 2.05M Params</span>
        <span style="background: #f1f5f9; color: #475569; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 9999px;">🎯 99.01% Top-1 Accuracy</span>
        <span style="background: #f1f5f9; color: #475569; font-size: 12px; font-weight: 600; padding: 4px 12px; border-radius: 9999px;">🌐 7 Land Classes</span>
    </div>
</div>
"""

@gpu_decorator
def predict(input_image):
    if input_image is None:
        return EMPTY_STATE_HTML, {}, ""
    
    t0 = time.perf_counter()
    original_pil = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image
    original_pil = original_pil.convert('RGB')
    
    inp_tensor = EVAL_TRANSFORMS(original_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(inp_tensor)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx]) * 100.0
    pred_icon = CLASS_ICONS[pred_class]
    pred_desc = CLASS_DESCRIPTIONS[pred_class]
    clean_name = pred_class.replace('_', ' ')

    prob_dict = {f"{CLASS_ICONS[c]} {c.replace('_', ' ')}": float(probs[i]) for i, c in enumerate(CLASS_NAMES)}

    device_label = "ZeroGPU (CUDA)" if DEVICE.type == 'cuda' else "CPU Acceleration"

    badge_html = f"""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 26px; box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.05), 0 8px 10px -6px rgba(15, 23, 42, 0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;">
            <span style="background: #f1f5f9; color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 700; padding: 4px 10px; border-radius: 6px;">
                Detected Category
            </span>
            <span style="background: linear-gradient(135deg, #0284c7, #2563eb); color: #ffffff; font-size: 13px; font-weight: 700; padding: 4px 14px; border-radius: 9999px; box-shadow: 0 2px 8px rgba(37,99,235,0.25);">
                {pred_conf:.1f}% Confidence
            </span>
        </div>
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px;">
            <div style="font-size: 38px; line-height: 1; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));">{pred_icon}</div>
            <div>
                <div style="font-size: 24px; font-weight: 800; color: #0f172a; letter-spacing: -0.3px;">{clean_name}</div>
                <div style="font-size: 13.5px; color: #64748b; line-height: 1.4; margin-top: 3px;">{pred_desc}</div>
            </div>
        </div>
        <div style="margin-top: 16px; background: #f8fafc; border-radius: 8px; height: 8px; overflow: hidden; border: 1px solid #e2e8f0;">
            <div style="background: linear-gradient(90deg, #38bdf8, #2563eb); height: 100%; width: {pred_conf:.1f}%; border-radius: 8px;"></div>
        </div>
    </div>
    """

    telemetry_html = f"""
    <div style="display: flex; justify-content: space-around; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; margin-top: 12px; font-size: 12.5px; color: #475569; box-shadow: 0 2px 6px rgba(0,0,0,0.02);">
        <div>⚡ <b>Latency:</b> <span style="color: #0f172a; font-weight: 700;">{elapsed_ms:.1f} ms</span></div>
        <div style="color: #cbd5e1;">|</div>
        <div>💻 <b>Compute:</b> <span style="color: #0f172a; font-weight: 600;">{device_label}</span></div>
        <div style="color: #cbd5e1;">|</div>
        <div>📦 <b>Model:</b> <span style="color: #0f172a; font-weight: 600;">2.05M Params</span></div>
    </div>
    """

    return badge_html, prob_dict, telemetry_html

# -----------------------------------------------------------------------------
# 5. MODERN PREMIUM GRADIO INTERFACE (ENGLISH ONLY)
# -----------------------------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

body, .gradio-container {
    background: #f8fafc !important;
    max-width: 1080px !important;
    margin: 0 auto !important;
    padding-top: 12px !important;
}

#header-card {
    text-align: center;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 28px 24px 22px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
}

#header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1d4ed8;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.3px;
}

#header-title {
    font-size: 32px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
}

#header-subtitle {
    font-size: 15px;
    color: #64748b;
    margin: 0 auto;
    max-width: 580px;
    line-height: 1.5;
}

.panel-card {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 18px !important;
    padding: 18px !important;
    box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04) !important;
}

.primary-btn {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
}

footer {
    display: none !important;
}
"""

with gr.Blocks(title="LiteRS-Mamba") as demo:
    
    gr.HTML("""
    <div id="header-card">
        <div id="header-badge">
            <span>⚡ LiteRS-Mamba v1.0 • O4-SSM Architecture</span>
        </div>
        <h1 id="header-title">🛰️ Satellite Land Cover Classification</h1>
        <p id="header-subtitle">
            Ultra-lightweight State Space Model engineered for high-precision Earth observation & multi-platform scene intelligence.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.Markdown("### 📥 Input Satellite Scene")
            image_input = gr.Image(
                type="pil",
                label="Upload or Drop Satellite Tile",
                height=300
            )
            classify_btn = gr.Button(
                "⚡ Classify Land Cover",
                variant="primary",
                size="lg",
                elem_classes=["primary-btn"]
            )

        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.Markdown("### 📊 Classification Intelligence")
            badge_output = gr.HTML(EMPTY_STATE_HTML)
            label_output = gr.Label(num_top_classes=5, label="Top Class Distribution")
            telemetry_output = gr.HTML("")

    classify_btn.click(
        fn=predict,
        inputs=[image_input],
        outputs=[badge_output, label_output, telemetry_output],
        api_name="predict"
    )

if __name__ == '__main__':
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        share=False
    )
