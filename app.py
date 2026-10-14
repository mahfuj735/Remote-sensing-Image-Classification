import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.cm as cm

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
    'Agricultural_Land': 'Cultivated cropland, paddy fields, and alluvial agricultural terrain.',
    'Beaches_Coastline': 'Sandy shorelines, intertidal coastal sand, and coastal mudflats.',
    'Built_up_Areas': 'Urban settlements, buildings, roads, and impervious infrastructure.',
    'Grasslands_Open_Land': 'Open grazing fields, riverine chars, and naturally vegetated ground.',
    'Hills_Mountains': 'Elevated hilly topography, tea gardens, and tropical slopes.',
    'Trees_Forests': 'Dense mangrove canopy, forestry tracts, and wooded areas.',
    'Water_Bodies': 'Rivers, canals, lakes, and seasonal flood wetland waters.'
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
        self.last_stem_features = None

    def forward(self, x):
        feat = self.stem(x)
        self.last_stem_features = feat
        tokens = feat.flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens, 14, 14)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        out = self.head(pooled)
        return out

# -----------------------------------------------------------------------------
# 3. MODEL MANAGEMENT & CACHING
# -----------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).parent.resolve()
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

MODEL_CACHE = {}

def get_model(checkpoint_choice='ArcGIS Map (Default)'):
    tag_map = {
        'ArcGIS Map (Default)': 'best_liters_mamba_ArcGIS_Map.pth',
        'Bing Maps': 'best_liters_mamba_Bing_Maps.pth',
        'Google Maps': 'best_liters_mamba_Google_Maps.pth'
    }
    filename = tag_map.get(checkpoint_choice, 'best_liters_mamba_ArcGIS_Map.pth')
    if filename in MODEL_CACHE:
        return MODEL_CACHE[filename]

    ckpt_path = CHECKPOINT_DIR / filename
    model = LiteRSMamba(num_classes=7)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    MODEL_CACHE[filename] = model
    return model

_ = get_model('ArcGIS Map (Default)')

# -----------------------------------------------------------------------------
# 4. TTA & ATTENTION SALIENCY
# -----------------------------------------------------------------------------
def apply_reference_histogram_matching(image_np):
    matched = np.zeros_like(image_np)
    target_stats = [(92.4, 45.2), (105.1, 41.8), (90.2, 38.6)]
    for c in range(3):
        ch = image_np[:, :, c].astype(np.float32)
        c_mean, c_std = ch.mean(), ch.std() + 1e-6
        t_mean, t_std = target_stats[c]
        aligned = (ch - c_mean) * (t_std / c_std) + t_mean
        matched[:, :, c] = np.clip(aligned, 0, 255).astype(np.uint8)
    return matched

def generate_saliency_overlay(model, original_pil):
    feat = model.last_stem_features
    if feat is None:
        return original_pil
    sal_map = feat.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    sal_map = (sal_map - sal_map.min()) / (sal_map.max() - sal_map.min() + 1e-8)
    sal_pil = Image.fromarray((sal_map * 255).astype(np.uint8)).resize(original_pil.size, Image.Resampling.BILINEAR)
    sal_norm = np.array(sal_pil).astype(np.float32) / 255.0
    colored_map = (cm.jet(sal_norm)[:, :, :3] * 255).astype(np.uint8)
    orig_np = np.array(original_pil.convert('RGB'))
    blended = (0.55 * orig_np + 0.45 * colored_map).astype(np.uint8)
    return Image.fromarray(blended)

# -----------------------------------------------------------------------------
# 5. PREDICTION HANDLER (ENGLISH ONLY, MINIMALIST)
# -----------------------------------------------------------------------------
@gpu_decorator
def predict(input_image, checkpoint_choice, use_tta, show_heatmap):
    if input_image is None:
        return (
            "<div style='text-align: center; color: #94a3b8; padding: 24px;'>Please upload a satellite image to classify.</div>",
            {},
            None,
            ""
        )
    
    t0 = time.perf_counter()
    original_pil = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image
    original_pil = original_pil.convert('RGB')
    
    if use_tta:
        img_np = np.array(original_pil)
        img_np = apply_reference_histogram_matching(img_np)
        processed_pil = Image.fromarray(img_np)
    else:
        processed_pil = original_pil

    inp_tensor = EVAL_TRANSFORMS(processed_pil).unsqueeze(0).to(DEVICE)
    model = get_model(checkpoint_choice)
    
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

    overlay_pil = generate_saliency_overlay(model, original_pil) if show_heatmap else processed_pil

    badge_html = f"""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; text-align: center;">
        <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 1.5px; color: #64748b; font-weight: 600; margin-bottom: 4px;">Predicted Category</div>
        <div style="font-size: 26px; font-weight: 800; color: #0f172a; margin: 4px 0;">{pred_icon} {clean_name}</div>
        <div style="font-size: 13px; color: #475569; margin-bottom: 10px; max-width: 440px; margin-left: auto; margin-right: auto;">{pred_desc}</div>
        <div style="display: inline-block; background: #e0f2fe; color: #0284c7; padding: 4px 14px; border-radius: 9999px; font-weight: 700; font-size: 14px;">
            {pred_conf:.1f}% Confidence
        </div>
    </div>
    """

    device_label = "GPU" if DEVICE.type == 'cuda' else "CPU"
    telemetry_html = f"""
    <div style="text-align: center; font-size: 12px; color: #64748b; padding-top: 6px;">
        ⚡ <b>Inference:</b> {elapsed_ms:.1f} ms ({device_label}) &nbsp;|&nbsp; 📦 <b>Model:</b> 2.05M Params (8.25 MB)
    </div>
    """

    return badge_html, prob_dict, overlay_pil, telemetry_html

# -----------------------------------------------------------------------------
# 6. MINIMALIST, CLEAN GRADIO INTERFACE (ENGLISH ONLY)
# -----------------------------------------------------------------------------
custom_css = """
.gradio-container { max-width: 1050px !important; margin: 0 auto !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
#title-banner { text-align: center; padding: 20px 0 10px 0; margin-bottom: 8px; }
#title-banner h1 { font-size: 28px; font-weight: 800; color: #0f172a; margin: 0 0 6px 0; }
#title-banner p { font-size: 14px; color: #64748b; margin: 0; }
footer { display: none !important; }
"""

with gr.Blocks(title="LiteRS-Mamba") as demo:
    
    gr.HTML("""
    <div id="title-banner">
        <h1>🛰️ LiteRS-Mamba</h1>
        <p>Satellite Land Cover Classification for Bangladesh</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Input Satellite Image", height=280)
            
            with gr.Row():
                checkpoint_dropdown = gr.Dropdown(
                    choices=['ArcGIS Map (Default)', 'Bing Maps', 'Google Maps'],
                    value='ArcGIS Map (Default)',
                    label="Model Training Domain"
                )
            
            with gr.Row():
                tta_checkbox = gr.Checkbox(label="Color Adaptation (TTA)", value=False)
                heatmap_checkbox = gr.Checkbox(label="Show Attention Heatmap", value=True)

            classify_btn = gr.Button("Classify Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            badge_output = gr.HTML("<div style='text-align: center; color: #94a3b8; padding: 24px; border: 1px dashed #cbd5e1; border-radius: 12px;'>Upload a satellite image to classify.</div>")
            image_output = gr.Image(label="Attention Heatmap Overlay", height=240)
            label_output = gr.Label(num_top_classes=3, label="Top Probabilities")
            telemetry_output = gr.HTML("")

    classify_btn.click(
        fn=predict,
        inputs=[image_input, checkpoint_dropdown, tta_checkbox, heatmap_checkbox],
        outputs=[badge_output, label_output, image_output, telemetry_output],
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




