# =============================================================================
# FIG. 6 — Grad-CAM panels rebuilt from the trained models (no burned-in titles)
# Needs, from the main notebook: cells 3-7 and 11-13 (paths, eval_transform,
# build_model, device) and the function definitions in CELL 32/33
# (get_target_layer, get_reshape_transform). The heavy Grad-CAM loop in that
# cell skips everything already saved, so re-running it is quick.
# The compressed images are recreated in memory with the same PIL call used in
# create_compressed_versions(), so no compressed/ folder is needed on disk.
# =============================================================================
import io, torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

for name in ["build_model", "eval_transform", "device", "get_target_layer", "get_reshape_transform"]:
    if name not in globals():
        raise RuntimeError(f"'{name}' is not defined: run notebook cells 3-7, 11-13 and the "
                           "function part of CELL 32/33 first.")

METHOD   = "Deepfakes"
ROWS     = [("resnet50", "(a) ResNet50: signal degradation"),
            ("mobilenetv2", "(b) MobileNetV2: attention drift")]
IMAGE_NAME = None        # set e.g. "000_003_0010.png" to force a specific test image

def load_weights(model_name):
    for p in (BASE_PATH / "models" / f"{METHOD}_{model_name}.pth", BASE_PATH / f"{METHOD}_{model_name}.pth"):
        if p.exists():
            mdl = build_model(model_name)
            mdl.load_state_dict(torch.load(p, map_location=device)); mdl.to(device).eval()
            return mdl
    raise FileNotFoundError(f"No weights for {METHOD}_{model_name}.pth in models/ or project root")

def jpeg(img, q):
    if q is None:
        return img
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def predict(mdl, img):
    with torch.no_grad():
        return int(torch.argmax(mdl(eval_transform(img).unsqueeze(0).to(device)), 1).item())

models_ = {m: load_weights(m) for m, _ in ROWS}
fake_dir = BASE_PATH / "dataset" / METHOD / "test" / "fake"
candidates = sorted(p for p in fake_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
if IMAGE_NAME:
    candidates = [fake_dir / IMAGE_NAME]
# first fake test image that BOTH models classify correctly (fake = class 0) at original quality
chosen = next(p for p in candidates
              if all(predict(mdl, Image.open(p).convert("RGB")) == 0 for mdl in models_.values()))
print("Image used:", chosen.name)

def overlay(mdl, model_name, img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb = np.asarray(img).astype(np.float32) / 255.0
    cam = GradCAM(model=mdl, target_layers=[get_target_layer(model_name, mdl)],
                  reshape_transform=get_reshape_transform(model_name))
    g = cam(input_tensor=eval_transform(img).unsqueeze(0).to(device))[0]
    return show_cam_on_image(rgb, g, use_rgb=True)

base = Image.open(chosen).convert("RGB")
qs = [None, 90, 70, 50, 30]
fig, axes = plt.subplots(2, 5, figsize=(FULL_W, 3.75))
for r, (m, row_label) in enumerate(ROWS):
    for c, (q, ql, qk) in enumerate(zip(qs, QLABELS, QUALS)):
        ax = axes[r, c]
        ax.imshow(overlay(models_[m], m, jpeg(base, q))); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4)
        acc = master[METHOD][m][qk]["accuracy"]          # seed-42 accuracy, as in the old figure
        ax.set_xlabel(f"{ql}\nacc. {acc:.3f}", fontsize=8, labelpad=2)
fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.10, wspace=0.04, hspace=0.42)
for r, (m, row_label) in enumerate(ROWS):
    p = axes[r, 0].get_position()
    fig.text(p.x0, p.y1 + 0.012, row_label, ha="left", va="bottom", fontsize=8, fontweight="bold")
save(fig, 6)
