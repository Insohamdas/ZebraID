import base64
import re
import numpy as np
from pathlib import Path
from PIL import Image

root = Path(__file__).parent.parent
static_dir = root / "demo" / "static"
static_dir.mkdir(exist_ok=True)

# 1. Process Zebra.png into dual favicons (Dark Mode White Stripes & Light Mode Black Stripes)
fav_src = root / "Zebra.png"
if not fav_src.exists():
    fav_src = root / "zebra.png"

fav_img = Image.open(fav_src).convert("RGBA")
arr = np.array(fav_img).copy()

rgb = arr[:, :, :3].astype(np.float32)
brightness = np.mean(rgb, axis=2)

# Array for Dark Mode (White Stripes #ffffff)
arr_dark = np.zeros_like(arr, dtype=np.uint8)
arr_dark[:, :, 0] = 255
arr_dark[:, :, 1] = 255
arr_dark[:, :, 2] = 255

# Array for Light Mode (Black Stripes #0c0a09)
arr_light = np.zeros_like(arr, dtype=np.uint8)
arr_light[:, :, 0] = 12
arr_light[:, :, 1] = 10
arr_light[:, :, 2] = 9

alpha = np.zeros_like(brightness, dtype=np.uint8)
mask_white = brightness >= 200
mask_bg = brightness <= 80
mask_edge = (~mask_white) & (~mask_bg)

alpha[mask_white] = 255
alpha[mask_bg] = 0

edge_alpha = 255.0 * (brightness[mask_edge] - 80.0) / (200.0 - 80.0)
alpha[mask_edge] = np.clip(edge_alpha, 0, 255).astype(np.uint8)

arr_dark[:, :, 3] = alpha
arr_light[:, :, 3] = alpha

img_dark = Image.fromarray(arr_dark, 'RGBA')
img_light = Image.fromarray(arr_light, 'RGBA')

# Tight crop bounding box
nonzero_y, nonzero_x = np.where(alpha > 0)
min_y, max_y = nonzero_y.min(), nonzero_y.max()
min_x, max_x = nonzero_x.min(), nonzero_x.max()
padding = 30
crop_x1 = max(0, min_x - padding)
crop_y1 = max(0, min_y - padding)
crop_x2 = min(fav_img.width, max_x + padding)
crop_y2 = min(fav_img.height, max_y + padding)

cropped_dark = img_dark.crop((crop_x1, crop_y1, crop_x2, crop_y2)).resize((256, 256), Image.Resampling.LANCZOS)
cropped_light = img_light.crop((crop_x1, crop_y1, crop_x2, crop_y2)).resize((256, 256), Image.Resampling.LANCZOS)

white_fav_path = static_dir / "white_stripes_favicon.png"
black_fav_path = static_dir / "black_stripes_favicon.png"

cropped_dark.save(white_fav_path, "PNG")
cropped_light.save(black_fav_path, "PNG")
cropped_light.save(static_dir / "favicon.png", "PNG")

# 2. Process Logo (zebraId_logo.png)
logo_src = root / "zebraId_logo.png"
logo_img = Image.open(logo_src).convert("RGBA")
logo_arr = np.array(logo_img)
logo_alpha = logo_arr[:, :, 3]

nonzero_y, nonzero_x = np.where(logo_alpha > 0)
min_y, max_y = nonzero_y.min(), nonzero_y.max()
min_x, max_x = nonzero_x.min(), nonzero_x.max()
padding = 4
crop_x1 = max(0, min_x - padding)
crop_y1 = max(0, min_y - padding)
crop_x2 = min(logo_img.width, max_x + padding)
crop_y2 = min(logo_img.height, max_y + padding)

cropped_logo = logo_img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
logo_perfect_path = static_dir / "zebraid_logo_perfect.png"
cropped_logo.save(logo_perfect_path, format="PNG")
cropped_logo.save(static_dir / "logo.png", format="PNG")

# 3. Base64 Encode
logo_b64 = base64.b64encode(logo_perfect_path.read_bytes()).decode('utf-8')
white_fav_b64 = base64.b64encode(white_fav_path.read_bytes()).decode('utf-8')
black_fav_b64 = base64.b64encode(black_fav_path.read_bytes()).decode('utf-8')

# 4. Embed into index.html
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

# Update nav-brand
nav_pattern = r'<a href="#" class="nav-brand">.*?</a>'
nav_replacement = f'''<a href="#" class="nav-brand">
        <img src="data:image/png;base64,{logo_b64}" alt="ZebraID Logo" style="height: 44px; width: auto; display: block; object-fit: contain;" onerror="this.onerror=null; this.src=\'/static/zebraid_logo_perfect.png\';">
      </a>'''
html = re.sub(nav_pattern, nav_replacement, html, flags=re.DOTALL, count=1)

# Update footer-brand
footer_pattern = r'<div class="footer-brand">.*?</div>'
footer_replacement = f'''<div class="footer-brand">
          <img src="data:image/png;base64,{logo_b64}" alt="ZebraID Logo" style="height: 38px; width: auto; display: block; margin-bottom: 12px; object-fit: contain;" onerror="this.onerror=null; this.src=\'/static/zebraid_logo_perfect.png\';">
        </div>'''
html = re.sub(footer_pattern, footer_replacement, html, flags=re.DOTALL, count=1)

# Update favicon tags with prefers-color-scheme media queries for Light Mode & Dark Mode!
fav_pattern = r'<link rel="icon".*?>\s*<link rel="shortcut icon".*?>(\s*<link rel="alternate icon".*?>)?'
fav_replacement = f'''<link rel="icon" type="image/png" media="(prefers-color-scheme: dark)" href="data:image/png;base64,{white_fav_b64}">
  <link rel="icon" type="image/png" media="(prefers-color-scheme: light)" href="data:image/png;base64,{black_fav_b64}">
  <link rel="shortcut icon" media="(prefers-color-scheme: dark)" href="data:image/png;base64,{white_fav_b64}">
  <link rel="shortcut icon" media="(prefers-color-scheme: light)" href="data:image/png;base64,{black_fav_b64}">
  <link rel="alternate icon" href="/static/black_stripes_favicon.png">'''

html = re.sub(fav_pattern, fav_replacement, html, flags=re.DOTALL, count=1)

html_path.write_text(html)
print("Successfully configured dynamic Light/Dark mode favicons (Black stripes in Light Mode, White stripes in Dark Mode) in index.html!")
