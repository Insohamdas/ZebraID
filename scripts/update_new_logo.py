import base64
import shutil
import re
from pathlib import Path
from PIL import Image

root = Path(__file__).parent.parent
static_dir = root / "demo" / "static"
static_dir.mkdir(exist_ok=True)

logo_src = root / "zebraId_logo.png"
fav_src = root / "zabraid_fabicon.png"

# Copy logo PNG to static
shutil.copy(logo_src, static_dir / "zebraid_logo.png")
shutil.copy(logo_src, static_dir / "logo.png")

# Resize favicon to high-res 256x256 PNG for fast loading
fav_img = Image.open(fav_src)
fav_256_path = static_dir / "zebraid_favicon.png"
fav_img.resize((256, 256), Image.Resampling.LANCZOS).save(fav_256_path, format="PNG")
fav_img.resize((256, 256), Image.Resampling.LANCZOS).save(static_dir / "favicon.png", format="PNG")

# Encode to base64
logo_b64 = base64.b64encode(logo_src.read_bytes()).decode('utf-8')
fav_b64 = base64.b64encode(fav_256_path.read_bytes()).decode('utf-8')

html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

# Update nav-brand
nav_pattern = r'<a href="#" class="nav-brand">.*?</a>'
nav_replacement = f'''<a href="#" class="nav-brand">
        <img src="data:image/png;base64,{logo_b64}" alt="ZebraID Logo" style="height: 44px; width: auto; display: block; object-fit: contain;" onerror="this.onerror=null; this.src=\'/static/zebraid_logo.png\';">
      </a>'''

html = re.sub(nav_pattern, nav_replacement, html, flags=re.DOTALL, count=1)

# Update footer-brand
footer_pattern = r'<div class="footer-brand">.*?</div>'
footer_replacement = f'''<div class="footer-brand">
          <img src="data:image/png;base64,{logo_b64}" alt="ZebraID Logo" style="height: 38px; width: auto; display: block; margin-bottom: 12px; object-fit: contain;" onerror="this.onerror=null; this.src=\'/static/zebraid_logo.png\';">
        </div>'''

html = re.sub(footer_pattern, footer_replacement, html, flags=re.DOTALL, count=1)

# Update favicon tags
fav_pattern = r'<link rel="icon".*?>\s*<link rel="shortcut icon".*?>(\s*<link rel="alternate icon".*?>)?'
fav_replacement = f'''<link rel="icon" type="image/png" href="data:image/png;base64,{fav_b64}">
  <link rel="shortcut icon" href="data:image/png;base64,{fav_b64}">
  <link rel="alternate icon" href="/static/zebraid_favicon.png">'''

html = re.sub(fav_pattern, fav_replacement, html, flags=re.DOTALL, count=1)

html_path.write_text(html)
print("Successfully processed and embedded zebraId_logo.png and zabraid_fabicon.png (256x256) into index.html!")
