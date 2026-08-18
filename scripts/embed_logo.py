import base64
from pathlib import Path

root = Path(__file__).parent.parent
logo_b64_path = root / "demo" / "static" / "logo_b64.txt"
fav_b64_path = root / "demo" / "static" / "favicon_b64.txt"
html_path = root / "demo" / "templates" / "index.html"

logo_b64 = logo_b64_path.read_text().strip()
fav_b64 = fav_b64_path.read_text().strip()
html = html_path.read_text()

nav_old = '<a href="#" class="nav-brand">\n        <img src="/static/logo.png" alt="ZebraID Logo" style="height: 38px; width: auto; display: block; object-fit: contain;">\n      </a>'
nav_new = f'<a href="#" class="nav-brand">\n        <img src="data:image/png;base64,{logo_b64}" alt="ZebraID Logo" style="height: 38px; width: auto; display: block; object-fit: contain;" onerror="this.onerror=null; this.src=\'/static/logo.png\';">\n      </a>'

footer_old = '<div class="footer-brand">\n          <img src="/static/logo.png" alt="ZebraID Logo" style="height: 34px; width: auto; display: block; margin-bottom: 12px; object-fit: contain;">\n        </div>'
footer_new = f'<div class="footer-brand">\n          <img src="data:image/png;base64,{logo_b64}" alt="ZebraID Logo" style="height: 34px; width: auto; display: block; margin-bottom: 12px; object-fit: contain;" onerror="this.onerror=null; this.src=\'/static/logo.png\';">\n        </div>'

fav_old = '  <link rel="icon" type="image/png" href="/static/favicon.png">\n  <link rel="shortcut icon" href="/static/favicon.png">'
fav_new = f'  <link rel="icon" type="image/png" href="data:image/png;base64,{fav_b64}">\n  <link rel="shortcut icon" href="data:image/png;base64,{fav_b64}">\n  <link rel="alternate icon" href="/static/favicon.png">'

if nav_old in html:
    html = html.replace(nav_old, nav_new)
    print("Replaced nav_brand image src successfully")
else:
    print("Warning: nav_old not found in html")

if footer_old in html:
    html = html.replace(footer_old, footer_new)
    print("Replaced footer_brand image src successfully")
else:
    print("Warning: footer_old not found in html")

if fav_old in html:
    html = html.replace(fav_old, fav_new)
    print("Replaced fav_old in html successfully")
else:
    print("Warning: fav_old not found in html")

html_path.write_text(html)
print("Updated demo/templates/index.html with inline base64 images successfully!")
