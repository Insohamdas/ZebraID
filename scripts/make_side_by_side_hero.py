import re
from pathlib import Path

root = Path(__file__).parent.parent
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

sbs_css = """
    /* ── TWO-COLUMN HERO LAYOUT ── */
    .hero-sbs {
      position: relative;
      min-height: 90vh;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 120px 5% 80px;
      background: var(--canvas);
      overflow: hidden;
      gap: 40px;
    }

    /* Background abstract gradient blobs */
    .hero-sbs::before {
      content: '';
      position: absolute;
      top: -20%;
      left: -10%;
      width: 60%;
      height: 60%;
      background: radial-gradient(circle, rgba(16,185,129,0.05) 0%, rgba(255,255,255,0) 70%);
      z-index: 0;
    }
    .hero-sbs::after {
      content: '';
      position: absolute;
      bottom: -20%;
      right: -10%;
      width: 50%;
      height: 50%;
      background: radial-gradient(circle, rgba(59,130,246,0.04) 0%, rgba(255,255,255,0) 70%);
      z-index: 0;
    }

    .hero-content {
      position: relative;
      z-index: 10;
      flex: 1;
      max-width: 560px;
      text-align: left;
    }

    .hero-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: var(--font-body);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--primary);
      margin-bottom: 24px;
      background: rgba(16, 185, 129, 0.1);
      padding: 6px 14px;
      border-radius: 9999px;
    }

    .hero-title {
      font-family: var(--font-display);
      font-size: 72px;
      font-weight: 400;
      line-height: 1.05;
      letter-spacing: -2px;
      color: var(--ink);
      margin-bottom: 24px;
    }
    .hero-title i {
      font-style: italic;
      color: var(--primary);
    }

    .hero-sub {
      font-family: var(--font-body);
      font-size: 18px;
      line-height: 1.6;
      color: var(--body);
      margin-bottom: 40px;
    }

    .hero-actions {
      display: flex;
      gap: 16px;
      margin-bottom: 48px;
    }
    .hero-actions .btn-primary {
      padding: 14px 28px;
      font-size: 16px;
      height: auto;
      border-radius: 12px;
    }
    .hero-actions .btn-secondary {
      padding: 14px 28px;
      font-size: 16px;
      background: #ffffff;
      color: var(--ink);
      border: 1px solid var(--hairline-strong);
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .hero-actions .btn-secondary:hover {
      background: var(--surface-strong);
    }

    .hero-stats-row {
      display: flex;
      gap: 32px;
      border-top: 1px solid var(--hairline);
      padding-top: 24px;
    }
    .stat-item {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .stat-item strong {
      font-family: var(--font-display);
      font-size: 28px;
      font-weight: 400;
      color: var(--ink);
      letter-spacing: -1px;
    }
    .stat-item span {
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--muted);
    }

    /* Visual Side */
    .hero-visual {
      position: relative;
      z-index: 10;
      flex: 1;
      display: flex;
      justify-content: center;
      align-items: center;
      perspective: 1000px; /* For 3D floating effect */
    }

    .ui-stack {
      position: relative;
      width: 400px;
      height: 400px;
      transform-style: preserve-3d;
      transform: rotateX(15deg) rotateY(-15deg);
      transition: transform 0.5s ease;
    }
    .hero-visual:hover .ui-stack {
      transform: rotateX(5deg) rotateY(-5deg);
    }

    .ui-layer {
      position: absolute;
      width: 100%;
      background: rgba(255, 255, 255, 0.8);
      backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.4);
      border-radius: 20px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.08);
      padding: 24px;
      transition: all 0.5s ease;
    }

    /* Top Layer: Photo Scan */
    .layer-1 {
      top: 0;
      left: 0;
      height: 200px;
      transform: translateZ(60px);
      display: flex;
      flex-direction: column;
      justify-content: center;
      overflow: hidden;
      border: 1px solid var(--hairline-strong);
    }
    .photo-stripe-bg {
      position: absolute;
      top: 0; left: 0; right: 0; bottom: 0;
      background: repeating-linear-gradient(
        -45deg,
        #f8f9fa,
        #f8f9fa 15px,
        #212529 15px,
        #212529 30px
      );
      opacity: 0.1;
      z-index: 0;
    }
    .scan-beam-h {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 3px;
      background: #10b981;
      box-shadow: 0 0 20px #10b981;
      animation: scanDown 3s ease-in-out infinite alternate;
      z-index: 2;
    }
    @keyframes scanDown {
      0% { top: 0%; }
      100% { top: 100%; }
    }
    .layer-1-content {
      position: relative;
      z-index: 1;
      text-align: center;
    }

    /* Middle Layer: Code */
    .layer-2 {
      top: 100px;
      left: 40px;
      height: 120px;
      background: #111;
      color: #10b981;
      transform: translateZ(30px);
      border: 1px solid rgba(255,255,255,0.1);
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .code-grid {
      font-family: monospace;
      font-size: 12px;
      line-height: 1.5;
      opacity: 0.8;
      word-break: break-all;
    }

    /* Bottom Layer: Match */
    .layer-3 {
      top: 240px;
      left: 0;
      height: 100px;
      background: #ffffff;
      transform: translateZ(90px);
      display: flex;
      align-items: center;
      gap: 16px;
      border: 1px solid var(--hairline-strong);
    }
    .match-icon {
      width: 48px;
      height: 48px;
      background: #ecfdf5;
      color: #10b981;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .match-text {
      text-align: left;
    }
    .match-text h4 {
      margin: 0;
      font-size: 18px;
      color: var(--ink);
    }
    .match-text p {
      margin: 4px 0 0;
      font-size: 13px;
      color: #10b981;
      font-weight: 600;
    }
"""

if "/* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── */" in html:
    css_pattern = r'/\* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── \*/.*?(?=</style>)'
    html = re.sub(css_pattern, sbs_css + "\n", html, flags=re.DOTALL)
elif "/* ── VISITOR FRIENDLY ANIMATION SYSTEM ── */" in html:
    css_pattern = r'/\* ── VISITOR FRIENDLY ANIMATION SYSTEM ── \*/.*?(?=</style>)'
    html = re.sub(css_pattern, sbs_css + "\n", html, flags=re.DOTALL)


sbs_hero_html = """
    <!-- Two-Column Side-by-Side Hero Section -->
    <section id="overview" class="hero-sbs">
      
      <!-- Left Content -->
      <div class="hero-content">
        <div class="hero-badge">
          <i data-lucide="scan-line" style="width:14px;"></i> BIOMETRIC AI FRAMEWORK
        </div>
        
        <h1 class="hero-title">
          Identify any zebra,<br>
          <i>instantly.</i>
        </h1>
        
        <p class="hero-sub">
          The open-source AI platform that turns raw field photography into 256-bit Z-Hash codes for instant, privacy-preserving population matching.
        </p>
        
        <div class="hero-actions">
          <button class="btn-primary" onclick="openAnalysis()">
            Identify Sighting <i data-lucide="arrow-right"></i>
          </button>
          <button class="btn-secondary" onclick="document.getElementById('architecture').scrollIntoView({behavior: 'smooth'})">
            How it Works
          </button>
        </div>
        
        <div class="hero-stats-row">
          <div class="stat-item">
            <strong>99.4%</strong>
            <span>mAP@1 Accuracy</span>
          </div>
          <div class="stat-item">
            <strong>256b</strong>
            <span>Z-Hash Code</span>
          </div>
          <div class="stat-item">
            <strong>&lt;5ms</strong>
            <span>Query Latency</span>
          </div>
        </div>
      </div>

      <!-- Right Visual (3D Floating Stack) -->
      <div class="hero-visual">
        <div class="ui-stack">
          
          <!-- Middle Layer: Code Processing -->
          <div class="ui-layer layer-2">
            <div style="font-size: 10px; color: #888; margin-bottom: 8px;">NEURAL ENCODER OUTPUT</div>
            <div class="code-grid" id="matrixStream">
              01101001 10010110 01110010 11010100<br>
              10100110 01011001 11001110 00110101
            </div>
          </div>
          
          <!-- Top Layer: Image Scanning -->
          <div class="ui-layer layer-1">
            <div class="photo-stripe-bg"></div>
            <div class="scan-beam-h"></div>
            <div class="layer-1-content">
              <i data-lucide="camera" style="width:32px; height:32px; color:var(--ink); margin-bottom:8px;"></i>
              <div style="font-size:12px; font-weight:700; letter-spacing:1px; color:var(--ink);">FIELD PHOTO UPLOAD</div>
            </div>
          </div>

          <!-- Bottom Layer: Match Result -->
          <div class="ui-layer layer-3">
            <div class="match-icon">
              <i data-lucide="check-circle" style="width:24px; height:24px;"></i>
            </div>
            <div class="match-text">
              <h4>Zebra #Z-492</h4>
              <p>CONFIDENCE: 99.8%</p>
            </div>
          </div>

        </div>
      </div>

    </section>
"""

hero_pattern = r'<section id="overview" class="wispr-hero">.*?</section>'
if re.search(hero_pattern, html, flags=re.DOTALL):
    html = re.sub(hero_pattern, sbs_hero_html, html, flags=re.DOTALL, count=1)

html_path.write_text(html)
print("Successfully applied Side-by-Side layout!")
