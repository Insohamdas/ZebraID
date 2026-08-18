import re
from pathlib import Path

root = Path(__file__).parent.parent
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

visitor_css = """
    /* ── VISITOR FRIENDLY ANIMATION SYSTEM ── */
    .wispr-hero {
      position: relative;
      min-height: 90vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 110px 24px 80px;
      overflow: hidden;
      background: var(--canvas);
    }

    .wispr-badge {
      font-family: var(--font-body);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--primary);
      margin-bottom: 20px;
      position: relative;
      z-index: 5;
    }

    .wispr-title {
      font-family: var(--font-display);
      font-size: 78px;
      font-weight: 400;
      line-height: 1.05;
      letter-spacing: -2.4px;
      color: var(--ink);
      margin-bottom: 24px;
      position: relative;
      z-index: 5;
    }
    .wispr-title i {
      font-style: italic;
      font-weight: 400;
      color: var(--primary);
    }

    .wispr-sub {
      font-family: var(--font-body);
      font-size: 19px;
      font-weight: 400;
      line-height: 1.6;
      color: var(--body);
      max-width: 600px;
      margin: 0 auto 36px;
      position: relative;
      z-index: 5;
    }

    /* Floating Switcher Pill */
    .wispr-nav-pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: #ffffff;
      border: 1px solid var(--hairline-strong);
      padding: 4px;
      border-radius: 9999px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.04);
      margin-bottom: 28px;
      position: relative;
      z-index: 10;
    }
    .wispr-tab {
      font-family: var(--font-body);
      font-size: 14px;
      font-weight: 500;
      color: var(--body);
      padding: 8px 20px;
      border-radius: 9999px;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .wispr-tab.active {
      background: var(--surface-strong);
      color: var(--ink);
      font-weight: 600;
    }

    /* Elegant SVG Ribbons */
    .wispr-ribbon-svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0;
    }
    .wispr-ribbon-text {
      font-family: var(--font-body);
      font-size: 13px;
      font-weight: 600;
      fill: var(--body);
      opacity: 0.2;
      letter-spacing: 2px;
      text-transform: uppercase;
    }
    .wispr-dark-ribbon-text {
      font-family: var(--font-body);
      font-size: 15px;
      font-weight: 500;
      fill: #ffffff;
      letter-spacing: 1px;
    }

    /* VISITOR FRIENDLY TRANSFORMER CARD */
    .visitor-card {
      position: relative;
      z-index: 10;
      width: 100%;
      max-width: 800px;
      background: #ffffff;
      border: 1px solid var(--hairline-strong);
      border-radius: 24px;
      padding: 24px;
      box-shadow: 0 24px 64px rgba(0, 0, 0, 0.08);
      margin-top: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .visitor-card:hover {
      transform: translateY(-4px);
      box-shadow: 0 32px 80px rgba(0, 0, 0, 0.12);
    }

    .step-block {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
      flex: 1;
    }
    .step-label {
      font-size: 12px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    /* Step 1: Photo */
    .photo-mock {
      position: relative;
      width: 100%;
      height: 80px;
      border-radius: 12px;
      background: repeating-linear-gradient(
        75deg,
        #f8f9fa,
        #f8f9fa 12px,
        #212529 12px,
        #212529 24px,
        #f8f9fa 24px,
        #f8f9fa 32px,
        #212529 32px,
        #212529 48px
      );
      border: 2px solid var(--hairline-strong);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .photo-mock svg {
      color: #fff;
      filter: drop-shadow(0 2px 4px rgba(0,0,0,0.8));
      z-index: 2;
      width: 28px;
      height: 28px;
    }
    .scan-beam {
      position: absolute;
      top: 0;
      left: 0;
      width: 3px;
      height: 100%;
      background: #3b82f6;
      box-shadow: 0 0 16px #3b82f6, 0 0 32px #3b82f6;
      animation: scanSweep 2s ease-in-out infinite alternate;
      z-index: 1;
    }
    @keyframes scanSweep {
      0% { left: -5%; }
      100% { left: 105%; }
    }

    /* Arrows */
    .transform-arrow {
      color: var(--muted-soft);
      animation: pulseArrow 1.5s infinite;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .transform-arrow svg {
      width: 20px;
      height: 20px;
    }
    @keyframes pulseArrow {
      0%, 100% { opacity: 0.3; transform: translateX(0); }
      50% { opacity: 1; transform: translateX(4px); }
    }

    /* Step 2: Code */
    .code-mock {
      width: 100%;
      height: 80px;
      background: #111827;
      border-radius: 12px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      border: 1px solid rgba(0,0,0,0.1);
    }
    .code-stream {
      color: #10b981;
      font-family: "SF Mono", ui-monospace, monospace;
      font-size: 10px;
      word-break: break-all;
      line-height: 1.4;
      font-weight: bold;
    }

    /* Step 3: Match */
    .match-mock {
      width: 100%;
      height: 80px;
      background: #f0fdf4;
      border: 1px solid #bbf7d0;
      border-radius: 12px;
      display: flex;
      align-items: center;
      padding: 0 16px;
      gap: 12px;
    }
    .match-avatar {
      width: 40px;
      height: 40px;
      background: #10b981;
      border-radius: 50%;
      display: flex;
      align-items: center;
      justify-content: center;
      color: white;
      font-weight: bold;
      font-size: 18px;
      flex-shrink: 0;
    }
    .match-info {
      text-align: left;
    }
    .match-name {
      font-size: 14px;
      font-weight: 800;
      color: #065f46;
    }
    .match-conf {
      font-size: 11px;
      font-weight: 700;
      color: #059669;
      margin-top: 2px;
    }
"""

if "/* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── */" in html:
    css_pattern = r'/\* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── \*/.*?(?=</style>)'
    html = re.sub(css_pattern, visitor_css + "\n", html, flags=re.DOTALL)


visitor_hero_html = """
    <!-- Visitor Friendly Hero Section -->
    <section id="overview" class="wispr-hero">
      
      <!-- Elegant SVG Curved Text Ribbon -->
      <svg class="wispr-ribbon-svg" viewBox="0 0 1400 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <path id="archPath" d="M -100,200 C 400,-100 1000,-100 1500,200" />
          <path id="darkRibbonPath" d="M -100,750 C 400,720 900,850 1500,720" />
        </defs>

        <text class="wispr-ribbon-text">
          <textPath href="#archPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="50%" dur="40s" repeatCount="indefinite" />
            ZebraID Biometric Framework · Cross-Population Search · Z-Hash Compression · ZebraID Biometric Framework · Cross-Population Search ·
          </textPath>
        </text>

        <!-- Dark Bottom Ribbon -->
        <path d="M -100,750 C 400,720 900,850 1500,720" fill="none" stroke="#0c0a09" stroke-width="60" stroke-linecap="round" opacity="1" />

        <text class="wispr-dark-ribbon-text">
          <textPath href="#darkRibbonPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="40%" dur="20s" repeatCount="indefinite" />
            Continental scale zebra recognition · Zero raw image transmission · Instant federated shard matching · 99.4% accuracy ·
          </textPath>
        </text>
      </svg>

      <!-- Wispr Header Switcher Pill -->
      <div class="wispr-nav-pill">
        <div class="wispr-tab active">Overview</div>
        <div class="wispr-tab" onclick="openAnalysis()">Federated Search</div>
      </div>

      <div class="wispr-badge">
        AI WILDLIFE BIOMETRICS
      </div>

      <h1 class="wispr-title">
        Don't tag,<br>
        <i>just identify.</i>
      </h1>

      <p class="wispr-sub">
        The AI framework that turns raw field photography into instant, verified individual zebra identities without invasive physical tagging.
      </p>

      <!-- Clear, Explanatory Transformation Card -->
      <div class="visitor-card" onclick="openAnalysis()" title="Click to open Analysis Workspace">
        
        <!-- Step 1: Input -->
        <div class="step-block">
          <div class="photo-mock">
             <i data-lucide="camera"></i>
             <div class="scan-beam"></div>
          </div>
          <span class="step-label">1. Field Photo</span>
        </div>
        
        <div class="transform-arrow">
          <i data-lucide="arrow-right"></i>
        </div>
        
        <!-- Step 2: Code -->
        <div class="step-block">
          <div class="code-mock">
            <div class="code-stream" id="matrixStream">
              01101001 10010110<br>
              01110010 11010100
            </div>
            <div style="color: #3b82f6; font-size: 9px; margin-top:4px; font-family: monospace; font-weight: bold;">
              Z-HASH: e4a91b8f
            </div>
          </div>
          <span class="step-label">2. Biometric Code</span>
        </div>
        
        <div class="transform-arrow">
          <i data-lucide="arrow-right"></i>
        </div>
        
        <!-- Step 3: Match -->
        <div class="step-block">
          <div class="match-mock">
            <div class="match-avatar">Z</div>
            <div class="match-info">
              <div class="match-name">Zebra #492</div>
              <div class="match-conf">99.8% Match</div>
            </div>
          </div>
          <span class="step-label">3. Individual Found</span>
        </div>

      </div>

    </section>
"""

hero_pattern = r'<section id="overview" class="wispr-hero">.*?</section>'
html = re.sub(hero_pattern, visitor_hero_html, html, flags=re.DOTALL, count=1)

# Keep the JS ticker but update the data if necessary, or just use CSS animations. 
# We'll slightly update the JS ticker for the new layout.
js_clean = """
    // Matrix Stream Ticker
    (function initMatrixTicker() {
      const matrixStream = document.getElementById('matrixStream');
      const binSamples = [
        '01101001 10010110\\n01110010 11010100',
        '11010110 01101001\\n10110010 01011101',
        '01011100 11100101\\n01001011 10100110',
      ];
      let idx = 0;
      setInterval(() => {
        if (matrixStream) {
          idx = (idx + 1) % binSamples.length;
          matrixStream.innerHTML = binSamples[idx].replace(/\\n/g, '<br>');
        }
      }, 800);
    })();
"""

js_pattern = r'// Matrix Stream Ticker for Transformer Card.*?(?=\n  </script>)'
if re.search(js_pattern, html, flags=re.DOTALL):
    html = re.sub(js_pattern, js_clean, html, flags=re.DOTALL)

html_path.write_text(html)
print("Successfully made animation visitor-friendly!")
