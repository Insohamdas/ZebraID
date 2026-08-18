import re
from pathlib import Path

root = Path(__file__).parent.parent
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

# 1. Update CSS to be cleaner
perfect_css = """
    /* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── */
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
      font-weight: 600;
      letter-spacing: 1.6px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 20px;
      position: relative;
      z-index: 5;
    }

    .wispr-title {
      font-family: var(--font-display);
      font-size: 78px;
      font-weight: 300;
      line-height: 1.02;
      letter-spacing: -2.4px;
      color: var(--ink);
      margin-bottom: 20px;
      position: relative;
      z-index: 5;
    }
    .wispr-title i {
      font-style: italic;
      font-weight: 400;
    }

    .wispr-sub {
      font-family: var(--font-body);
      font-size: 18px;
      font-weight: 400;
      line-height: 1.55;
      letter-spacing: 0.16px;
      color: var(--body);
      max-width: 580px;
      margin: 0 auto 32px;
      position: relative;
      z-index: 5;
    }

    /* Floating Switcher Pill (Wispr Header Style) */
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

    /* SVG Curved Text Ribbon - Pushed to Background */
    .wispr-ribbon-svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 0; /* Behind everything */
    }
    .wispr-ribbon-text {
      font-family: var(--font-body);
      font-size: 15px;
      font-weight: 500;
      fill: var(--body);
      opacity: 0.25; /* Softer, elegant opacity */
      letter-spacing: 1px;
    }
    .wispr-dark-ribbon-text {
      font-family: var(--font-body);
      font-size: 15px;
      font-weight: 500;
      fill: #ffffff;
      letter-spacing: 1px;
    }

    /* PRO VISUAL TRANSFORMER CARD (HERO CENTERPIECE) */
    .pro-transformer-card {
      position: relative;
      z-index: 10;
      width: 100%;
      max-width: 680px;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(20px);
      border: 1px solid var(--hairline-strong);
      border-radius: 20px; /* Sleeker rounding */
      padding: 24px 32px;
      box-shadow: 0 12px 48px rgba(0, 0, 0, 0.06);
      margin-top: 36px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .pro-transformer-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 16px 56px rgba(0, 0, 0, 0.1);
    }

    .stripe-preview-box {
      position: relative;
      width: 130px;
      height: 90px;
      border-radius: 12px;
      background: #000;
      overflow: hidden;
      border: 1px solid var(--hairline-strong);
      flex-shrink: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .stripe-pattern-canvas {
      width: 120%;
      height: 120%;
      background-image: url('/static/logo.png');
      background-size: cover;
      background-position: center;
      filter: grayscale(100%) contrast(150%);
      opacity: 0.8;
    }
    .scan-beam {
      position: absolute;
      top: 0;
      left: 0;
      width: 2px;
      height: 100%;
      background: #10b981; /* Premium Green */
      box-shadow: 0 0 12px #10b981, 0 0 24px #10b981;
      animation: scanSweep 2.5s ease-in-out infinite alternate;
      z-index: 2;
    }
    @keyframes scanSweep {
      0% { left: -10%; }
      100% { left: 110%; }
    }

    .transform-arrow {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1px;
      text-transform: uppercase;
    }
    .transform-pulse-dots {
      display: flex;
      gap: 6px;
    }
    .pulse-dot {
      width: 5px;
      height: 5px;
      border-radius: 50%;
      background: #10b981;
      animation: dotPulse 1.2s ease-in-out infinite alternate;
    }
    .pulse-dot:nth-child(1) { animation-delay: 0s; }
    .pulse-dot:nth-child(2) { animation-delay: 0.3s; }
    .pulse-dot:nth-child(3) { animation-delay: 0.6s; }

    @keyframes dotPulse {
      0% { opacity: 0.1; transform: scale(0.8); }
      100% { opacity: 1; transform: scale(1.4); }
    }

    .code-matrix-box {
      flex: 1;
      text-align: left;
      background: #111111; /* Sleek black terminal */
      border-radius: 12px;
      padding: 16px 20px;
      color: #ffffff;
      font-family: "SF Mono", ui-monospace, monospace;
      font-size: 13px;
      overflow: hidden;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .code-header {
      font-size: 10px;
      color: #888;
      letter-spacing: 1px;
      text-transform: uppercase;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .match-badge {
      color: #10b981;
      background: rgba(16, 185, 129, 0.1);
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
    }
    .code-stream {
      color: #10b981; /* Emerald Green */
      word-break: break-all;
      line-height: 1.5;
      font-weight: 500;
    }
    .code-hex {
      color: #3b82f6; /* Blue */
      margin-top: 6px;
      font-size: 12px;
    }
"""

if "/* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── */" in html:
    pattern_css = r'/\* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── \*/.*?\n\s*\}'
    # Need to match the whole block. Better to just slice.
    
# Let's cleanly replace the style block content
import re

css_pattern = r'/\* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── \*/.*?(?=</style>)'
if re.search(css_pattern, html, flags=re.DOTALL):
    html = re.sub(css_pattern, perfect_css + "\n", html, flags=re.DOTALL)


perfect_hero_html = """
    <!-- World-Class Wispr Flow Style Hero Section -->
    <section id="overview" class="wispr-hero">
      
      <!-- Elegant SVG Curved Text Ribbon - Simplified paths, no intersection with main text -->
      <svg class="wispr-ribbon-svg" viewBox="0 0 1400 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <!-- Elegant outer arch -->
          <path id="archPath" d="M -100,200 C 400,-100 1000,-100 1500,200" />
          <path id="darkRibbonPath" d="M -100,750 C 400,720 900,850 1500,720" />
        </defs>

        <!-- Light Ribbon Text (Background Arch) -->
        <text class="wispr-ribbon-text">
          <textPath href="#archPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="50%" dur="30s" repeatCount="indefinite" />
            ZebraID Biometric Framework · 512d Dense MegaDescriptor Embedding · Z-Hash 256b Binary Compression · Privacy-Preserving Federated Search · ZebraID Biometric Framework · 512d Dense MegaDescriptor Embedding ·
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
        <div class="wispr-tab active">Stripe Biometrics</div>
        <div class="wispr-tab" onclick="openAnalysis()">Z-Hash Code</div>
      </div>

      <div class="wispr-badge">
        STRIPE PATTERN ➔ NEURAL ENCODER ➔ Z-HASH CODE
      </div>

      <h1 class="wispr-title">
        Don't search,<br>
        <i>just convert.</i>
      </h1>

      <p class="wispr-sub">
        The stripe-to-code AI framework that transforms raw zebra field photography into 256-bit Z-Hash binary codes and instant cross-population match results.
      </p>

      <!-- Professional Interactive Visual Transformer Card (Hero Centerpiece) -->
      <div class="pro-transformer-card" onclick="openAnalysis()" title="Click to open Analysis Workspace">
        
        <!-- Left: Stripe Crop + Laser Beam -->
        <div class="stripe-preview-box">
          <div class="stripe-pattern-canvas"></div>
          <div class="scan-beam"></div>
        </div>

        <!-- Middle: Transformation Stream -->
        <div class="transform-arrow">
          <span>Encode</span>
          <div class="transform-pulse-dots">
            <div class="pulse-dot"></div>
            <div class="pulse-dot"></div>
            <div class="pulse-dot"></div>
          </div>
        </div>

        <!-- Right: Code Matrix Output -->
        <div class="code-matrix-box">
          <div class="code-header">
            <span>Z-HASH 256-BIT STREAM</span>
            <span class="match-badge">99.8% MATCH</span>
          </div>
          <div class="code-stream" id="matrixStream">
            01101001 10010110 01110010 11010100
          </div>
          <div class="code-hex" id="matrixHex">
            HEX: [e4a91b8f3c7d2e0f4a8b]
          </div>
        </div>

      </div>

    </section>
"""

hero_pattern = r'<section id="overview" class="wispr-hero">.*?</section>'
html = re.sub(hero_pattern, perfect_hero_html, html, flags=re.DOTALL, count=1)

# Clean up JS by removing the old noisy Canvas
js_clean = """
    // Matrix Stream Ticker for Transformer Card
    (function initMatrixTicker() {
      const matrixStream = document.getElementById('matrixStream');
      const matrixHex = document.getElementById('matrixHex');

      const binSamples = [
        '01101001 10010110 01110010 11010100',
        '11010110 01101001 10110010 01011101',
        '01011100 11100101 01001011 10100110',
      ];
      const hexSamples = [
        'HEX: [e4a91b8f3c7d2e0f4a8b]',
        'HEX: [9c1d2e3f4a5b6c7d8e9f]',
        'HEX: [1b8f3c7d2e0f4a8b9c1d]',
      ];

      let idx = 0;
      setInterval(() => {
        if (matrixStream && matrixHex) {
          idx = (idx + 1) % binSamples.length;
          matrixStream.textContent = binSamples[idx];
          matrixHex.textContent = hexSamples[idx];
        }
      }, 1400);
    })();
"""

js_pattern = r'// 60fps GPU Canvas Animation for Stripe Particles & Code Flow.*?(?=\n  </script>)'
if re.search(js_pattern, html, flags=re.DOTALL):
    html = re.sub(js_pattern, js_clean, html, flags=re.DOTALL)

html_path.write_text(html)
print("Successfully applied perfection touches to the animation!")
