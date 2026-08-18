import base64
import re
from pathlib import Path

root = Path(__file__).parent.parent
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

# World-Class Professional Animation CSS
pro_css = """
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

    /* 60fps Flow Canvas */
    #heroCanvas {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 2;
    }

    /* SVG Curved Text Ribbon */
    .wispr-ribbon-svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 3;
    }
    .wispr-ribbon-text {
      font-family: var(--font-body);
      font-size: 14px;
      font-weight: 500;
      fill: var(--body);
      opacity: 0.55;
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
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(20px);
      border: 1px solid var(--hairline-strong);
      border-radius: var(--rounded-xxl);
      padding: 24px 32px;
      box-shadow: 0 20px 48px rgba(12, 10, 9, 0.08);
      margin-top: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
    }

    .stripe-preview-box {
      position: relative;
      width: 130px;
      height: 90px;
      border-radius: var(--rounded-lg);
      background: #111;
      overflow: hidden;
      border: 1px solid var(--hairline);
      flex-shrink: 0;
    }
    .stripe-pattern-canvas {
      width: 100%;
      height: 100%;
      background: repeating-linear-gradient(
        -45deg,
        #111,
        #111 8px,
        #fff 8px,
        #fff 16px
      );
    }
    .scan-beam {
      position: absolute;
      top: 0;
      left: 0;
      width: 4px;
      height: 100%;
      background: var(--semantic-success);
      box-shadow: 0 0 12px var(--semantic-success), 0 0 24px var(--semantic-success);
      animation: scanSweep 2s ease-in-out infinite alternate;
    }
    @keyframes scanSweep {
      0% { left: 0%; }
      100% { left: 96%; }
    }

    .transform-arrow {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.5px;
    }
    .transform-pulse-dots {
      display: flex;
      gap: 4px;
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 9999px;
      background: var(--semantic-success);
      animation: dotPulse 1.2s ease-in-out infinite alternate;
    }
    .pulse-dot:nth-child(1) { animation-delay: 0s; }
    .pulse-dot:nth-child(2) { animation-delay: 0.3s; }
    .pulse-dot:nth-child(3) { animation-delay: 0.6s; }

    @keyframes dotPulse {
      0% { opacity: 0.2; transform: scale(0.8); }
      100% { opacity: 1; transform: scale(1.3); }
    }

    .code-matrix-box {
      flex: 1;
      text-align: left;
      background: var(--surface-dark);
      border-radius: var(--rounded-lg);
      padding: 14px 18px;
      color: #ffffff;
      font-family: monospace;
      font-size: 13px;
      overflow: hidden;
    }
    .code-header {
      font-size: 11px;
      color: var(--muted-soft);
      letter-spacing: 0.8px;
      text-transform: uppercase;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    .code-stream {
      color: #4ade80; /* Code Green */
      word-break: break-all;
      line-height: 1.4;
    }
    .code-hex {
      color: #60a5fa; /* Code Blue */
    }
"""

if "/* ── WORLD-CLASS STRIPE-TO-CODE ANIMATION SYSTEM ── */" not in html:
    if "/* ── STRIPE TO CODE TRANSFORMATION ANIMATION ── */" in html:
        pattern_css = r'/\* ── STRIPE TO CODE TRANSFORMATION ANIMATION ── \*/.*?\n\s*\}'
        html = re.sub(pattern_css, pro_css, html, flags=re.DOTALL)
    elif "/* ── WISPR FLOW HERO ANIMATION ── */" in html:
        pattern_css = r'/\* ── WISPR FLOW HERO ANIMATION ── \*/.*?\n\s*\}'
        html = re.sub(pattern_css, pro_css, html, flags=re.DOTALL)
    else:
        html = html.replace("</style>", pro_css + "\n</style>")

# Professional Wispr Flow Style Hero HTML
pro_hero_html = """
    <!-- World-Class Wispr Flow Style Hero Section -->
    <section id="overview" class="wispr-hero">
      
      <!-- 60fps GPU Canvas for Particle & Stripe Flow -->
      <canvas id="heroCanvas"></canvas>

      <!-- SVG Curved Text Flowing Ribbon -->
      <svg class="wispr-ribbon-svg" viewBox="0 0 1400 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <path id="spiralPath" d="M -200,650 C 150,550 350,250 200,120 C 50,-10 -150,150 50,450 C 250,750 750,700 1200,450 C 1450,300 1600,600 1800,700" />
          <path id="darkRibbonPath" d="M -100,720 C 400,680 900,820 1500,650" />
        </defs>

        <!-- Light Ribbon Text -->
        <text class="wispr-ribbon-text">
          <textPath href="#spiralPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="50%" dur="24s" repeatCount="indefinite" />
            ZebraID Biometric Framework · 512d Dense MegaDescriptor Embedding · Z-Hash 256b Binary Compression · Privacy-Preserving Federated Search · ZebraID Biometric Framework · 512d Dense MegaDescriptor Embedding ·
          </textPath>
        </text>

        <!-- Dark Bottom Ribbon -->
        <path d="M -100,720 C 400,680 900,820 1500,650" fill="none" stroke="#0c0a09" stroke-width="48" stroke-linecap="round" opacity="0.95" />

        <text class="wispr-dark-ribbon-text">
          <textPath href="#darkRibbonPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="40%" dur="18s" repeatCount="indefinite" />
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

      <div class="hero-cta">
        <button class="btn-primary" onclick="openAnalysis()" style="height: 44px; padding: 12px 28px; font-size: 16px;">
          Identify Sighting <i data-lucide="sparkles" style="width:18px;"></i>
        </button>
      </div>

      <!-- Professional Interactive Visual Transformer Card (Hero Centerpiece) -->
      <div class="pro-transformer-card" onclick="openAnalysis()" title="Click to open Analysis Workspace">
        
        <!-- Left: Stripe Crop + Laser Beam -->
        <div class="stripe-preview-box">
          <div class="stripe-pattern-canvas"></div>
          <div class="scan-beam"></div>
        </div>

        <!-- Middle: Transformation Stream -->
        <div class="transform-arrow">
          <span>ENCODE</span>
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
            <span style="color:#4ade80;">99.8% MATCH</span>
          </div>
          <div class="code-stream" id="matrixStream">
            01101001 10010110 01110010 11010100
          </div>
          <div class="code-hex" id="matrixHex" style="margin-top:4px;">
            HEX: [e4a91b8f3c7d2e0f4a8b]
          </div>
        </div>

      </div>

    </section>
"""

# Replace hero section
hero_pattern = r'<section id="overview" class="wispr-hero">.*?</section>'
html = re.sub(hero_pattern, pro_hero_html, html, flags=re.DOTALL, count=1)

# Add 60fps Canvas Particle Animation JS
pro_js = """
    // 60fps GPU Canvas Animation for Stripe Particles & Code Flow
    (function initHeroCanvas() {
      const canvas = document.getElementById('heroCanvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d');

      function resize() {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;
      }
      resize();
      window.addEventListener('resize', resize);

      // Binary particles floating along curves
      const particles = [];
      const numParticles = 35;

      for (let i = 0; i < numParticles; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: 0.3 + Math.random() * 0.6,
          vy: (Math.random() - 0.5) * 0.4,
          size: 11 + Math.random() * 5,
          text: Math.random() > 0.5 ? '0' : '1',
          opacity: 0.15 + Math.random() * 0.35,
        });
      }

      function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
          p.x += p.vx;
          p.y += p.vy;
          if (p.x > canvas.width + 20) p.x = -20;
          if (p.y < -20) p.y = canvas.height + 20;
          if (p.y > canvas.height + 20) p.y = -20;

          ctx.font = `${p.size}px monospace`;
          ctx.fillStyle = `rgba(16, 185, 129, ${p.opacity})`;
          ctx.fillText(p.text, p.x, p.y);
        });

        requestAnimationFrame(animate);
      }
      animate();
    })();

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

# Replace old js ticker if present or append
if "// 60fps GPU Canvas Animation for Stripe Particles & Code Flow" not in html:
    if "// Live Stripe -> Binary -> Hex Code Morphing Ticker" in html:
        pattern_js = r'// Live Stripe -> Binary -> Hex Code Morphing Ticker.*?\n\s*\}\, 900\);'
        html = re.sub(pattern_js, pro_js, html, flags=re.DOTALL)
    else:
        html = html.replace("</script>", pro_js + "\n  </script>")

html_path.write_text(html)
print("Successfully installed World-Class 60fps Visual Transformer & Canvas Animation into index.html!")
