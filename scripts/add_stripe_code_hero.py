import base64
import re
from pathlib import Path

root = Path(__file__).parent.parent
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

# CSS for Stripe-to-Code Transformation Animation
stripe_code_css = """
    /* ── STRIPE TO CODE TRANSFORMATION ANIMATION ── */
    .wispr-hero {
      position: relative;
      min-height: 88vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 120px 24px 80px;
      overflow: hidden;
      background: var(--canvas);
    }

    .wispr-badge {
      font-family: var(--font-body);
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 24px;
      position: relative;
      z-index: 5;
    }

    .wispr-title {
      font-family: var(--font-display);
      font-size: 82px;
      font-weight: 300;
      line-height: 1.02;
      letter-spacing: -2.4px;
      color: var(--ink);
      margin-bottom: 24px;
      position: relative;
      z-index: 5;
    }
    .wispr-title i {
      font-style: italic;
      font-weight: 400;
    }

    .wispr-sub {
      font-family: var(--font-body);
      font-size: 19px;
      font-weight: 400;
      line-height: 1.55;
      letter-spacing: 0.16px;
      color: var(--body);
      max-width: 600px;
      margin: 0 auto 36px;
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
      margin-bottom: 32px;
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

    /* Animated Ribbon SVG Paths */
    .wispr-ribbon-svg {
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 2;
    }
    .wispr-ribbon-text {
      font-family: monospace, var(--font-body);
      font-size: 14px;
      font-weight: 600;
      fill: var(--body);
      opacity: 0.65;
      letter-spacing: 1px;
    }
    .wispr-dark-ribbon-text {
      font-family: monospace, var(--font-body);
      font-size: 15px;
      font-weight: 600;
      fill: #ffffff;
      letter-spacing: 1px;
    }

    /* Floating Capsule Widget (Stripe -> Code Transformer) */
    .flow-capsule {
      display: inline-flex;
      align-items: center;
      gap: 16px;
      background: #ffffff;
      border: 1.5px solid var(--ink);
      padding: 12px 28px;
      border-radius: 9999px;
      box-shadow: 0 16px 40px rgba(12, 10, 9, 0.1);
      cursor: pointer;
      position: relative;
      z-index: 10;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .flow-capsule:hover {
      transform: scale(1.03);
      box-shadow: 0 20px 48px rgba(12, 10, 9, 0.15);
    }
    
    .stripe-code-bits {
      display: flex;
      align-items: center;
      gap: 4px;
      font-family: monospace;
      font-size: 16px;
      font-weight: 700;
      color: var(--ink);
    }
    .bit-char {
      display: inline-block;
      min-width: 14px;
      text-align: center;
      transition: all 0.3s ease;
    }
"""

if "/* ── STRIPE TO CODE TRANSFORMATION ANIMATION ── */" not in html:
    # Replace previous wispr css if present or append
    if "/* ── WISPR FLOW HERO ANIMATION ── */" in html:
        pattern_css = r'/\* ── WISPR FLOW HERO ANIMATION ── \*/.*?\n\s*\}'
        html = re.sub(pattern_css, stripe_code_css, html, flags=re.DOTALL)
    else:
        html = html.replace("</style>", stripe_code_css + "\n</style>")

# Create Stripe -> Z-Hash Code Transformation Hero HTML Block
stripe_code_hero_html = """
    <!-- Wispr Flow Style Stripe -> Z-Hash Code Hero Band -->
    <section id="overview" class="wispr-hero">
      
      <!-- SVG Curved Text & Code Flowing Ribbon -->
      <svg class="wispr-ribbon-svg" viewBox="0 0 1400 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <!-- Spiral Light Curve Path -->
          <path id="spiralPath" d="M -200,650 C 150,550 350,250 200,120 C 50,-10 -150,150 50,450 C 250,750 750,700 1200,450 C 1450,300 1600,600 1800,700" />
          
          <!-- Bottom Dark Ribbon Curve Path -->
          <path id="darkRibbonPath" d="M -100,720 C 400,680 900,820 1500,650" />
        </defs>

        <!-- Light Spiral Ribbon: Stripe Patterns ▌│█║ ➔ Neural Vectors ➔ Z-Hash 256b ➔ Binary/Hex Code -->
        <text class="wispr-ribbon-text">
          <textPath href="#spiralPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="50%" dur="22s" repeatCount="indefinite" />
            STRIPE PATTERN ▌│█║▌│║▌║ ──▶ NEURAL ENCODER ──▶ Z-HASH 256b 01101001 10010110 01110010 ──▶ HEX CODE [e4a91b8f3c7d2e0f] ──▶ SHARD MATCH ──▶ STRIPE PATTERN ▌│█║▌│║▌║ ──▶ NEURAL ENCODER ──▶ Z-HASH 256b 01101001 10010110 ──▶
          </textPath>
        </text>

        <!-- Dark Ribbon Strip Background -->
        <path d="M -100,720 C 400,680 900,820 1500,650" fill="none" stroke="#0c0a09" stroke-width="48" stroke-linecap="round" opacity="0.95" />

        <!-- Dark Ribbon Text Stream -->
        <text class="wispr-dark-ribbon-text">
          <textPath href="#darkRibbonPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="40%" dur="16s" repeatCount="indefinite" />
            010110100101 ▌│█║ Z-HASH BINARY [e4a91b8f3c7d2e0f] ──▶ CONTINENTAL SHARD SEARCH ──▶ MATCH CONFIDENCE 99.4% ──▶ 01101001 ▌│█║
          </textPath>
        </text>
      </svg>

      <!-- Wispr Header Switcher Pill -->
      <div class="wispr-nav-pill">
        <div class="wispr-tab active">Stripe Biometrics</div>
        <div class="wispr-tab" onclick="openAnalysis()">Z-Hash Code</div>
      </div>

      <div class="wispr-badge">
        STRIPE PATTERN ➔ Z-HASH CODE ENCODER
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
          Convert Stripe to Code <i data-lucide="binary" style="width:18px;"></i>
        </button>
      </div>

      <!-- Floating Morphing Stripe -> Code Capsule (Interactive Transformer) -->
      <div class="flow-capsule" onclick="openAnalysis()" title="Click to open Analysis Workspace">
        <div class="stripe-code-bits" id="capsuleBits">
          <span class="bit-char">▌</span>
          <span class="bit-char">│</span>
          <span class="bit-char">█</span>
          <span class="bit-char">║</span>
          <span class="bit-char">0</span>
          <span class="bit-char">1</span>
          <span class="bit-char">0</span>
        </div>
        <span style="font-size:14px; font-weight:600; color:var(--ink);" id="capsuleLabel">
          Stripe ➔ Z-Hash Code: <code style="background:var(--surface-strong); padding:2px 8px; border-radius:4px; font-family:monospace; color:var(--primary);">e4a91b8f...</code>
        </span>
      </div>

    </section>
"""

# Replace hero section
hero_pattern = r'<section id="overview" class="wispr-hero">.*?</section>'
html = re.sub(hero_pattern, stripe_code_hero_html, html, flags=re.DOTALL, count=1)

# Add live JavaScript morphing ticker for capsule bits if not present
js_ticker = """
    // Live Stripe -> Binary -> Hex Code Morphing Ticker
    const stripeGlyphs = ['▌', '│', '█', '║', '░', '▒', '▓'];
    const binaryBits  = ['0', '1', '0', '1', '1', '0', '1'];
    const hexCodes    = ['e4', 'a9', '1b', '8f', '3c', '7d', '2e'];

    let tickCount = 0;
    setInterval(() => {
      const bitElements = document.querySelectorAll('.stripe-code-bits .bit-char');
      if (bitElements.length > 0) {
        tickCount++;
        bitElements.forEach((el, idx) => {
          if (tickCount % 3 === 0) {
            el.textContent = stripeGlyphs[idx % stripeGlyphs.length];
            el.style.color = '#292524';
          } else if (tickCount % 3 === 1) {
            el.textContent = binaryBits[idx % binaryBits.length];
            el.style.color = '#16a34a'; // Code success green accent
          } else {
            el.textContent = hexCodes[idx % hexCodes.length];
            el.style.color = '#2563eb'; // Blue hex accent
          }
        });
      }
    }, 900);
"""

if "// Live Stripe -> Binary -> Hex Code Morphing Ticker" not in html:
    html = html.replace("</script>", js_ticker + "\n  </script>")

html_path.write_text(html)
print("Successfully added Stripe-to-Code Transformation animation and live morphing ticker to index.html!")
