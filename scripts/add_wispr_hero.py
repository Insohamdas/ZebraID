import base64
import re
from pathlib import Path

root = Path(__file__).parent.parent
html_path = root / "demo" / "templates" / "index.html"
html = html_path.read_text()

# Add Wispr Flow animation CSS before </style>
wispr_css = """
    /* ── WISPR FLOW HERO ANIMATION ── */
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
      font-size: 80px;
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
      max-width: 580px;
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

    /* Floating Capsule Widget (Bottom Soundwave Equalizer) */
    .wispr-capsule {
      display: inline-flex;
      align-items: center;
      gap: 16px;
      background: #ffffff;
      border: 1.5px solid var(--ink);
      padding: 10px 24px;
      border-radius: 9999px;
      box-shadow: 0 16px 40px rgba(12, 10, 9, 0.1);
      cursor: pointer;
      position: relative;
      z-index: 10;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .wispr-capsule:hover {
      transform: scale(1.03);
      box-shadow: 0 20px 48px rgba(12, 10, 9, 0.15);
    }
    .wispr-wave-bars {
      display: flex;
      align-items: center;
      gap: 3px;
      height: 22px;
    }
    .wispr-bar {
      width: 3px;
      background: var(--ink);
      border-radius: 9999px;
      animation: wisprPulse 1.2s ease-in-out infinite alternate;
    }
    .wispr-bar:nth-child(1) { height: 8px; animation-delay: 0.1s; }
    .wispr-bar:nth-child(2) { height: 16px; animation-delay: 0.3s; }
    .wispr-bar:nth-child(3) { height: 22px; animation-delay: 0.2s; }
    .wispr-bar:nth-child(4) { height: 12px; animation-delay: 0.5s; }
    .wispr-bar:nth-child(5) { height: 20px; animation-delay: 0.4s; }
    .wispr-bar:nth-child(6) { height: 10px; animation-delay: 0.6s; }
    .wispr-bar:nth-child(7) { height: 18px; animation-delay: 0.25s; }

    @keyframes wisprPulse {
      0% { transform: scaleY(0.3); }
      100% { transform: scaleY(1.0); }
    }
"""

if "/* ── WISPR FLOW HERO ANIMATION ── */" not in html:
    html = html.replace("</style>", wispr_css + "\n</style>")

# Create Wispr Flow Hero HTML Block
wispr_hero_html = """
    <!-- Wispr Flow Style Hero Band -->
    <section id="overview" class="wispr-hero">
      
      <!-- SVG Curved Text Flowing Ribbon -->
      <svg class="wispr-ribbon-svg" viewBox="0 0 1400 800" preserveAspectRatio="xMidYMid slice">
        <defs>
          <!-- Spiral Light Curve Path -->
          <path id="spiralPath" d="M -200,650 C 150,550 350,250 200,120 C 50,-10 -150,150 50,450 C 250,750 750,700 1200,450 C 1450,300 1600,600 1800,700" />
          
          <!-- Bottom Dark Ribbon Curve Path -->
          <path id="darkRibbonPath" d="M -100,720 C 400,680 900,820 1500,650" />
        </defs>

        <!-- Light Spiral Ribbon Text -->
        <text class="wispr-ribbon-text">
          <textPath href="#spiralPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="50%" dur="25s" repeatCount="indefinite" />
            ZebraID Biometric Framework · 512d Dense MegaDescriptor Embedding · Z-Hash 256b Binary Compression · Privacy-Preserving Federated Search · ZebraID Biometric Framework · 512d Dense MegaDescriptor Embedding · Z-Hash 256b Binary Compression ·
          </textPath>
        </text>

        <!-- Dark Ribbon Strip Background -->
        <path d="M -100,720 C 400,680 900,820 1500,650" fill="none" stroke="#0c0a09" stroke-width="48" stroke-linecap="round" opacity="0.95" />

        <!-- Dark Ribbon Text -->
        <text class="wispr-dark-ribbon-text">
          <textPath href="#darkRibbonPath" startOffset="0%">
            <animate attributeName="startOffset" from="0%" to="40%" dur="18s" repeatCount="indefinite" />
            Continental scale zebra recognition · Zero raw image transmission · Instant federated shard matching · 99.4% accuracy ·
          </textPath>
        </text>
      </svg>

      <!-- Wispr Header Switcher Pill -->
      <div class="wispr-nav-pill">
        <div class="wispr-tab active">Identification</div>
        <div class="wispr-tab" onclick="openAnalysis()">Federated Shards</div>
      </div>

      <div class="wispr-badge">
        ZEBRAID BIOMETRIC DICTATION & SEARCH
      </div>

      <h1 class="wispr-title">
        Don't search,<br>
        <i>just identify.</i>
      </h1>

      <p class="wispr-sub">
        The stripe-to-vector AI that turns zebra field imagery into instant, verified individual identity across continental populations.
      </p>

      <div class="hero-cta">
        <button class="btn-primary" onclick="openAnalysis()" style="height: 44px; padding: 12px 28px; font-size: 16px;">
          Identify Sighting <i data-lucide="scan" style="width:18px;"></i>
        </button>
      </div>

      <!-- Floating Equalizer Waveform Capsule (Wispr Style) -->
      <div class="flow-capsule" onclick="openAnalysis()">
        <div class="wispr-wave-bars">
          <div class="wispr-bar"></div>
          <div class="wispr-bar"></div>
          <div class="wispr-bar"></div>
          <div class="wispr-bar"></div>
          <div class="wispr-bar"></div>
          <div class="wispr-bar"></div>
          <div class="wispr-bar"></div>
        </div>
        <span style="font-size:14px; font-weight:600; color:var(--ink);">Biometric Stripe Match Engine</span>
      </div>

    </section>
"""

# Replace existing hero-band with Wispr Flow Hero
hero_pattern = r'<section id="overview" class="hero-band">.*?</section>'
html = re.sub(hero_pattern, wispr_hero_html, html, flags=re.DOTALL, count=1)

html_path.write_text(html)
print("Successfully added Wispr Flow style animated text ribbon and hero section to index.html!")
