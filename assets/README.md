# ZebraID Media & Brand Asset Directory

This directory contains all brand identity, media visuals, logos, and favicons for the ZebraID project.

## Directory Structure

```text
assets/
├── branding/
│   ├── favicon-dark.png         # 512x512 PNG for Light browser themes (Dark logo mark)
│   ├── favicon-light.png        # 512x512 PNG for Dark browser themes (White logo mark)
│   ├── favicon.svg              # Scalable vector icon
│   ├── logo.svg                 # Primary vector brand logo
│   ├── logo.png                 # Primary raster brand logo
│   ├── logo_dark_ink.png        # High-contrast dark ink logo mark
│   ├── logo_light_ink.png       # High-contrast light ink logo mark
│   ├── zebraid_logo_perfect.png # Production navbar logo
│   └── source/                  # Original raw photography, uncompressed master files & historical drafts
│       ├── Zebra.png            # Master source photograph (6250x6250)
│       ├── zabraid_fabicon.png  # Master vector-rendered icon source (6250x6250)
│       └── zebraId_logo.png     # Master logo render
└── media/
    └── hero_biometric_scan.jpg  # Hero section biometric computer vision visual
```

## Production Asset Distribution

- **`demo/static/`**: Clean runtime asset bundle served by the FastAPI coordinator demo.
- **`public/`**: Public web assets for web deployment targets (Next.js / static hosting).
