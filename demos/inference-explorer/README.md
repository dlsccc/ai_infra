# Inference Explorer

A plain HTML/CSS/JS demo for visualizing model inference.

Current coverage:
- ResNet-50 inference flow
- BERT encoder inference flow (with attention heatmap)

## Files
- `index.html` entry page
- `styles.css` visual style
- `app.js` graph rendering and step playback

## Run
1. `cd E:\ai_infra\demos\inference-explorer`
2. Open `index.html` directly, or run:
3. `python -m http.server 8080`
4. Open `http://localhost:8080`

## Interactions
- Switch model: `ResNet / BERT`
- Control steps: `Prev / Next / Auto Play`
- Right panel: step description, IO shape, key ops
- BERT attention steps: heatmap rendering

## Future expansion
- Add more models (ViT, U-Net, Llama, MoE)
- Drive animations from real inference logs (JSON)
- Show per-layer latency, params, and FLOPs
