# CXR Disease Detection (Streamlit Starter)

An end-to-end **Streamlit** starter to deploy your **Tuberculosis** and **COVID-19** CXR classifiers quickly.
Includes:
- Image upload (PNG/JPG/DICOM)
- Model selection (TB / COVID-19)
- **Grad-CAM** heatmap for explainability
- Caching for fast inference
- Option to load large weights from local `models/` or **Hugging Face Hub**

> ⚠️ **Medical Disclaimer**: This app is a research demo and **not** for clinical use.

---

## Quickstart (Local)

```bash
# 1) Create environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put your weights (≈0.7GB each) here:
#    - models/tb_model.pth
#    - models/covid_model.pth

# (Optional) Or set HF repo to auto-download if files are missing
export HF_REPO_ID=your-username/your-cxr-weights-repo

# 4) Run
streamlit run app.py
```

Open the URL shown by Streamlit (usually `http://localhost:8501`).

---

## Deploy Options

### A) Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. On streamlit.io → “New app” → select repo → app file: `app.py`.
3. If weights are too large for the repo:
   - Upload weights to a private/public **Hugging Face Hub** repo and set **secrets** in Streamlit Cloud:
     - `HF_TOKEN` (if repo is private)
     - `HF_REPO_ID` (e.g., `your-username/cxr-weights`)
4. App starts and downloads weights on first run (cached afterward during session).

### B) Hugging Face Spaces
- Create a new **Space** with **Streamlit**.
- Add `requirements.txt` and push code.
- Store weights in the Space (if within limits) or another HF repo and download at runtime.

### C) Render / Railway
- Create a Web Service; start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Use **persistent storage** or **startup download** from HF Hub/S3 for large weights.
- Pick an instance with ≥4GB RAM for smoother FP32 inference.

---

## Project Structure
```
cxr_streamlit_starter/
├── app.py                  # Streamlit UI
├── inference.py            # Model/Grad-CAM logic
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Minor UI tweaks
└── models/                 # Place your .pth weights here (or use HF Hub)
```

---

## Swap in Your Models

Edit `inference.py`:
- Replace the `build_model()` function with your actual architecture.
- Adjust preprocessing to match your training pipeline (image size, mean/std, etc.).
- Ensure your weights match the architecture and `num_classes`.

For private weights on HF Hub, set env vars:
```
HF_REPO_ID=your-username/cxr-weights
HF_TOKEN=hf_...
```
Files expected (override via env):
- `TB_WEIGHTS_PATH=models/tb_model.pth`
- `COVID_WEIGHTS_PATH=models/covid_model.pth`

---

## License
MIT
