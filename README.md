# Multi-Modal Fake News Detector (ML + Agentic AI)

A Flask project that analyzes text, URL content, images, and videos, then combines all signals through a meta-agent to predict whether news is `REAL` or `FAKE`.

## What This Project Includes

- Text analysis using trained NLP model (`model.pkl` + `vectorizer.pkl`)
- URL-based extraction and text analysis
- Image analysis agent
- Video analysis agent
- Verifier agent for consistency checks
- Meta-agent for final decision + confidence + explanation
- OpenAI-powered chatbot endpoint (`/chat`) for result explanation
- SQLite history (last results)
- PDF report export

## Architecture (Simple View)

1. Input comes from text, URL, image, video, or any mix.
2. Each input is handled by its own agent.
3. Verifier agent checks consistency/conflict.
4. Meta-agent combines available agent outputs into final result.
5. UI shows final output + agent-wise details.
6. Chatbot explains the result using latest analysis context.

## Project Structure

```text
news_detector_ai/
  app.py
  agent.py
  text_detector.py
  image_detector.py
  video_detector.py
  url_detector.py
  verifier.py
  history.py
  history.db
  train_model.py
  train_image_model.py
  train_video_model.py
  model.pkl
  vectorizer.pkl
  image_model.pkl
  video_model.pkl
  requirements.txt
  templates/
    index.html
  static/
    uploads/
```

## Dataset Setup

### Text dataset

Supported setup:

- `News_Dataset/Fake.csv`
- `News_Dataset/True.csv`

or

- `news.csv` (unstructured/raw supported by fallback logic in training script)

### Image dataset

```text
dataset/
  Train/
    Fake/
    Real/
  Test/
    Fake/
    Real/
```

### Video dataset

```text
video dataset/
  attack/      # fake
  real_video/  # real
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train Models

Run only if model files are missing or you want retraining:

```powershell
python train_model.py
python train_image_model.py
python train_video_model.py
```

Outputs:

- `model.pkl`
- `vectorizer.pkl`
- `image_model.pkl`
- `video_model.pkl`

## Environment Variables (Important)

Set these before running app:

```powershell
$env:FLASK_SECRET_KEY="your_strong_secret_key"
$env:OPENAI_API_KEY="your_openai_api_key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Notes:

- If `OPENAI_API_KEY` is missing, chatbot falls back to local non-LLM response.
- `FLASK_SECRET_KEY` should not be hardcoded in production.

## Run

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Publish Publicly (Render)

This project is now deployment-ready for Render.

### 1. Push to GitHub

```powershell
git init
git add .
git commit -m "Deploy-ready fake news detector"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New +** -> **Blueprint**
3. Select your GitHub repo
4. Render will detect `render.yaml`
5. Set `OPENAI_API_KEY` in Render environment variables
6. Deploy

Your app will be live on a public `onrender.com` URL.

### 3. Important

- Keep `OPENAI_API_KEY` private (set only in Render dashboard, not code).
- If your model files are large, ensure they are included in repo or available at runtime.

## Main Features in UI

- Mode buttons: Text, URL, Image, Video, Mix
- Premium interactive result card
- Agent-wise status and confidence
- Final confidence bar
- Explain-more toggles
- Feedback buttons
- Dark/light mode
- Floating OpenAI chatbot

## API Routes

- `GET /` -> main page
- `POST /analyze` -> run detection pipeline
- `POST /chat` -> chatbot response (OpenAI + fallback)
- `GET /export-report` -> PDF export for latest analysis

## Security and Stability Updates Already Applied

- Secure upload names using `secure_filename` + UUID prefix
- Upload folder auto-created (`static/uploads`)
- Graceful error handling for analyze/chat flows
- Consistent final confidence rendering
- Scikit-learn pinned in `requirements.txt` (`1.8.0`)

## Submission Checklist

- Install requirements in fresh venv
- Confirm model files exist
- Set environment variables
- Test all 5 flows:
  - text only
  - URL only
  - image only
  - video only
  - mix
- Test chatbot after one analysis
- Test PDF export

## Current Scope and Limitations

- Quality depends on dataset size/quality, especially image/video.
- Meta-agent is rule-guided orchestration (agentic workflow), not fully autonomous planning.
- For production-grade reliability, use larger curated datasets and stronger model monitoring.

## Tech Stack

- Python, Flask
- scikit-learn, pandas, numpy
- Pillow, OpenCV
- requests, BeautifulSoup4
- HTML/CSS/JavaScript

---

If you want, next step I can add a short `DEPLOY.md` with Render/Railway deployment commands so you can host this quickly for demo.
