# Effort Estimator Package

This folder contains the packaged version of the effort estimator app.

## Features
- BRS upload for PDF, DOCX, and TXT documents
- GPT-backed feature extraction when `OPENAI_API_KEY` is set
- Chat assistant for estimation follow-up
- PDF report download
- Clean responsive UI
- Neural network estimation with visible math breakdown

## Run

From inside this folder or from the repository root:

```bash
python -m effort_estimator_package.upgraded_estimator
```

You can also run the root launcher:

```bash
python upgraded_estimator.py
```

Both launch the same packaged Flask app.

## Optional OpenAI support

Set:

```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini
```

When the key is not set, the app uses local heuristics for extraction and chat.
