#!/usr/bin/env python3
"""Effort estimator Flask app packaged for easy download and reuse."""

from __future__ import annotations

import io
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
TEMPLATE_DIR = PACKAGE_DIR / "templates"
MODEL_JSON = PACKAGE_DIR / "effort_nn_model.json"
MODEL_WEIGHTS = PACKAGE_DIR / "effort_nn_weights.weights.h5"
SCALER_FILE = PACKAGE_DIR / "scaler.pkl"

REQUIRED_PACKAGES = [
    "flask",
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "pdfplumber",
    "python-docx",
    "reportlab",
    "requests",
    "openai",
]


def _ensure_package(package: str) -> None:
    import_name = package.replace("-", "_")
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


for package in REQUIRED_PACKAGES:
    _ensure_package(package)

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, make_response, render_template, request
from docx import Document
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.secret_key = os.urandom(24)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = int(os.environ.get("OPENAI_API_TIMEOUT", "45"))
COPILOT_ENABLED = os.environ.get("COPILOT_ENABLED", "true").lower() == "true"

PHASE_DIST = {
    "Requirement analysis": 0.08,
    "Clarifications": 0.05,
    "Design": 0.12,
    "Development": 0.35,
    "Unit Test": 0.10,
    "SIT": 0.08,
    "UAT": 0.07,
    "Preparation change": 0.05,
    "Deployment": 0.06,
    "Warranty": 0.04,
}

MODEL = None
SCALER = None


def _best_model_source() -> Path | None:
    if MODEL_JSON.exists() and MODEL_WEIGHTS.exists() and SCALER_FILE.exists():
        return PACKAGE_DIR
    if (ROOT_DIR / MODEL_JSON.name).exists() and (ROOT_DIR / MODEL_WEIGHTS.name).exists() and (ROOT_DIR / SCALER_FILE.name).exists():
        return ROOT_DIR
    return None


def _train_model() -> tuple[Any, StandardScaler]:
    np.random.seed(42)
    n_samples = 3000

    req_count = np.random.poisson(lam=50, size=n_samples) + 10
    complexity = np.random.uniform(1, 5, n_samples)
    team_exp = np.random.uniform(0, 15, n_samples)
    tech_unc = np.random.uniform(0, 1, n_samples)
    integrations = np.random.poisson(lam=5, size=n_samples)
    doc_quality = np.random.uniform(1, 5, n_samples)
    deadline = np.random.uniform(0.5, 2.0, n_samples)
    domain = np.random.uniform(1, 5, n_samples)

    x = np.column_stack([
        req_count,
        complexity,
        team_exp,
        tech_unc,
        integrations,
        doc_quality,
        deadline,
        domain,
    ])

    base = 200 + 8 * req_count + 40 * complexity - 25 * team_exp + 120 * tech_unc + 12 * integrations
    base += 5 * doc_quality - 30 * deadline + 8 * domain
    base += 15 * complexity * tech_unc
    base += -10 * team_exp * doc_quality
    noise = np.random.normal(0, 50, n_samples)
    y = np.maximum(base + noise, 20)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = keras.Sequential([
        layers.Input(shape=(8,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(x_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    mse = float(np.mean((y_test - model.predict(x_test_scaled, verbose=0).ravel()) ** 2))
    print(f"Model trained. Test MSE: {mse:.2f}")

    with open(MODEL_JSON, "w", encoding="utf-8") as f:
        f.write(model.to_json())
    model.save_weights(MODEL_WEIGHTS)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    return model, scaler


def _load_or_train_model() -> tuple[Any, StandardScaler]:
    source = _best_model_source()
    if source is None:
        print("Model artifacts not found in package or repo root. Training a new model.")
        return _train_model()

    source_model_json = source / MODEL_JSON.name
    source_weights = source / MODEL_WEIGHTS.name
    source_scaler = source / SCALER_FILE.name

    with open(source_model_json, "r", encoding="utf-8") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights(source_weights)

    with open(source_scaler, "rb") as f:
        scaler = pickle.load(f)

    if source != PACKAGE_DIR:
        PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
        if not MODEL_JSON.exists():
            MODEL_JSON.write_text(model_json, encoding="utf-8")
        if not MODEL_WEIGHTS.exists():
            MODEL_WEIGHTS.write_bytes(source_weights.read_bytes())
        if not SCALER_FILE.exists():
            SCALER_FILE.write_bytes(source_scaler.read_bytes())

    return model, scaler


MODEL, SCALER = _load_or_train_model()


def _call_openai(messages: List[Dict[str, str]], max_tokens: int = 900) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured")

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=OPENAI_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    import pdfplumber

    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def _extract_text_from_docx(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    return "\n".join([paragraph.text for paragraph in document.paragraphs])


def _extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def _heuristic_features(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    req_keywords = ["requirement", "shall", "must", "should", "feature", "functionality"]
    req_count = sum(text_lower.count(keyword) for keyword in req_keywords)
    req_count = max(20, min(200, req_count // 3))

    complexity = 3
    for keyword, score in {
        "simple": 1,
        "basic": 1,
        "straightforward": 2,
        "complex": 4,
        "advanced": 4,
        "sophisticated": 5,
        "integration": 3,
        "machine learning": 4,
        "ai": 3,
        "algorithm": 3,
    }.items():
        if keyword in text_lower:
            complexity = max(complexity, score)

    integration_keywords = ["api", "integration", "third-party", "3rd party", "external", "database", "microservice"]
    integrations = min(15, sum(text_lower.count(keyword) for keyword in integration_keywords))

    doc_quality = 3 + min(2, sum(1 for keyword in ["specification", "documented", "design", "architecture", "diagram", "flowchart"] if keyword in text_lower))
    tech_unc = 0.3 + min(0.5, 0.1 * sum(text_lower.count(keyword) for keyword in ["new technology", "poc", "proof of concept", "research", "evaluation", "unproven"]))
    deadline = 1.5 if any(keyword in text_lower for keyword in ["urgent", "asap", "immediately", "tight", "phased", "phase"]) else 1.0

    return {
        "req_count": int(req_count),
        "complexity": float(min(5, complexity)),
        "team_exp": 5.0,
        "tech_unc": float(min(1, tech_unc)),
        "integrations": int(integrations),
        "doc_quality": float(min(5, doc_quality)),
        "deadline": float(min(2, deadline)),
        "domain": 3.0,
    }


def _llm_extract_features(text: str) -> Dict[str, Any] | None:
    if not OPENAI_API_KEY:
        return None

    prompt = f"""
Analyze this business requirements document and return ONLY valid JSON with keys:
req_count, complexity, team_exp, tech_unc, integrations, doc_quality, deadline, domain.
Use numbers only. Clamp complexity/domain/doc_quality to 1-5, tech_unc to 0-1, deadline to 0.5-2.

Document:
{text[:50000]}
""".strip()

    try:
        content = _call_openai(
            [
                {"role": "system", "content": "You extract structured estimation inputs from business requirements documents."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
        )
        content = re.sub(r"```json\s*|\s*```", "", content.strip())
        parsed = json.loads(content)
        return {
            "req_count": int(max(1, min(500, parsed.get("req_count", 60)))),
            "complexity": float(max(1, min(5, parsed.get("complexity", 3)))),
            "team_exp": float(max(0, min(15, parsed.get("team_exp", 5)))),
            "tech_unc": float(max(0, min(1, parsed.get("tech_unc", 0.5)))),
            "integrations": int(max(0, min(50, parsed.get("integrations", 5)))),
            "doc_quality": float(max(1, min(5, parsed.get("doc_quality", 3)))),
            "deadline": float(max(0.5, min(2, parsed.get("deadline", 1)))),
            "domain": float(max(1, min(5, parsed.get("domain", 3)))),
        }
    except Exception as exc:
        print(f"LLM extraction failed, falling back to heuristics: {exc}")
        return None


def _normalize_features(values: List[float]) -> List[float]:
    x = np.array(values, dtype=float).reshape(1, -1)
    return SCALER.transform(x)[0].tolist()


def predict_total(features: List[float]) -> float:
    x = np.array(features, dtype=float).reshape(1, -1)
    x_scaled = SCALER.transform(x)
    total = MODEL.predict(x_scaled, verbose=0)[0][0]
    return max(float(total), 0.0)


def _math_calculation_html(features: List[float], total: float) -> str:
    names = [
        ("Requirements", features[0], 8),
        ("Complexity", features[1], 40),
        ("Team experience", features[2], -25),
        ("Tech uncertainty", features[3], 120),
        ("Integrations", features[4], 12),
        ("Documentation quality", features[5], 5),
        ("Deadline pressure", features[6], -30),
        ("Domain knowledge", features[7], 8),
    ]
    contribution_rows = "".join(
        f"<tr><td>{label}</td><td>{value:.2f}</td><td>{coeff:+.0f}</td><td>{value * coeff:+.1f}</td></tr>"
        for label, value, coeff in names
    )
    normalized = _normalize_features(features)
    normalized_rows = "".join(
        f"<tr><td>{label}</td><td>{value:.2f}</td></tr>"
        for label, value in zip([item[0] for item in names], normalized)
    )
    return f"""
    <div class="math-card">
        <h3>Mathematical breakdown</h3>
        <p>The model first standardizes the input vector, then passes it through a 64-neuron layer, dropout, a 32-neuron layer, and a linear output neuron.</p>
        <table class="math-table">
            <thead><tr><th>Feature</th><th>Scaled value</th></tr></thead>
            <tbody>{normalized_rows}</tbody>
        </table>
        <table class="math-table">
            <thead><tr><th>Feature</th><th>Raw value</th><th>Weight hint</th><th>Contribution hint</th></tr></thead>
            <tbody>{contribution_rows}</tbody>
        </table>
        <p><strong>Predicted effort:</strong> {total:.1f} person-hours</p>
        <p class="muted">Formula view: y = w3(ReLU(w2(ReLU(w1x + b1)) + b2)) + b3</p>
    </div>
    """


def _explanation_text(features: List[float], total: float, use_copilot: bool = True) -> str:
    order = sorted(
        [
            ("requirements", features[0], 8),
            ("complexity", features[1], 40),
            ("team experience", features[2], -25),
            ("tech uncertainty", features[3], 120),
            ("integrations", features[4], 12),
            ("documentation quality", features[5], 5),
            ("deadline pressure", features[6], -30),
            ("domain knowledge", features[7], 8),
        ],
        key=lambda item: abs(item[1] * item[2]),
        reverse=True,
    )[:3]

    if total < 500:
        level = "low"
    elif total < 1000:
        level = "moderate"
    else:
        level = "high"

    lines = [
        f"The model predicts a {level} effort of {total:.1f} person-hours.",
        "Top drivers:",
    ]
    for name, value, coeff in order:
        direction = "raises" if coeff > 0 else "reduces"
        lines.append(f"- {name.title()} = {value:.2f} ({direction} effort)")
    lines.append("Development receives 35% of the total as the largest phase.")
    if use_copilot and OPENAI_API_KEY:
        lines.append("Copilot support is active for improved extraction and chat responses.")
    return "\n".join(lines)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _chat_response(question: str, params: Dict[str, Any], total: Any, chat_history: List[Dict[str, str]]) -> str:
    if OPENAI_API_KEY:
        prompt = f"""
Project parameters:
- req_count: {params.get('req_count', '?')}
- complexity: {params.get('complexity', '?')}
- team_exp: {params.get('team_exp', '?')}
- tech_unc: {params.get('tech_unc', '?')}
- integrations: {params.get('integrations', '?')}
- doc_quality: {params.get('doc_quality', '?')}
- deadline: {params.get('deadline', '?')}
- domain: {params.get('domain', '?')}
Total effort: {total} person-hours.

User question: {question}
""".strip()
        try:
            return _call_openai(
                [
                    {"role": "system", "content": "You help a project manager understand effort estimates, risks, and next steps."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=250,
            )
        except Exception as exc:
            print(f"Chat fallback triggered: {exc}")

    answer = [
        "I can help interpret the estimate even without an LLM key.",
        f"Current total: {total} person-hours.",
    ]
    tech_unc = _as_float(params.get("tech_unc", 0), 0.0)
    integrations = _as_float(params.get("integrations", 0), 0.0)
    deadline = _as_float(params.get("deadline", 1), 1.0)

    if tech_unc >= 0.6:
        answer.append("High technical uncertainty is the biggest risk driver.")
    if integrations >= 5:
        answer.append("Integration count suggests extra coordination and test effort.")
    if deadline < 1:
        answer.append("Tight deadline usually increases delivery risk.")
    answer.append(f"Question noted: {question}")
    return " ".join(answer)


def _pdf_story(params: Dict[str, Any], total: float, phases: List[Dict[str, Any]], explanation: str, math_html: str, chat_messages: List[Dict[str, str]]) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.spaceAfter = 8

    story: List[Any] = [
        Paragraph("Effort Estimation Report", title_style),
        Spacer(1, 0.2 * inch),
        Paragraph("Input Parameters", heading_style),
    ]

    param_rows = [["Parameter", "Value"]] + [[key.replace("_", " ").title(), str(value)] for key, value in params.items()]
    param_table = Table(param_rows, colWidths=[2.7 * inch, 2.4 * inch])
    param_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12324a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e2ec")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f8fbfd")]),
    ]))
    story.extend([param_table, Spacer(1, 0.2 * inch)])

    story.extend([
        Paragraph(f"Total Estimated Effort: {total:.1f} person-hours", heading_style),
        Spacer(1, 0.08 * inch),
        Paragraph("Phase Breakdown", heading_style),
    ])

    phase_rows = [["Phase", "Hours"]] + [[phase["name"], f"{phase['hours']:.1f}"] for phase in phases]
    phase_table = Table(phase_rows, colWidths=[3.4 * inch, 1.7 * inch])
    phase_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12324a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d9e2ec")),
    ]))
    story.extend([phase_table, Spacer(1, 0.2 * inch), Paragraph("Math Breakdown", heading_style)])

    story.append(Paragraph(re.sub(r"<[^>]+>", " ", math_html).replace("  ", " ").strip(), body_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Chat History", heading_style))
    for message in chat_messages:
        prefix = "User" if message.get("type") == "user" else "Assistant"
        story.append(Paragraph(f"<b>{prefix}:</b> {message.get('text', '')}", body_style))

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Explanation", heading_style))
    story.append(Paragraph(explanation.replace("\n", "<br/>") , body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_documents() -> Any:
    if "documents" not in request.files:
        return jsonify({"error": "No documents provided"}), 400

    files = request.files.getlist("documents")
    texts: List[str] = []
    file_names: List[str] = []

    for file in files:
        if not file.filename:
            continue
        file_names.append(file.filename)
        file_bytes = file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            texts.append(_extract_text_from_pdf(file_bytes))
        elif filename.endswith(".docx"):
            texts.append(_extract_text_from_docx(file_bytes))
        elif filename.endswith(".txt"):
            texts.append(_extract_text_from_txt(file_bytes))

    combined_text = "\n".join(text for text in texts if text.strip())
    if not combined_text.strip():
        return jsonify({"error": "No readable text extracted"}), 400

    features = _llm_extract_features(combined_text) or _heuristic_features(combined_text)
    extraction_details = [
        f"Processed: {', '.join(file_names)}",
        f"Requirements Count: {features['req_count']}",
        f"Complexity: {features['complexity']:.1f}/5",
        f"Team Experience: {features['team_exp']:.1f} years",
        f"Tech Uncertainty: {features['tech_unc']:.2f}",
        f"Integrations: {features['integrations']}",
        f"Documentation Quality: {features['doc_quality']:.1f}/5",
        f"Deadline Pressure: {features['deadline']:.1f}",
        f"Domain Knowledge: {features['domain']:.1f}/5",
    ]
    features["extraction_details"] = "<div class='feature-list'>" + "".join(f"<div>{item}</div>" for item in extraction_details) + "</div>"
    return jsonify(features)


@app.route("/predict", methods=["POST"])
def predict() -> Any:
    try:
        data = request.get_json(force=True)
        features = [
            float(data["req_count"]),
            float(data["complexity"]),
            float(data["team_exp"]),
            float(data["tech_unc"]),
            float(data["integrations"]),
            float(data["doc_quality"]),
            float(data["deadline"]),
            float(data["domain"]),
        ]
        total = predict_total(features)
        phases = [{"name": name, "hours": round(total * pct, 1)} for name, pct in PHASE_DIST.items()]
        return jsonify({
            "total": round(total, 1),
            "phases": phases,
            "explanation": _explanation_text(features, total, use_copilot=COPILOT_ENABLED),
            "math_calculation": _math_calculation_html(features, total),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/chat", methods=["POST"])
def chat() -> Any:
    data = request.get_json(force=True)
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Question is required"}), 400
    params = data.get("params", {}) or {}
    total = data.get("total", "unknown")
    chat_history = data.get("chatMessages", []) or []
    answer = _chat_response(question, params, total, chat_history)
    return jsonify({"answer": answer})


@app.route("/export_pdf", methods=["POST"])
def export_pdf() -> Any:
    data = request.get_json(force=True)
    params = data.get("params", {}) or {}
    total = float(data.get("total", 0) or 0)
    phases = data.get("phases", []) or []
    explanation = data.get("explanation", "") or ""
    math_calculation = data.get("math_calculation", "") or ""
    chat_messages = data.get("chatMessages", []) or []

    pdf_bytes = _pdf_story(params, total, phases, explanation, math_calculation, chat_messages)
    response = make_response(pdf_bytes)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=effort-estimation-report.pdf"
    return response


@app.route("/health")
def health() -> Any:
    return jsonify({"status": "ok", "openai_enabled": bool(OPENAI_API_KEY), "copilot_enabled": COPILOT_ENABLED})


def main() -> None:
    print("\n" + "=" * 60)
    print("Effort Estimator is starting")
    print(f"Working directory: {PACKAGE_DIR}")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
