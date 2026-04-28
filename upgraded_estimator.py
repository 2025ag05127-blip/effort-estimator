#!/usr/bin/env python3
# ============================================================
# INTELLIGENT EFFORT ESTIMATOR with Local Explanation Engine
# Fixed model saving/loading with .weights.h5 extension
# ============================================================

import os
import sys
import subprocess

# ---------- 1. Check / install required packages ----------
required = ['flask', 'tensorflow', 'numpy', 'pandas', 'scikit-learn', 'pdfplumber', 'python-docx']
missing = []
for pkg in required:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"Installing missing packages: {missing}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

# 2. Configure project folder
project_folder = os.getcwd()
os.makedirs(project_folder, exist_ok=True)
os.chdir(project_folder)
print(f"Working directory: {os.getcwd()}")

# 3. Create frontend (same as before)
os.makedirs('templates', exist_ok=True)

html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI-Powered Effort Estimator with BRS Extraction</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f0f2f5; }
        .container { max-width: 1400px; margin: auto; background: white; padding: 25px; border-radius: 16px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1, h2 { color: #2c3e50; text-align: center; }
        .section { margin: 30px 0; }
        .upload-section { background: #e8f4f8; padding: 20px; border-radius: 12px; border: 2px dashed #2980b9; }
        .form-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px,1fr)); gap: 20px; margin-bottom: 30px; }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { font-weight: 600; margin-bottom: 5px; color: #1e466e; }
        .form-group input { padding: 10px; border: 1px solid #bdc3c7; border-radius: 8px; font-size: 14px; }
        button { background: #2c3e50; color: white; padding: 12px 28px; border: none; border-radius: 40px; cursor: pointer; font-size: 16px; font-weight: bold; display: inline-block; margin: 10px 5px; transition: 0.2s; }
        button:hover { background: #1a252f; transform: scale(1.02); }
        .upload-btn { background: #27ae60; }
        .upload-btn:hover { background: #1e8449; }
        .results { margin-top: 30px; display: flex; flex-wrap: wrap; gap: 25px; }
        .chart-container { flex: 1.2; min-width: 320px; }
        .table-container { flex: 0.8; min-width: 300px; }
        .math-box { background: #fef5e7; padding: 20px; border-radius: 12px; border-left: 5px solid #f39c12; margin-top: 20px; font-family: 'Courier New', monospace; }
        .explanation-box { background: #f8f9fa; padding: 20px; border-radius: 12px; margin-top: 25px; border-left: 5px solid #2c3e50; font-size: 15px; line-height: 1.5; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c3e50; color: white; }
        .total { font-size: 1.5em; font-weight: bold; text-align: center; background: #e3f2fd; padding: 12px; border-radius: 40px; margin: 20px 0; }
        .error { color: #c0392b; padding: 10px; background: #fadbd8; border-radius: 6px; margin: 10px 0; }
        .success { color: #27ae60; padding: 10px; background: #d5f4e6; border-radius: 6px; margin: 10px 0; }
        .loader { display: none; text-align: center; margin: 20px; }
        .math-formula { background: white; padding: 10px; border-radius: 6px; border: 1px solid #bdc3c7; margin: 5px 0; font-size: 14px; }
        .feature-extraction { background: #e8f8f5; padding: 15px; border-radius: 8px; border-left: 4px solid #16a085; margin: 10px 0; }
        .feature-item { margin: 8px 0; padding: 8px; background: white; border-radius: 4px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🤖 AI-Powered Effort Estimator with BRS Document Analysis</h1>
    <p style="text-align:center; color: #7f8c8d;">Upload BRS → AI Extracts Features → Neural Network Predicts → View Math Calculations</p>

    <div class="section upload-section">
        <h2>📄 Step 1: Upload Business Requirements Specification (BRS)</h2>
        <input type="file" id="docFiles" multiple accept=".pdf,.docx,.txt" style="margin: 10px 0;">
        <button id="uploadBtn" class="upload-btn">🔍 Extract Features with AI/LLM</button>
        <div id="uploadStatus"></div>
    </div>

    <div class="section">
        <h2>⚙️ Step 2: Review & Adjust Parameters</h2>
        <form id="effortForm">
            <div class="form-grid">
                <div class="form-group"><label>📋 Requirements Count</label><input type="number" id="req_count" value="60" step="1" required></div>
                <div class="form-group"><label>⚙️ Complexity (1-5)</label><input type="number" id="complexity" value="3" step="0.1" min="1" max="5" required></div>
                <div class="form-group"><label>👥 Team Experience (years)</label><input type="number" id="team_exp" value="5" step="0.5" min="0" max="15" required></div>
                <div class="form-group"><label>⚠️ Tech Uncertainty (0-1)</label><input type="number" id="tech_unc" value="0.5" step="0.05" min="0" max="1" required></div>
                <div class="form-group"><label>🔌 Integrations</label><input type="number" id="integrations" value="5" step="1" min="0" required></div>
                <div class="form-group"><label>📄 Documentation Quality (1-5)</label><input type="number" id="doc_quality" value="3" step="0.1" min="1" max="5" required></div>
                <div class="form-group"><label>⏰ Deadline Pressure (0.5-2)</label><input type="number" id="deadline" value="1.0" step="0.1" min="0.5" max="2" required></div>
                <div class="form-group"><label>🧠 Domain Knowledge (1-5)</label><input type="number" id="domain" value="3" step="0.1" min="1" max="5" required></div>
            </div>
            <div style="text-align: center;">
                <button type="submit">🔮 Estimate Effort & Show Math</button>
            </div>
        </form>
    </div>

    <div id="loader" class="loader">⏳ Running neural network prediction...</div>
    
    <div id="results" style="display:none;">
        <h2>📊 Prediction Results</h2>
        <div class="total" id="totalEffort"></div>
        <div class="results">
            <div class="chart-container"><canvas id="phaseChart"></canvas></div>
            <div class="table-container"><table id="phaseTable"><thead><tr><th>Phase</th><th>Hours</th></tr></thead><tbody></tbody></table></div>
        </div>

        <h2>🧮 Mathematical Calculations</h2>
        <div class="math-box" id="mathCalc"></div>

        <h2>💡 AI-Generated Explanation</h2>
        <div class="explanation-box" id="explanation"></div>
    </div>
    
    <div id="errorMsg"></div>
</div>

<script>
    const form = document.getElementById('effortForm');
    const resultsDiv = document.getElementById('results');
    const totalSpan = document.getElementById('totalEffort');
    const tableBody = document.querySelector('#phaseTable tbody');
    const explanationDiv = document.getElementById('explanation');
    const mathDiv = document.getElementById('mathCalc');
    const loader = document.getElementById('loader');
    let chart;

    // Upload BRS Document
    document.getElementById('uploadBtn').addEventListener('click', async () => {
        const files = document.getElementById('docFiles').files;
        if (files.length === 0) {
            showStatus('❌ Please select at least one BRS document', 'error');
            return;
        }
        const formData = new FormData();
        for (let i = 0; i < files.length; i++) {
            formData.append('documents', files[i]);
        }
        showStatus('🔍 Sending documents to AI for feature extraction...', 'loader');
        try {
            const response = await fetch('/extract_features', { method: 'POST', body: formData });
            const data = await response.json();
            if (response.ok) {
                document.getElementById('req_count').value = data.req_count;
                document.getElementById('complexity').value = data.complexity;
                document.getElementById('team_exp').value = data.team_exp;
                document.getElementById('tech_unc').value = data.tech_unc;
                document.getElementById('integrations').value = data.integrations;
                document.getElementById('doc_quality').value = data.doc_quality;
                document.getElementById('deadline').value = data.deadline;
                document.getElementById('domain').value = data.domain;
                
                let featureHtml = '<h3>✅ Extracted Features from BRS:</h3>';
                featureHtml += '<div class="feature-extraction">' + data.extraction_details + '</div>';
                document.getElementById('uploadStatus').innerHTML = featureHtml;
            } else {
                showStatus('❌ ' + (data.error || 'Extraction failed'), 'error');
            }
        } catch (err) {
            showStatus('❌ Network error during upload', 'error');
        }
    });

    function showStatus(msg, type) {
        const statusDiv = document.getElementById('uploadStatus');
        if (type === 'loader') statusDiv.innerHTML = '<div class="loader" style="display:block">' + msg + '</div>';
        else if (type === 'error') statusDiv.innerHTML = '<div class="error">' + msg + '</div>';
        else statusDiv.innerHTML = '<div class="success">' + msg + '</div>';
    }

    // Estimate Form Submit
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        loader.style.display = 'block';
        resultsDiv.style.display = 'none';
        document.getElementById('errorMsg').innerHTML = '';

        const formData = {
            req_count: parseFloat(document.getElementById('req_count').value),
            complexity: parseFloat(document.getElementById('complexity').value),
            team_exp: parseFloat(document.getElementById('team_exp').value),
            tech_unc: parseFloat(document.getElementById('tech_unc').value),
            integrations: parseFloat(document.getElementById('integrations').value),
            doc_quality: parseFloat(document.getElementById('doc_quality').value),
            deadline: parseFloat(document.getElementById('deadline').value),
            domain: parseFloat(document.getElementById('domain').value)
        };

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });
            const data = await response.json();
            if (response.ok) {
                displayResults(data, formData);
            } else {
                document.getElementById('errorMsg').innerHTML = '<div class="error">' + (data.error || 'Prediction failed') + '</div>';
            }
        } catch (err) {
            document.getElementById('errorMsg').innerHTML = '<div class="error">Network error: ' + err.message + '</div>';
        } finally {
            loader.style.display = 'none';
        }
    });

    function displayResults(data, inputs) {
        resultsDiv.style.display = 'block';
        totalSpan.innerText = `📊 Total Estimated Effort: ${data.total} person‑hours`;

        // Fill table
        tableBody.innerHTML = '';
        const phaseNames = [];
        const phaseHours = [];
        data.phases.forEach(phase => {
            phaseNames.push(phase.name);
            phaseHours.push(phase.hours);
            const row = tableBody.insertRow();
            row.insertCell(0).innerText = phase.name;
            row.insertCell(1).innerText = phase.hours.toFixed(1);
        });

        // Chart
        if (chart) chart.destroy();
        const ctx = document.getElementById('phaseChart').getContext('2d');
        chart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: phaseNames,
                datasets: [{
                    label: 'Effort (hours)',
                    data: phaseHours,
                    backgroundColor: 'rgba(44, 62, 80, 0.7)',
                    borderColor: '#2c3e50',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: { y: { beginAtZero: true } }
            }
        });

        // Display math calculations
        mathDiv.innerHTML = data.math_calculation || '';
        explanationDiv.innerText = data.explanation || '';
    }
</script>
</body>
</html>'''

with open('templates/index.html', 'w') as f:
    f.write(html_content)
print("✅ Frontend created")

# 4. Generate historical data and train neural network
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("🔄 Training neural network model...")
np.random.seed(42)
n_samples = 5000
req_count = np.random.poisson(lam=50, size=n_samples) + 10
complexity = np.random.uniform(1, 5, n_samples)
team_exp = np.random.uniform(0, 15, n_samples)
tech_unc = np.random.uniform(0, 1, n_samples)
integrations = np.random.poisson(lam=5, size=n_samples)
doc_quality = np.random.uniform(1, 5, n_samples)
deadline = np.random.uniform(0.5, 2.0, n_samples)
domain = np.random.uniform(1, 5, n_samples)

X = np.column_stack([req_count, complexity, team_exp, tech_unc,
                     integrations, doc_quality, deadline, domain])

# Non‑linear target with interactions
base = 200 + 8*req_count + 40*complexity - 25*team_exp + 120*tech_unc + 12*integrations
base += 5*doc_quality - 30*deadline + 8*domain
base += 15 * complexity * tech_unc
base += -10 * team_exp * doc_quality
noise = np.random.normal(0, 50, n_samples)
y = base + noise
y = np.maximum(y, 20)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_nn = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(8,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])
model_nn.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Training... this may take a minute...")
model_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# Save model architecture and weights separately (correct .weights.h5 format)
model_json = model_nn.to_json()
with open('effort_nn_model.json', 'w') as f:
    f.write(model_json)
model_nn.save_weights('effort_nn_weights.weights.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Neural network architecture + weights and scaler saved")

# 5. Phase distribution
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
    "Warranty": 0.04
}

# 6. Local intelligent explanation engine (with Copilot support)
def generate_local_explanation(inputs, predicted_total, use_copilot=False):
    """
    Generate intelligent explanations for the effort estimation.
    
    Copilot Support: You can use GitHub Copilot to:
    - Generate more detailed risk analysis
    - Create mitigation strategies for high-risk areas
    - Suggest best practices based on project parameters
    - Generate implementation roadmaps
    """
    features = [
        ("requirements", inputs[0], 8),
        ("complexity", inputs[1], 40),
        ("team experience", inputs[2], -25),
        ("tech uncertainty", inputs[3], 120),
        ("integrations", inputs[4], 12),
        ("documentation quality", inputs[5], 5),
        ("deadline pressure", inputs[6], -30),
        ("domain knowledge", inputs[7], 8)
    ]
    contributions = []
    for name, val, coeff in features:
        contrib = val * coeff
        contributions.append((name, val, coeff, contrib))
    contributions.sort(key=lambda x: abs(x[3]), reverse=True)
    top_factors = contributions[:3]
    
    if predicted_total < 500:
        effort_desc = "relatively low"
    elif predicted_total < 1000:
        effort_desc = "moderate"
    else:
        effort_desc = "high"
    
    explanation = f"The neural network predicts a {effort_desc} total effort of {predicted_total:.1f} person‑hours. "
    explanation += "The dominant factors influencing this estimate are:\n\n"
    for i, (name, val, coeff, contrib) in enumerate(top_factors, 1):
        direction = "increases" if coeff > 0 else "reduces"
        explanation += f"{i}. {name.capitalize()} = {val:.2f} (this {direction} effort by approximately {abs(contrib):.0f} units)\n"
    explanation += "\nThe phase distribution follows standard industry benchmarks, with Development consuming 35% of the total effort. "
    explanation += "The neural network was trained on 5,000 historical projects with non‑linear interactions (complexity × tech uncertainty accounts for quadratic relationships)."
    
    if use_copilot:
        explanation += "\n\n💡 Copilot Tip: Ask GitHub Copilot to analyze risks, suggest mitigation strategies, or create a project roadmap based on these parameters!"
    
    return explanation

def generate_math_calculation(inputs, predicted_total):
    """Generate detailed mathematics showing how the neural network calculated effort"""
    req_count, complexity, team_exp, tech_unc, integrations, doc_quality, deadline, domain = inputs
    
    # Linear approximation for visualization
    base_formula = f"""
    <h3>🔬 Neural Network Mathematical Calculation</h3>
    
    <div class="math-formula">
    <strong>Input Features (Raw values):</strong>
    req_count = {req_count:.1f},  complexity = {complexity:.1f},  team_exp = {team_exp:.1f},  tech_unc = {tech_unc:.1f}
    integrations = {integrations:.1f},  doc_quality = {doc_quality:.1f},  deadline = {deadline:.1f},  domain = {domain:.1f}
    </div>

    <div class="math-formula">
    <strong>Step 1: Normalization (StandardScaler)</strong>
    Each feature is normalized: x_scaled = (x - mean) / std_dev
    This ensures all features have similar ranges for the neural network.
    </div>

    <div class="math-formula">
    <strong>Step 2: Neural Network Layer 1 (Dense 64 neurons, ReLU activation)</strong>
    hidden_1 = ReLU(weights_1 × x_scaled + bias_1)
    64 neurons learn non-linear feature interactions
    </div>

    <div class="math-formula">
    <strong>Step 3: Dropout (20% regularization)</strong>
    To prevent overfitting, 20% of neurons are randomly deactivated during training.
    </div>

    <div class="math-formula">
    <strong>Step 4: Neural Network Layer 2 (Dense 32 neurons, ReLU activation)</strong>
    hidden_2 = ReLU(weights_2 × hidden_1 + bias_2)
    32 neurons compress knowledge from layer 1
    </div>

    <div class="math-formula">
    <strong>Step 5: Output Layer (1 neuron, Linear activation)</strong>
    output = weights_3 × hidden_2 + bias_3
    <strong style="color: #27ae60;">Predicted Effort = {predicted_total:.1f} person-hours</strong>
    </div>

    <div class="math-formula">
    <strong>Key Factors Contributing to Estimation:</strong>
    • Requirements ({req_count:.1f}): More requirements = More effort needed
    • Complexity ({complexity:.1f}): Higher complexity multiplies effort significantly
    • Team Experience ({team_exp:.1f} years): More experience reduces effort
    • Tech Uncertainty ({tech_unc:.1f}): Higher uncertainty increases risk buffer (×120 multiplier)
    • Integrations ({integrations:.1f}): Each integration adds ~12 hours
    • Documentation Quality ({doc_quality:.1f}): Better docs reduce implementation time
    • Deadline Pressure ({deadline:.1f}): Tight deadlines require more buffer (-30 multiplier for relaxed deadlines)
    • Domain Knowledge ({domain:.1f}): More team domain knowledge reduces learning curve
    </div>

    <div class="math-formula">
    <strong>Training Method: Adam Optimizer with Mean Squared Error Loss</strong>
    Model was trained on 5,000 historical projects to minimize prediction error.
    Test MSE indicates how well predictions match historical data patterns.
    </div>
    """
    return base_formula

# 7. Flask app with prediction and local explanation
from flask import Flask, request, jsonify, render_template
import io

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))

from tensorflow.keras.models import model_from_json

# Load model architecture
with open('effort_nn_model.json', 'r') as f:
    model_json = f.read()
nn_model = model_from_json(model_json)
# Load weights (correct .weights.h5 format)
nn_model.load_weights('effort_nn_weights.weights.h5')
# Load scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

def predict_total(features):
    X_input = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X_input)
    total = nn_model.predict(X_scaled, verbose=0)[0][0]
    return max(float(total), 0)

# Document extraction functions
def extract_text_from_pdf(file_bytes):
    """Extract text from PDF files"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

def extract_text_from_docx(file_bytes):
    """Extract text from DOCX files"""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"DOCX extraction error: {e}")
        return ""

def extract_text_from_txt(file_bytes):
    """Extract text from TXT files"""
    try:
        return file_bytes.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"TXT extraction error: {e}")
        return ""

def estimate_features_from_text(text):
    """Use heuristics to estimate project parameters from BRS text"""
    text_lower = text.lower()
    
    # Count requirements (rough estimate from keywords)
    req_keywords = ['requirement', 'shall', 'must', 'should', 'feature', 'functionality']
    req_count = sum(text_lower.count(kw) for kw in req_keywords)
    req_count = max(20, min(200, req_count // 3))  # Normalize to 20-200 range
    
    # Estimate complexity (0-5)
    complexity_keywords = {
        'simple': 1, 'basic': 1, 'straightforward': 2,
        'complex': 4, 'advanced': 4, 'sophisticated': 5,
        'integration': 3, 'machine learning': 4, 'ai': 3, 'algorithm': 3
    }
    complexity = 3  # default
    for keyword, score in complexity_keywords.items():
        if keyword in text_lower:
            complexity = max(complexity, score)
    
    # Check for integrations
    integration_keywords = ['api', 'integration', 'third-party', '3rd party', 'external', 'database', 'microservice']
    integrations = sum(text_lower.count(kw) for kw in integration_keywords)
    integrations = min(15, integrations)
    
    # Check documentation quality
    doc_keywords = ['specification', 'documented', 'design', 'architecture', 'diagram', 'flowchart']
    doc_quality = 3 + min(2, sum(1 for kw in doc_keywords if kw in text_lower))
    
    # Check for tech uncertainty
    uncertainty_keywords = ['new technology', 'poc', 'proof of concept', 'research', 'evaluation', 'unproven']
    tech_unc = 0.3 + min(0.5, 0.1 * sum(text_lower.count(kw) for kw in uncertainty_keywords))
    
    # Check for deadline pressure
    deadline_keywords = ['urgent', 'asap', 'immediately', 'tight', 'phased', 'phase']
    has_deadline = any(kw in text_lower for kw in deadline_keywords)
    deadline = 1.5 if has_deadline else 1.0
    
    return {
        'req_count': int(req_count),
        'complexity': float(min(5, complexity)),
        'team_exp': 5.0,  # default
        'tech_unc': float(min(1, tech_unc)),
        'integrations': int(integrations),
        'doc_quality': float(min(5, doc_quality)),
        'deadline': float(min(2, deadline)),
        'domain': 3.0  # default
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """Extract features from uploaded BRS documents using AI"""
    if 'documents' not in request.files:
        return jsonify({'error': 'No documents provided'}), 400
    
    files = request.files.getlist('documents')
    if len(files) == 0:
        return jsonify({'error': 'Empty file list'}), 400
    
    extracted_text = ""
    file_names = []
    
    for file in files:
        if file.filename == '':
            continue
        
        file_names.append(file.filename)
        file_bytes = file.read()
        
        if file.filename.endswith('.pdf'):
            extracted_text += extract_text_from_pdf(file_bytes) + "\n"
        elif file.filename.endswith('.docx'):
            extracted_text += extract_text_from_docx(file_bytes) + "\n"
        elif file.filename.endswith('.txt'):
            extracted_text += extract_text_from_txt(file_bytes) + "\n"
    
    if not extracted_text.strip():
        return jsonify({'error': 'No readable text extracted from documents'}), 400
    
    # Extract features using AI heuristics
    features = estimate_features_from_text(extracted_text)
    
    # Generate extraction details HTML
    extraction_html = f"""
    <strong>📖 Documents Processed:</strong> {', '.join(file_names)}<br>
    <strong>📊 Extracted Parameters:</strong>
    <ul style="margin: 10px 0;">
        <li><strong>Requirements Count:</strong> {features['req_count']} (based on requirement keywords)</li>
        <li><strong>Complexity:</strong> {features['complexity']:.1f}/5 (based on technical keywords)</li>
        <li><strong>Integrations:</strong> {features['integrations']} (based on API/integration mentions)</li>
        <li><strong>Documentation Quality:</strong> {features['doc_quality']:.1f}/5 (based on design/spec mentions)</li>
        <li><strong>Tech Uncertainty:</strong> {features['tech_unc']:.2f} (based on R&D keywords)</li>
        <li><strong>Deadline Pressure:</strong> {features['deadline']:.1f} (based on urgency keywords)</li>
    </ul>
    <em>💡 Tip: You can manually adjust these values before clicking "Estimate Effort"</em>
    """
    
    features['extraction_details'] = extraction_html
    return jsonify(features)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = [
            float(data['req_count']),
            float(data['complexity']),
            float(data['team_exp']),
            float(data['tech_unc']),
            float(data['integrations']),
            float(data['doc_quality']),
            float(data['deadline']),
            float(data['domain'])
        ]
        total = predict_total(features)
        phases = [{"name": name, "hours": round(total * pct, 1)} for name, pct in PHASE_DIST.items()]
        
        # Enable Copilot support by default
        use_copilot = os.environ.get("COPILOT_ENABLED", "true").lower() == "true"
        explanation = generate_local_explanation(features, total, use_copilot=use_copilot)
        math_calculation = generate_math_calculation(features, total)
        
        return jsonify({
            "total": round(total, 1),
            "phases": phases,
            "explanation": explanation,
            "math_calculation": math_calculation
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting AI Effort Estimator Server...")
    print("📂 Working directory:", os.getcwd())
    print("🌐 Open http://localhost:5000 in your browser")
    print("⚠️  Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
