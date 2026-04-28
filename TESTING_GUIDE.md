# 🧪 Testing Guide - AI Effort Estimator

## Quick Verification Checklist

### ✅ Server Status
```bash
# Check if server is running on port 5000
lsof -i :5000  # or netstat -tlnp | grep 5000
```
**Expected**: Python process running `upgraded_estimator.py`

---

## 🧪 Test Scenarios

### Test 1: Web Interface Loading
1. Open browser: `http://localhost:5000`
2. **Expected**: 
   - ✅ Title: "🤖 AI-Powered Effort Estimator with BRS Document Analysis"
   - ✅ Three sections visible: Upload, Parameters, Results
   - ✅ Upload button: "🔍 Extract Features with AI/LLM"
   - ✅ Estimate button: "🔮 Estimate Effort & Show Math"

---

### Test 2: Manual Parameter Estimation (Without Upload)
1. Keep default values or change them
2. Click "🔮 Estimate Effort & Show Math"
3. **Expected Results**:
   - ✅ Total effort shows (e.g., "📊 Total Estimated Effort: 850.5 person‑hours")
   - ✅ Bar chart displays with phases
   - ✅ Table shows breakdown (10 phases)
   - ✅ **Mathematical Calculations section visible** showing:
     - Input features
     - Normalization step
     - Neural network layers (Layer 1: 64 neurons, Layer 2: 32 neurons)
     - Output calculation
     - Key factors explanation
   - ✅ AI explanation visible at bottom

---

### Test 3: Create Sample BRS Document
```bash
# Create a test BRS file
cat > /tmp/sample_brs.txt << 'EOF'
# Business Requirements Specification - Project Alpha

## 1. Requirements
The system shall provide the following requirements:
1. User authentication and authorization
2. Dashboard for project management
3. Real-time data visualization
4. API integration with external services
5. Database management system
6. Reporting functionality
7. Mobile application support
8. Advanced analytics features

## 2. Technical Complexity
This project involves:
- Complex machine learning algorithms
- Advanced data processing
- Third-party API integrations
- Microservice architecture
- Real-time data synchronization
- Mobile app development

## 3. Integrations
- Salesforce CRM integration
- Google Analytics integration
- AWS cloud services
- Payment gateway (Stripe)
- Email service provider

## 4. Documentation
- Complete API specifications
- System architecture diagrams
- Database schema documentation
- User flow diagrams

## 5. Timeline
This is a high-priority project with tight deadlines.
Required delivery: ASAP

## 6. Risks
- New technology adoption
- Unproven tool evaluation needed
- Research phase required for ML algorithms
EOF
```

### Test 4: Upload & Extract Features
1. Select the BRS file (from Test 3)
2. Click "🔍 Extract Features with AI/LLM"
3. **Expected**:
   - ✅ Green success message appears
   - ✅ Shows "Documents Processed: sample_brs.txt"
   - ✅ Parameters extracted:
     - Requirements Count: ~50-80 (based on keyword count)
     - Complexity: 4-5/5 (due to ML/advanced keywords)
     - Integrations: 4-5 (detected API mentions)
     - Documentation Quality: 3-4/5 (spec mentions)
     - Tech Uncertainty: 0.4-0.6 (POC/research keywords)
   - ✅ Form fields auto-populated with extracted values

---

### Test 5: Test Math Calculations Display
1. After extraction or manual entry, click "🔮 Estimate Effort & Show Math"
2. Scroll to "🧮 Mathematical Calculations" section
3. **Verify all steps visible**:
   - ✅ Shows "Input Features (Raw values)"
   - ✅ Shows "Step 1: Normalization (StandardScaler)"
   - ✅ Shows "Step 2: Neural Network Layer 1 (Dense 64 neurons, ReLU activation)"
   - ✅ Shows "Step 3: Dropout (20% regularization)"
   - ✅ Shows "Step 4: Neural Network Layer 2 (Dense 32 neurons, ReLU activation)"
   - ✅ Shows "Step 5: Output Layer (1 neuron, Linear activation)"
   - ✅ **Final effort number in green**: "Predicted Effort = X person-hours"
   - ✅ Shows key factors (complexity impact, tech uncertainty importance, etc.)
   - ✅ Training method explanation visible

---

### Test 6: Copilot Support Tip
1. Complete an estimation
2. Scroll to "💡 AI-Generated Explanation"
3. **Expected**: 
   - ✅ Explanation mentions "Copilot Tip"
   - ✅ Suggests: "Ask GitHub Copilot to analyze risks, suggest mitigation strategies, or create a project roadmap"

---

### Test 7: API Endpoint Testing
```bash
# Test /extract_features endpoint
curl -X POST -F "documents=@/tmp/sample_brs.txt" \
  http://localhost:5000/extract_features | python -m json.tool

# Test /predict endpoint
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "req_count": 60,
    "complexity": 3.5,
    "team_exp": 5,
    "tech_unc": 0.5,
    "integrations": 5,
    "doc_quality": 3,
    "deadline": 1.0,
    "domain": 3
  }' \
  http://localhost:5000/predict | python -m json.tool
```

**Expected Response**:
```json
{
  "total": 850.5,
  "phases": [
    {"name": "Requirement analysis", "hours": 68.0},
    {"name": "Development", "hours": 297.7},
    ...
  ],
  "explanation": "The neural network predicts a moderate total effort...",
  "math_calculation": "<div class=\"math-formula\">..."
}
```

---

## 📊 Expected Estimation Ranges

| Input Type | Total Hours | Interpretation |
|------------|-------------|-----------------|
| Simple (req=20, complexity=1) | 200-300 | Small project |
| Moderate (req=60, complexity=3) | 700-900 | Medium project |
| Complex (req=100, complexity=5) | 1200-1500 | Large project |
| High Risk (tech_unc=0.8+) | +200-300 | Add risks buffer |
| Tight Deadline | +100-200 | Pressure buffer |

---

## 🐛 Troubleshooting

### Issue: "Neural network model not found"
```bash
# Check if model files exist
ls -la effort_nn_*.* scaler.pkl
```
**Solution**: Re-run the script to generate models

### Issue: "Port 5000 already in use"
```bash
# Kill existing process
pkill -f upgraded_estimator.py
# Or use different port (modify code)
```

### Issue: "Document extraction returns empty"
- Check file format (PDF, DOCX, or TXT)
- Ensure file has readable text
- Try with sample test file from Test 3

### Issue: "Math calculations not showing"
- Refresh browser (Ctrl+F5)
- Check browser console for JS errors
- Verify `/predict` endpoint returns `math_calculation` field

---

## ✅ Success Criteria

All of these should be working:

- [ ] Web interface loads
- [ ] Default estimation works
- [ ] Math calculations display correctly
- [ ] Document upload works
- [ ] Feature extraction displays parameters
- [ ] Form auto-fills with extracted values
- [ ] Copilot tips visible
- [ ] API endpoints respond correctly
- [ ] Charts render properly
- [ ] Tables show all 10 phases

**If all boxes are checked: ✅ SYSTEM IS READY FOR PRODUCTION!**

---

## 📝 Sample Test BRS

See `/tmp/sample_brs.txt` created in Test 3 for a realistic example.

---

*Last Updated: 2026-04-28*
*System Version: 1.0 with Copilot Support*
