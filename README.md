# 🛡️ CyberSmart — AI-Powered Phishing Detection



**A privacy-preserving mobile security application that detects phishing URLs using on-device AI and Federated Learning.**

---

## 📱 About

CyberSmart is a final year project that combines **Federated Learning** with **on-device TFLite inference** to detect phishing URLs on Android devices — with **no internet required** and **zero data leaving the device**.

> 🔒 **Privacy First**: Your URLs never leave your phone. The AI model runs entirely on-device.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **On-Device AI** | TFLite model runs directly on phone — no server, no internet |
| 🔒 **Privacy Preserving** | URLs never transmitted anywhere — complete offline operation |
| ⚡ **Ultra Fast** | ~50ms inference using 56 URL features |
| 📱 **URL Scanner** | Paste any URL and get instant phishing/safe verdict |
| 💬 **SMS/Email Scanner** | Paste suspicious messages — app extracts and scans all links |
| 📊 **Scan History** | Local history with filter by safe/phishing and statistics |
| 🛡️ **Whitelist Layer** | Trusted domains (Google, HBL, etc.) bypass ML for speed |
| 🌐 **Federated Learning** | Privacy-preserving distributed training simulation (5 clients, 10 rounds) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Flutter Mobile App                │
│  ┌─────────┐ ┌───────────┐ ┌───────────┐   │
│  │ Scanner │ │SMS/Email  │ │  History  │   │
│  └────┬────┘ └─────┬─────┘ └───────────┘   │
│       └────────────┘                        │
│              │                              │
│   ┌──────────▼──────────┐                  │
│   │   PhishingDetector  │                  │
│   │  ┌───────────────┐  │                  │
│   │  │ Whitelist     │  │                  │
│   │  │ Check         │  │                  │
│   │  └───────┬───────┘  │                  │
│   │          │           │                  │
│   │  ┌───────▼───────┐  │                  │
│   │  │ Feature       │  │                  │
│   │  │ Extraction    │  │                  │
│   │  │ (56 features) │  │                  │
│   │  └───────┬───────┘  │                  │
│   │          │           │                  │
│   │  ┌───────▼───────┐  │                  │
│   │  │ TFLite Model  │  │                  │
│   │  │ (on-device)   │  │                  │
│   │  └───────┬───────┘  │                  │
│   │          │           │                  │
│   │  ┌───────▼───────┐  │                  │
│   │  │ Risk Level    │  │                  │
│   │  │ Assignment    │  │                  │
│   │  └───────────────┘  │                  │
│   └─────────────────────┘                  │
└─────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Mobile App** | Flutter 3.x + Dart |
| **On-Device ML** | TensorFlow Lite (TFLite) |
| **Model Training** | PyTorch 2.x |
| **Federated Learning** | Custom FedAvg implementation |
| **Backend API** | Python Flask (optional, for testing) |
| **Storage** | SharedPreferences (local) |

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **96.67%** |
| **F1 Score** | 0.9656 |
| **AUC-ROC** | 0.9946 |
| **Precision** | 98.40% |
| **Recall** | 94.78% |
| **Production Benchmark** | **55/55 = 100%** |

### Federated Learning vs Centralised

| Approach | Accuracy | F1 | AUC-ROC |
|----------|----------|-----|---------|
| **Federated (10 rounds)** | **96.67%** | **0.9656** | **0.9946** |
| Centralised Baseline | 96.67% | 0.9656 | 0.9956 |
| Gap | **0.00%** | **0.0000** | 0.0010 |

> **Key Finding**: Federated model achieves exact accuracy parity with centralised training — privacy comes at zero accuracy cost.

### Federated Learning Convergence

| Round | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| 1 | 88.10% | 0.8642 | 0.9815 |
| 2 | 94.57% | 0.9427 | 0.9897 |
| 5 | 96.15% | 0.9600 | 0.9938 |
| 8 | 96.91% | 0.9683 | 0.9945 |
| 10 | 96.67% | 0.9656 | 0.9946 |

### Production Benchmark (55 URLs)

| Attack Type | Detected | Rate |
|-------------|----------|------|
| Domain Spoofing | 8/8 | 100% |
| Subdomain Spoofing | 4/4 | 100% |
| IP-based Phishing | 3/3 | 100% |
| Suspicious TLD | 6/6 | 100% |
| Brand Impersonation | 2/2 | 100% |
| Legitimate Domains | 30/30 | 100% |
| **TOTAL** | **55/55** | **100%** |

---

## 📁 Project Structure

```
CyberSmart/
│
├── 📱 mobile_app/                    # Flutter Application
│   ├── lib/
│   │   ├── main.dart                 # App entry point
│   │   ├── models/
│   │   │   └── scan_result.dart      # Data model
│   │   ├── services/
│   │   │   ├── phishing_detector.dart # On-device TFLite inference
│   │   │   └── history_service.dart  # Local history storage
│   │   ├── screens/
│   │   │   ├── scanner_screen.dart   # URL Scanner
│   │   │   ├── text_scanner_screen.dart # SMS/Email Scanner
│   │   │   └── history_screen.dart  # Scan History
│   │   └── widgets/
│   │       └── shared_widgets.dart  # Reusable UI components
│   ├── assets/
│   │   ├── model.tflite             # TFLite AI model (~50KB)
│   │   └── model_info.json          # Model configuration
│   └── pubspec.yaml
│
├── 🤖 ml_backend/                   # Machine Learning
│   ├── model.py                     # Neural network architecture
│   ├── feature_extractor.py         # 56 URL feature extraction
│   ├── train_on_dataset.py          # Model training script
│   ├── app.py                       # Flask REST API
│   ├── convert_to_tflite.py         # PyTorch → TFLite conversion
│   ├── bulk_test.py                 # Production benchmark (55 URLs)
│   └── dataset_phishing.csv         # Training dataset (11,430 URLs)
│
├── 🔒 federated_learning/           # FL Simulation
│   ├── federated_simulation.py      # FedAvg implementation (5 clients, 10 rounds)
│   └── generate_figures.py          # Research paper figures
│
└── 📄 README.md
```

---

## 🚀 Installation

### Prerequisites

- Flutter SDK 3.x
- Python 3.10+
- Android device (Android 8.0+ / API 26+)
- Android Studio or VS Code

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cybersmart.git
cd cybersmart
```

### 2. ML Model Setup (Python)

```bash
cd ml_backend
pip install torch tensorflow scikit-learn pandas flask flask-cors
python train_on_dataset.py        # Train the model
python convert_to_tflite.py       # Convert to TFLite
```

### 3. Copy Model Files

```bash
cp ml_backend/model.tflite  mobile_app/assets/
cp ml_backend/model_info.json mobile_app/assets/
```

### 4. Run Flutter App

```bash
cd mobile_app
flutter pub get
flutter run
```

### 5. Build Release APK

```bash
flutter build apk --release
# APK: build/app/outputs/flutter-apk/app-release.apk
```

---

## 🔬 Federated Learning Simulation

Run the complete FL simulation:

```bash
cd federated_learning
python federated_simulation.py
```

**Configuration:**
- Clients: 5
- Rounds: 10
- Local epochs: 3
- Client fraction: 80%
- Data split: IID
- Algorithm: FedAvg (McMahan et al., 2017)

---

## 📱 App Screenshots

> Scanner Screen | SMS Scanner | History Screen

*(Add your screenshots here)*

---

## 🧠 Neural Network Architecture

```
Input (56 features)
    │
    ▼
[FC: 56 → 128] → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
[FC: 128 → 64] → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
[FC: 64 → 32]  → BatchNorm → ReLU → Dropout(0.3)
    │
    ▼
[FC: 32 → 16]  → ReLU
    │
    ▼
[FC: 16 → 1]   → Sigmoid
    │
    ▼
Output (0-1 probability)
Threshold: 0.55 → PHISHING / SAFE
```

**Total Parameters:** ~19,000  
**Training:** Adam optimizer, lr=0.001, BCELoss  
**Dataset:** 11,430 URLs (50/50 balanced)

---

## 🔒 Privacy Analysis

| Property | Traditional ML | CyberSmart (FL) |
|----------|---------------|-----------------|
| Raw URL sent to server | ✅ Yes | ❌ No |
| Server stores history | ✅ Yes | ❌ No |
| Works offline | ❌ No | ✅ Yes |
| GDPR compliant | ⚠️ Risk | ✅ Yes |
| Breach impact | 🔴 High | 🟢 Low |

---

## 📖 Research

This project is based on and extends:

- **McMahan et al. (2017)** — Communication-Efficient Learning of Deep Networks from Decentralized Data
- **Sahingoz et al. (2019)** — Machine learning based phishing detection from URLs
- **Preuveneers et al. (2018)** — Chained anomaly detection models for federated learning

**Citation:**
```bibtex
@article{malik2025cybersmart,
  title={Federated Learning for Privacy-Preserving Phishing URL Detection on Mobile Devices},
  author={Malik, Zybii},
  year={2025},
  journal={Unpublished Manuscript}
}
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License .

---

## 👨‍💻 Author

**Muhammad Faisal**  
Final Year Information Technology Student  


---
