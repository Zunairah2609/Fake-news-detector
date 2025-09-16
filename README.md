A machine learning project that detects whether a news article is real or fake.
Built with Python, scikit-learn, and Flask for the web interface.

Live Demo: https://fake-news-detector-4-2rnl.onrender.com/

Features
- Pre-trained ML model (model.joblib + vectorizer.joblib)
- Web interface using Flask (app.py)
- Simple UI (templates + static files)
- CLI script (predict.py) for quick testing

Project structure (high-level)
News detector/
- Data/ (Fake.csv, True.csv)
- static/ (CSS, images)
- templates/ (HTML)
- app.py
- predict.py
- news.py
- model.joblib
- vectorizer.joblib
- requirements.txt

 Quick start
1. Create & activate a virtual environment:
   - `python -m venv .venv`
   - On Windows: `.venv\Scripts\activate`  | On macOS/Linux: `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run the Flask app:
   - `python app.py`
   - Open http://127.0.0.1:5000

## Notes
- `Data/` CSV files are large (>50MB). Consider using Git LFS for large files.
- If model files are not present, you can include them as release assets or use Git LFS.

Author
Zunairah Fathima

