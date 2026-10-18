# SmartCart — Streamlit Customer Segmentation App

An interactive Streamlit dashboard based on the uploaded `smartcart` ML notebook. It performs feature engineering, one-hot encoding, standardization, PCA, and customer clustering, then presents segment profiles, purchase behaviour, campaign response, a 3D PCA map, and downloadable CSV exports.

## Run locally

1. Install Python 3.10 or newer.
2. Put `app.py` and `requirements.txt` in a folder.
3. Open a terminal in that folder and run:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -r requirements.txt
streamlit run app.py
```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`).
5. Upload `smartcart_customers.csv` using the sidebar.

## Dataset

The app expects the customer CSV used by the notebook. Required columns include `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, the six `Mnt...` spending columns, purchase-channel counts, `NumWebVisitsMonth`, `Complain`, and `Response`. `ID` is optional for model fitting but is retained in the displayed/exported customer table if present.

## Notes

- Default configuration mirrors the notebook's 4-cluster Agglomerative Clustering workflow.
- K-Means is available as an alternative.
- Age is computed as `2026 - Year_Birth`, matching the notebook.
- The app uses the notebook's age and income outlier filters.
- Segment names are heuristic descriptions based on cluster averages; validate them before using them for decisions.
