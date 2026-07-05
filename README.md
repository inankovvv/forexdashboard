# Forex Dashboard

A Streamlit dashboard for scanning forex and other market instruments for technical trading signals.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Deploy online

Fastest option: Streamlit Community Cloud.

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Connect the repository and select app.py as the entrypoint.
4. Deploy.
