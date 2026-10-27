# Fake News Detector (BCA Mini Project)

A beginner-friendly web application that demonstrates text classification with Python, Flask, and basic Natural Language Processing (NLP), now with **free internet verification**.

> **Learning demo only:** The included dataset is small and made up for this project. The model learns word patterns in those examples; it does not check facts or decide whether real news is true. Internet verification checks whether similar text appears on the public web; it does not prove a story is true or false. Do not use its result to make real-world decisions.

## What the project does

1. A visitor pastes a headline or a short article into the web page.
2. The app turns the words into numeric TF-IDF features. TF-IDF gives more weight to words that help distinguish examples.
3. A Logistic Regression classifier uses those features to predict the label `FAKE` or `REAL`.
4. **New:** The app also checks the public web (via DuckDuckGo search, completely free, no API key needed) to see if the statement appears online.
5. The ML prediction and internet verification are combined into a final result with an adjusted confidence score.
6. Flask sends the prediction back to the page.

This is supervised machine learning: each training example in `data/news.csv` has a text and a label.

## Project files

- `app.py` — reads the training CSV, trains the text model, runs the Flask website, and combines ML + internet verification.
- `fact_check.py` — free internet verification module (no API key required).
- `data/news.csv` — small example training dataset.
- `templates/index.html` — page layout and form.
- `static/style.css` — page styling.
- `test_app.py` — simple automated tests for the web page and prediction form.
- `requirements.txt` — Python packages the project needs.

## Step 1: Check Python

Open PowerShell or Command Prompt and run:

```powershell
python --version
```

Use Python 3.10 or newer. If `python` is not recognized, install Python and enable the option to add it to PATH, then open a new terminal.

## Step 2: Open the project folder

In PowerShell:

```powershell
cd "C:\Users\Mohammed Maaz\Downloads\fake_news_detector.zip\fake_news_detector"
```

Use the actual folder location where you extracted the project.

## Step 3: Create and activate a virtual environment

A virtual environment keeps this project's packages separate from other Python projects.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## Step 4: Install the packages

```powershell
python -m pip install -r requirements.txt
```

## Step 5: Start the web application

```powershell
python app.py
```

Wait for Flask's local address, usually `http://127.0.0.1:5000`, then open it in your browser. Leave the terminal open while you use the site. Press `Ctrl+C` in the terminal to stop the server.

## Step 6: Try the app

Paste one of the example texts from `data/news.csv`, or try a short sentence of your own, and select **Check this text**. You will see:

- **Final prediction** — REAL or FAKE with overall confidence.
- **ML model** — the text classifier's prediction and confidence.
- **Internet verification** — whether the statement was found on the public web, with sample results.
- **Reason** — a plain-English explanation of how the final decision was reached.

The result is only a prediction learned from this toy dataset plus a web presence check; it is not a fact-check.

## Step 7: Run the tests

With the virtual environment active, open another terminal in the project folder and run:

```powershell
python -m unittest
```

The tests check that the home page loads, that submitting text returns a prediction, and that blank text does not show a prediction.

## Beginner notes

- **Flask** is a Python library for building web applications.
- **NLP** means working with human language using computers.
- **TF-IDF** turns words into numbers that a machine-learning model can use.
- **Logistic Regression** is the classifier that learns patterns from the labeled examples.
- **Confidence** is the model's score for its selected label. It is not proof that a story is true.
- **Internet verification** uses DuckDuckGo HTML search (free, no API key). It may not work if you are offline or if the search endpoint is blocked.

For a stronger academic project, replace the sample rows with a larger, properly sourced and labeled dataset, keep training and test data separate, and report evaluation scores. Avoid claiming that this small demo detects misinformation reliably.
