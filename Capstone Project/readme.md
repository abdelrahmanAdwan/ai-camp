
# Global Land Temperature Dashboard 

An interactive web dashboard for exploring global temperature trends from **1850 to today**, built with Python and deployed straight from Google Colab. This is the capstone project of the **AI RAG Agent MCP Training Plan**.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![Plotly](https://img.shields.io/badge/Plotly-charts-purple)

---

## Project Overview

The dashboard loads a global temperature-anomaly dataset, cleans it with Pandas, and presents it as a live, interactive web app. Users can pick a data source and a range of years and instantly see how global temperatures have changed — a small, shareable tool for exploring climate trends.

**Temperature anomaly** = how much warmer or cooler a year was compared to a historical baseline (a positive value means warmer than baseline).

## 🎯 Objectives

- Combine **data cleaning**, **app development**, and **visualization** into one seamless project.
- Build a user-friendly UI with interactive Streamlit widgets (slider, radio, expander).
- Create interactive Plotly charts that react instantly to user input.
- Learn the basics of deploying a Python web app to a public URL from the cloud.

## 🧰 Technologies & Frameworks

| Tool | Role |
|------|------|
| 🐼 **Pandas** | Load and clean the temperature data |
| ✨ **Streamlit** | Build the web app UI |
| 📊 **Plotly Express** | Interactive line charts |
| 🔗 **pyngrok /  ngrok** | Expose the app on a public, shareable URL |
| ☁️ **Google Colab** | Cloud runtime for the whole project |

## ✨ Features

- **Title & description** explaining the dashboard.
- **Sidebar controls**: a data-source selector (GCAG / GISTEMP) and a **year-range slider**.
- **Key metrics**: warmest year, average anomaly, and the selected range.
- **Dynamic line chart** that updates on the fly as you move the slider.
- **Expandable raw-data section** with a **CSV download** button.

## 🚀 How to Run

### Option A — Google Colab (recommended)
1. Open `climate_dashboard.ipynb` in [Google Colab](https://colab.research.google.com).
2. Get a free ngrok token from [dashboard.ngrok.com](https://dashboard.ngrok.com/signup) and paste it into **Step 4**.
3. Click **Runtime → Run all**.
4. Open the public URL printed in **Step 5**. 🎉

### Option B — Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## 📁 Project Structure

```
capstone_climate_dashboard/
├── app.py                   # Streamlit dashboard application
├── climate_dashboard.ipynb  # Colab notebook: install → clean → write app → deploy
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── LINKEDIN_POST.md         # Draft LinkedIn portfolio post
└── screenshots/             # Proof-of-work screenshots (add your own)
```

## 📊 Data Source

Global temperature anomaly series (NOAA **GCAG** & NASA **GISTEMP**), via the
[datahub.io global-temp dataset](https://github.com/datasets/global-temp).


---

Built as part of the AI RAG Agent MCP Training Plan.
