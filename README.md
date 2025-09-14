# Final-Deployment--WRM

💧 Global Water Consumption – Final Deployment
🌍 Project Overview

This project focuses on analyzing and visualizing global water consumption patterns.
After data cleaning and regression modeling (Week 2), the final step is deployment.
An interactive Streamlit dashboard has been created to make the dataset accessible and insightful for end-users.

📂 Dataset

File: cleaned_global_water_consumption.csv

Main Columns:

Year

Agricultural Water Use (%)

Industrial Water Use (%)

Household Water Use (%)

Rainfall Impact (Annual Precipitation in mm)

Groundwater Depletion Rate (%)

Target: Total Water Consumption (Billion Cubic Meters)

⚙️ Tech Stack

Python 3.x

Libraries:

pandas → data handling

matplotlib → visualization

streamlit → deployment/dashboard

📊 Features of the App

Interactive data preview (head of dataset)

Summary statistics of water consumption factors

Dynamic histograms for numerical columns

Year filter to explore specific time periods

User-friendly and ready for deployment on Kaggle or Streamlit Cloud

▶️ Running the App
🔹 Local Machine

Place app.py and cleaned_global_water_consumption.csv in the same folder

Install dependencies (if needed):

pip install pandas matplotlib streamlit


Run the app:

streamlit run app.py


Open the given local URL in your browser
🔹 Kaggle Deployment

Create a Kaggle Notebook

Upload:

app.py

cleaned_global_water_consumption.csv

Run this in a notebook cell:

streamlit run app.py --server.port 6006 --server.address 0.0.0.0


Open the app via the “Web” tab

✅ Final Notes
This README represents the final deployment stage of the Week2-WRM Project

Earlier modules included data cleaning and regression model training

Deployment ensures the project is accessible, interactive, and ready for demonstration

Future enhancements: integrate regression model predictions into the dashboard
