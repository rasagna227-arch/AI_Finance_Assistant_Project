# AI Finance Assistant

A beginner-friendly Flask + MySQL + scikit-learn project based on the supplied SRS.

## Features
- Registration and login
- Password hashing
- Income and expense management
- ML-based expense categorization
- Budget creation
- Dashboard with totals and category spending
- Basic financial insights
- Spending data foundation for forecasting
- Simple application FAQ chatbot
- JSON prediction API

## Setup

1. Install Python 3.10+.
2. Create a MySQL database:
   `CREATE DATABASE ai_finance;`
3. Open `app.py` and replace `YOUR_MYSQL_PASSWORD` with your MySQL password.
4. Open a terminal in this folder.
5. Run:
   `pip install -r requirements.txt`
6. Train the ML model:
   `python train_model.py`
7. Start the application:
   `python app.py`
8. Open:
   `http://127.0.0.1:5000`

The application automatically creates the required tables.

## Important
This is an educational project. It does not provide professional investment, tax, or financial advice.
