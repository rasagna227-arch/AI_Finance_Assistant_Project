from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
from functools import wraps
import joblib
import os

app = Flask(__name__)
app.secret_key = "change-this-secret-key"

# MySQL configuration:
# Create a database named ai_finance before running.
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:MySQL%40123@localhost/ai_finance"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

MODEL_PATH = os.path.join("model", "expense_classifier.joblib")

try:
    classifier = joblib.load(MODEL_PATH)
except Exception:
    classifier = None


class User(db.Model):
    __tablename__ = "users"
    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Transaction(db.Model):
    __tablename__ = "transactions"
    transaction_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    date = db.Column(db.Date, nullable=False)
    description = db.Column(db.String(255), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(20), nullable=False)  # income / expense
    category = db.Column(db.String(50), nullable=False)


class Budget(db.Model):
    __tablename__ = "budgets"
    budget_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    month = db.Column(db.String(7), nullable=False)  # YYYY-MM
    category = db.Column(db.String(50), nullable=False, default="Overall")
    limit_amount = db.Column(db.Float, nullable=False)


class Prediction(db.Model):
    __tablename__ = "predictions"
    prediction_id = db.Column(db.Integer, primary_key=True)
    transaction_id = db.Column(db.Integer, db.ForeignKey("transactions.transaction_id"), nullable=False)
    predicted_category = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Insight(db.Model):
    __tablename__ = "insights"
    insight_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.user_id"), nullable=False)
    insight_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def predict_category(description):
    if classifier is None:
        return "Other", 0.0
    try:
        pred = classifier.predict([description])[0]
        confidence = 0.0
        if hasattr(classifier, "predict_proba"):
            confidence = float(max(classifier.predict_proba([description])[0]))
        return pred, confidence
    except Exception:
        return "Other", 0.0


def make_insights(user_id):
    rows = Transaction.query.filter_by(user_id=user_id, type="expense").all()
    if not rows:
        return ["Add some expenses to receive spending insights."]

    totals = {}
    for row in rows:
        totals[row.category] = totals.get(row.category, 0) + row.amount

    highest = max(totals, key=totals.get)
    total_expense = sum(totals.values())
    message = f"Your highest spending category is {highest} with ₹{totals[highest]:.2f}."
    result = [message]

    current_month = date.today().strftime("%Y-%m")
    budget = Budget.query.filter_by(
        user_id=user_id, month=current_month, category="Overall"
    ).first()

    if budget:
        if total_expense > budget.limit_amount:
            result.append("Your recorded expenses are above the current overall budget.")
        elif total_expense >= budget.limit_amount * 0.8:
            result.append("You have used 80% or more of your current overall budget.")

    return result


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")
            return render_template("register.html")

        user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.user_id
            session["user_name"] = user.name
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    user_id = session["user_id"]
    transactions = Transaction.query.filter_by(user_id=user_id).order_by(
        Transaction.date.desc(), Transaction.transaction_id.desc()
    ).all()

    income = sum(t.amount for t in transactions if t.type == "income")
    expenses = sum(t.amount for t in transactions if t.type == "expense")
    balance = income - expenses

    category_totals = {}
    for t in transactions:
        if t.type == "expense":
            category_totals[t.category] = category_totals.get(t.category, 0) + t.amount

    insights = make_insights(user_id)

    return render_template(
        "dashboard.html",
        transactions=transactions,
        income=income,
        expenses=expenses,
        balance=balance,
        category_totals=category_totals,
        insights=insights
    )


@app.route("/transactions/add", methods=["GET", "POST"])
@login_required
def add_transaction():
    if request.method == "POST":
        trans_type = request.form["type"]
        description = request.form["description"].strip()
        amount = float(request.form["amount"])
        trans_date = datetime.strptime(request.form["date"], "%Y-%m-%d").date()

        if amount <= 0:
            flash("Amount must be greater than zero.", "danger")
            return render_template("add_transaction.html")

        category = request.form.get("category", "").strip() or "Other"
        confidence = 0.0

        if trans_type == "expense":
            category, confidence = predict_category(description)

        transaction = Transaction(
            user_id=session["user_id"],
            date=trans_date,
            description=description,
            amount=amount,
            type=trans_type,
            category=category
        )
        db.session.add(transaction)
        db.session.flush()

        if trans_type == "expense":
            db.session.add(Prediction(
                transaction_id=transaction.transaction_id,
                predicted_category=category,
                confidence=confidence
            ))

        db.session.commit()
        flash(f"Transaction added. Category: {category}", "success")
        return redirect(url_for("dashboard"))

    return render_template("add_transaction.html", today=date.today().isoformat())


@app.route("/transactions/delete/<int:transaction_id>", methods=["POST"])
@login_required
def delete_transaction(transaction_id):
    transaction = Transaction.query.filter_by(
        transaction_id=transaction_id, user_id=session["user_id"]
    ).first_or_404()

    Prediction.query.filter_by(transaction_id=transaction_id).delete()
    db.session.delete(transaction)
    db.session.commit()
    flash("Transaction deleted.", "success")
    return redirect(url_for("dashboard"))


@app.route("/budgets", methods=["GET", "POST"])
@login_required
def budgets():
    user_id = session["user_id"]

    if request.method == "POST":
        month = request.form["month"]
        category = request.form["category"].strip() or "Overall"
        limit_amount = float(request.form["limit_amount"])

        if limit_amount <= 0:
            flash("Budget must be greater than zero.", "danger")
            return redirect(url_for("budgets"))

        budget = Budget(
            user_id=user_id,
            month=month,
            category=category,
            limit_amount=limit_amount
        )
        db.session.add(budget)
        db.session.commit()
        flash("Budget created.", "success")
        return redirect(url_for("budgets"))

    all_budgets = Budget.query.filter_by(user_id=user_id).order_by(
        Budget.month.desc()
    ).all()

    return render_template("budgets.html", budgets=all_budgets)


@app.route("/api/predict-category", methods=["POST"])
@login_required
def api_predict_category():
    data = request.get_json(silent=True) or {}
    description = str(data.get("description", "")).strip()

    if not description:
        return jsonify({"error": "Description is required."}), 400

    category, confidence = predict_category(description)
    return jsonify({
        "category": category,
        "confidence": round(confidence, 4)
    })


@app.route("/chatbot", methods=["GET", "POST"])
@login_required
def chatbot():
    answer = None
    if request.method == "POST":
        question = request.form["question"].lower()

        if "budget" in question:
            answer = "Open the Budget page to create and monitor a monthly budget."
        elif "expense" in question or "transaction" in question:
            answer = "Use Add Transaction to record an income or expense. Expense descriptions are categorized by the ML model."
        elif "balance" in question:
            answer = "Your dashboard shows total income, expenses, and remaining balance."
        elif "category" in question:
            answer = "Expense descriptions are classified into categories such as Food, Transport, Shopping, Bills, Entertainment, Health, Education, and Other."
        else:
            answer = "I can answer general questions about transactions, budgets, categories, balance, and the dashboard."

    return render_template("chatbot.html", answer=answer)


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)
