from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response
from router.smart_query_router import detect_and_route
from dotenv import load_dotenv
import os
import unicodedata

# === Initialize App ===
load_dotenv()
app = Flask(__name__)
app.secret_key = 'foresight-secret'  # 🚨 Replace with env var for production
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# === In-Memory Demo Users ===
users = {
    "admin@foresight.ai": "123456",
    "nykaa@insights.com": "nykaa2024"
}

# === Unicode-safe text sanitizer ===
def safe_unicode(text):
    return unicodedata.normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")

# === Home Page ===
@app.route("/")
def home():
    return render_template("index.html", user=session.get("user"))

# === Login ===
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        if users.get(email) == password:
            session["user"] = email
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials", "danger")
    return render_template("login.html")

# === Logout ===
@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

# === About Page ===
@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))

# === Query Handler ===
@app.route("/query", methods=["POST"])
def handle_query():
    user_query = request.form.get("query", "")
    try:
        result = detect_and_route(user_query)
    except Exception as e:
        result = f"❌ Something went wrong while processing your query.\n\nError: {str(e)}"
        flash("⚠️ Failed to process your query. Please try again.", "warning")

    clean_result = safe_unicode(result)
    return render_template("result.html", result=clean_result, query=user_query, user=session.get("user"))

# === Run Server ===
if __name__ == "__main__":
    app.run(debug=True)
