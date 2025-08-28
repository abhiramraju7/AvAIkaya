from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import random
import numpy as np
from scipy.optimize import curve_fit
from fpdf import FPDF
import os
import openai
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
import os
from flask import send_file
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///store14.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'this_is_a_random_secret_key'
db = SQLAlchemy(app)

from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='customer')
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)  # ✅ New column added

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


    def __repr__(self):
        return f"<User {self.username} ({self.role})>"



# Customer Class inherits from User
class Customer(User):
    __tablename__ = 'customer'
    id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)

    def __repr__(self):
        return f"<Customer {self.username}>"

# Manager Class inherits from User
class Manager(User):
    __tablename__ = 'manager'
    id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)

    def __repr__(self):
        return f"<Manager {self.username}>"

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.String(500), nullable=True)
    price = db.Column(db.Float, nullable=False)
    image = db.Column(db.String(255), nullable=True)  # ✅ Added image field

    def __repr__(self):
        return f"<Product {self.name}>"



class CartItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)  # Foreign key to User
    quantity = db.Column(db.Integer, default=1)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)  # Foreign key for Product
    price = db.Column(db.Float, nullable=False)

    user = db.relationship('User', backref='cart_items_list', lazy=True)
    product = db.relationship('Product', backref='cart_items', lazy=True)

    def __repr__(self):
        return f"<CartItem {self.product.name} (x{self.quantity})>"

class CheckoutItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    product = db.relationship('Product', backref='checkout_items', lazy=True)

    def __repr__(self):
        return f"<CheckoutItem {self.product.name} (x{self.quantity})>"

class UserPurchaseHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, nullable=False)
    total_amount_spent = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    product = db.relationship('Product', backref='user_purchase_history', lazy=True)
    user = db.relationship('User', backref='user_purchase_history', lazy=True)

    def __repr__(self):
        return f"<UserPurchaseHistory {self.user_id} bought {self.quantity} of {self.product.name} for {self.total_amount_spent}>"

def init_pickles():
    if not Product.query.all():
        pickles = [
            Product(name="Cucumber Avakaya", description="Spicy cucumber pickle", price=5.99,
                    image="/static/images/cucumber_pickle.jpg"),
            Product(name="Ginger Avakaya", description="Zesty ginger pickle", price=6.49,
                    image="/static/images/ginger_pickle.jpg"),
            Product(name="Mango Avakaya", description="Tangy mango pickle", price=7.99,
                    image="/static/images/mango_pickle.jpg"),
            Product(name="Lemon Avakaya", description="Sour lemon pickle", price=4.99,
                    image="/static/images/lemon_pickle.jpeg"),
            Product(name="Garlic Avakaya", description="Aromatic garlic pickle", price=6.99,
                    image="/static/images/garlic_pickle.jpg"),
        ]

        db.session.add_all(pickles)
        db.session.commit()



class Ingredient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    total_stock = db.Column(db.Float, nullable=False)  # Total available quantity (in kg, g, liters, etc.)

    def __repr__(self):
        return f"<Ingredient {self.name}, Stock: {self.total_stock}>"

class ProductIngredient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    ingredient_id = db.Column(db.Integer, db.ForeignKey('ingredient.id'), nullable=False)
    quantity_needed = db.Column(db.Float, nullable=False)  # Quantity needed per order

    product = db.relationship('Product', backref='ingredients')
    ingredient = db.relationship('Ingredient', backref='used_in_products')

# Initial stock data in client inventory
def init_ingredients():
    if not Ingredient.query.all():
        ingredients = [
            Ingredient(name="Chilli Powder", total_stock=100.0),  # 100 kg
            Ingredient(name="Garlic", total_stock=10.0),
            Ingredient(name="Vinegar", total_stock=200.0),  # 200 liters
            Ingredient(name="Mustard Seeds", total_stock=80.0),
            Ingredient(name="Oil", total_stock=500.0),
            Ingredient(name="Salt", total_stock=300.0),
        ]
        db.session.add_all(ingredients)
        db.session.commit()

def init_product_ingredients():
    if not ProductIngredient.query.all():
        relations = [
            # Cucumber Pickle
            ProductIngredient(product_id=1, ingredient_id=1, quantity_needed=0.1),  # Chilli Powder (100g)
            ProductIngredient(product_id=1, ingredient_id=3, quantity_needed=0.2),  # Vinegar (200ml)
            ProductIngredient(product_id=1, ingredient_id=5, quantity_needed=0.5),  # Oil (500ml)

            # Mango Pickle
            ProductIngredient(product_id=2, ingredient_id=1, quantity_needed=0.15),  # Chilli Powder (150g)
            ProductIngredient(product_id=2, ingredient_id=4, quantity_needed=0.05),  # Mustard Seeds (50g)
            ProductIngredient(product_id=2, ingredient_id=5, quantity_needed=0.3),  # Oil (300ml)

            # Garlic Pickle
            ProductIngredient(product_id=3, ingredient_id=2, quantity_needed=0.25),  # Garlic (250g)
            ProductIngredient(product_id=3, ingredient_id=5, quantity_needed=0.4),  # Oil (400ml)
            ProductIngredient(product_id=3, ingredient_id=6, quantity_needed=0.1),  # Salt (100g)
        ]
        db.session.add_all(relations)
        db.session.commit()

with app.app_context():
    db.create_all()
    init_ingredients()
    init_product_ingredients()
    init_pickles()

def is_logged_in():
    return 'user_id' in session

@app.route('/home')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session.get('user_id')

    products = Product.query.all()
    recommended_products = recommend_products(user_id)

    return render_template(
        'index.html',
        products=products,
        recommended_products=recommended_products
    )


@app.route('/add_to_cart/<int:product_id>', methods=['POST'])
def add_to_cart(product_id):
    if not is_logged_in():
        return redirect(url_for('login'))

    user_id = session.get('user_id')
    product = Product.query.get_or_404(product_id)

    existing_item = CartItem.query.filter_by(user_id=user_id, product_id=product.id).first()

    if existing_item:
        existing_item.quantity += 1
    else:
        new_item = CartItem(user_id=user_id, product_id=product.id, quantity=1, price=product.price)
        db.session.add(new_item)

    db.session.commit()

    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    session['cart_count'] = len(cart_items)

    return redirect(url_for('home'))

@app.route('/update_quantity/<int:item_id>/<action>', methods=['POST'])
def update_quantity(item_id, action):
    if not is_logged_in():
        return redirect(url_for('login'))

    item = CartItem.query.get(item_id)
    if item:
        if action == 'increase':
            item.quantity += 1
        elif action == 'decrease' and item.quantity > 1:
            item.quantity -= 1
        db.session.commit()

    user_id = session.get('user_id')
    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    session['cart_count'] = len(cart_items)



@app.route('/remove_from_cart/<int:item_id>', methods=['POST'])
def remove_from_cart(item_id):
    if not is_logged_in():
        return redirect(url_for('login'))

    item = CartItem.query.get(item_id)
    if item:
        db.session.delete(item)
        db.session.commit()

    return redirect(url_for('view_cart'))

@app.route('/cart')
def view_cart():
    if not is_logged_in():
        return redirect(url_for('login'))

    user_id = session.get('user_id')

    if 'cart_count' not in session:
        session['cart_count'] = 0

    cart_items = CartItem.query.filter_by(user_id=user_id).all()
    total_amount = sum([item.price * item.quantity for item in cart_items])

    return render_template('cart.html', cart_items=cart_items, total_amount=total_amount)

@app.route("/api/purchase_history")
def get_purchase_history():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session['user_id']
    orders = UserPurchaseHistory.query.filter_by(user_id=user_id).order_by(UserPurchaseHistory.timestamp.desc()).all()

    history = [{
        "date": order.timestamp.strftime("%Y-%m-%d"),
        "product": order.product.name,
        "quantity": order.quantity,
        "total": round(order.total_amount_spent, 2),
    } for order in orders]

    return jsonify(history)

#Recommender system setup & train
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_product_embeddings():
    with app.app_context():
        products = Product.query.all()
        #based on description section from my product table
        descriptions = [product.description for product in products]
        embeddings = model.encode(descriptions)
    return products, embeddings

products_cache, embeddings_cache = get_product_embeddings()

def recommend_products(user_id, k=3):
    last_purchase = (
        UserPurchaseHistory.query.filter_by(user_id=user_id)
        .order_by(UserPurchaseHistory.timestamp.desc())
        .first()
    )

    if not last_purchase:
        return products_cache[:k]

    last_product = Product.query.get(last_purchase.product_id)

    product_indices = {product.id: idx for idx, product in enumerate(products_cache)}
    last_idx = product_indices.get(last_product.id)

    if last_idx is None:
        return products_cache[:k]

    similarity_matrix = cosine_similarity([embeddings_cache[last_idx]], embeddings_cache)[0]
    similarity_matrix[last_idx] = -1

    top_indices = np.argsort(similarity_matrix)[-k:][::-1]
    recommended_products = [products_cache[idx] for idx in top_indices]

    return recommended_products

@app.route("/api/recommendations")
def get_recommendations():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session['user_id']
    recommendations = recommend_products(user_id)

    data = [{
        "id": product.id,
        "name": product.name,
        "price": round(product.price, 2)
    } for product in recommendations]

    return jsonify(data)


@app.route('/checkout', methods=['POST'])
def checkout():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    cart_items = CartItem.query.filter_by(user_id=user_id).all()

    if not cart_items:
        flash("Your cart is empty!", "warning")
        return redirect(url_for('view_cart'))

    total_amount = sum([item.price * item.quantity for item in cart_items])

    for item in cart_items:
        purchase = UserPurchaseHistory(
            user_id=user_id,
            product_id=item.product_id,
            quantity=item.quantity,
            total_amount_spent=item.price * item.quantity
        )
        db.session.add(purchase)

        product_ingredients = ProductIngredient.query.filter_by(product_id=item.product_id).all()
        for pi in product_ingredients:
            ingredient = Ingredient.query.get(pi.ingredient_id)
            if ingredient.total_stock >= pi.quantity_needed * item.quantity:
                ingredient.total_stock -= pi.quantity_needed * item.quantity
            else:
                flash(f"Warning: Low stock for {ingredient.name}!", "danger")

    CartItem.query.filter_by(user_id=user_id).delete()
    db.session.commit()

    flash("Checkout successful!", "success")
    return redirect(url_for('confirmation'))


@app.route('/api/sales_data')
def sales_data():
    if 'user_id' not in session or session.get('role') != 'manager':
        return jsonify({'error': 'Unauthorized'}), 403

    sales = (
        db.session.query(
            Product.name,
            db.func.sum(UserPurchaseHistory.quantity).label('total_sales'),
            db.func.sum(UserPurchaseHistory.total_amount_spent).label('total_revenue')
        )
        .join(UserPurchaseHistory)
        .group_by(Product.id)
        .all()
    )

    return jsonify([{'product': p[0], 'sales': p[1], 'revenue': p[2]} for p in sales])

@app.route('/api/revenue_forecast')
def revenue_forecast():
    if 'user_id' not in session or session.get('role') != 'manager':
        return jsonify({'error': 'Unauthorized'}), 403

    # Get revenue in the past 30 days
    last_month = datetime.utcnow() - timedelta(days=30)
    total_revenue = db.session.query(db.func.sum(UserPurchaseHistory.total_amount_spent)).filter(UserPurchaseHistory.timestamp >= last_month).scalar() or 0

    # Simple forecast: Assume next month earns the same as last 30 days
    forecasted_revenue = total_revenue * 1.1  # Adding 10% growth assumption

    return jsonify({'last_30_days_revenue': total_revenue, 'forecasted_next_month': forecasted_revenue})

@app.route('/manager_dashboard')
def manager_dashboard():
    if 'user_id' not in session or session.get('role') != 'manager':
        return redirect(url_for('login'))

    transactions = UserPurchaseHistory.query.order_by(UserPurchaseHistory.timestamp.desc()).all()
    return render_template('manager_dashboard.html', transactions=transactions)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    purchase_history = (
        UserPurchaseHistory.query.filter_by(user_id=user.id)
        .order_by(UserPurchaseHistory.timestamp.desc())
        .all()
    )

    total_spent = sum(p.total_amount_spent for p in purchase_history)  # Calculate total spend

    return render_template(
        'profile.html',
        user=user,
        purchase_history=purchase_history,
        total_spent=total_spent
    )

import logging

logging.basicConfig(level=logging.DEBUG)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Log the incoming form data
        logging.debug("Received form data: %s", request.form)

        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']
        passkey = request.form.get('passkey', None)  # For manager role, this will be provided

        # Check if username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            logging.debug("Username already exists: %s", username)
            return jsonify({'success': False, 'message': "Username already exists!"})

        # If the role is 'manager', check for the passkey
        if role == 'manager' and passkey != 'xyz':  # Replace 'xyz' with actual passkey
            logging.debug("Invalid passkey for manager role")
            return jsonify({'success': False, 'message': "Invalid passkey for manager account!"})

        # If passwords do not match
        if password != confirm_password:
            logging.debug("Passwords do not match")
            return jsonify({'success': False, 'message': "Passwords do not match"})

        # Create new user and hash the password before storing it
        new_user = User(username=username, email=email, role=role)
        new_user.set_password(password)  # Hash password before storing

        try:
            db.session.add(new_user)
            db.session.commit()
            logging.debug("User created successfully: %s", username)
            return jsonify({'success': True, 'message': "Registration successful! Please log in."})
        except Exception as e:
            logging.error("Error during registration: %s", str(e))
            return jsonify({'success': False, 'message': f"An error occurred: {str(e)}. Please try again."})

    return render_template('register.html')  # Show the registration form if GET request



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session['cart_count'] = 0  # Initialize cart count

            # Get existing cart items
            cart_items = CartItem.query.filter_by(user_id=user.id).all()
            session['cart_count'] = len(cart_items)  # Update count

            flash("Login successful!", 'success')
            return redirect(url_for('dashboard' if user.role == 'manager' else 'home'))
        else:
            flash("Invalid username or password!", 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", 'info')
    return redirect(url_for('login'))



@app.route('/confirmation')
def confirmation():
    return render_template('checkout_confirmation.html')


@app.route('/api/daily_revenue')
def daily_revenue():
    if 'user_id' not in session or session.get('role') != 'manager':
        return jsonify({'error': 'Unauthorized'}), 403

    from sqlalchemy import func

    last_30_days = datetime.utcnow() - timedelta(days=30)

    #real revenue data from database
    revenue_data = (
        db.session.query(
            func.strftime('%Y-%m-%d', UserPurchaseHistory.timestamp).label('date'),
            func.sum(UserPurchaseHistory.total_amount_spent).label('revenue')
        )
        .filter(UserPurchaseHistory.timestamp >= last_30_days)
        .group_by('date')
        .order_by('date')
        .all()
    )

    revenue_dict = {d[0]: d[1] for d in revenue_data}

    # synthetic data for missing dates
    dates = [(datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
    synthetic_data = []

    for date in reversed(dates):
        revenue = revenue_dict.get(date, random.randint(45, 60))  # sample revenue data for functionality
        synthetic_data.append({'date': date, 'revenue': revenue})

    return jsonify(synthetic_data)

@app.route('/api/transaction_data')
def transaction_data():
    if 'user_id' not in session or session.get('role') != 'manager':
        return jsonify({'error': 'Unauthorized'}), 403

    transactions = UserPurchaseHistory.query.order_by(UserPurchaseHistory.timestamp.desc()).all()

    transaction_list = [
        {
            'date': t.timestamp.strftime('%Y-%m-%d'),
            'product': t.product.name,
            'quantity': t.quantity,
            'amount': t.total_amount_spent
        } for t in transactions
    ]

    return jsonify(transaction_list)



def linear(x, a, b):
    return a * x + b

def exponential(x, a, b):
    return a * np.exp(b * x)

def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def logarithmic(x, a, b):
    return a * np.log(x + 1) + b  #log(0) is undefined so needed to use log(x+1)

def find_best_model(dates, revenue):
    x_data = np.array(range(len(dates)))
    y_data = np.array(revenue)

    models = {
        "Linear": linear,
        "Exponential": exponential,
        "Polynomial": polynomial,
        "Logarithmic": logarithmic
    }

    results = {}
    best_model = None
    best_fit = None
    best_error = float("inf")

    for name, func in models.items():
        try:
            params, _ = curve_fit(func, x_data, y_data, maxfev=2000)
            predicted = func(x_data, *params)
            error = np.mean((y_data - predicted) ** 2)

            results[name] = list(predicted)

            if error < best_error:
                best_error = error
                best_model = name
                best_fit = list(predicted)
        except:
            continue

    return best_model, best_fit, results

@app.route('/api/ai_revenue_model')
def ai_revenue_model():
    if 'user_id' not in session or session.get('role') != 'manager':
        return jsonify({'error': 'Unauthorized'}), 403

    from sqlalchemy import func

    last_30_days = datetime.utcnow() - timedelta(days=30)

    revenue_data = (
        db.session.query(
            func.strftime('%Y-%m-%d', UserPurchaseHistory.timestamp).label('date'),
            func.sum(UserPurchaseHistory.total_amount_spent).label('revenue')
        )
        .filter(UserPurchaseHistory.timestamp >= last_30_days)
        .group_by('date')
        .order_by('date')
        .all()
    )

    revenue_dict = {d[0]: d[1] for d in revenue_data}

    dates = [(datetime.utcnow() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
    revenue_values = []

    for date in reversed(dates):
        if date == "2024-02-13":
            revenue = revenue_dict.get(date, 275)
        else:
            revenue = revenue_dict.get(date, 280 + random.uniform(-10, 10))
        revenue_values.append(revenue)

    best_model, best_fit, all_models = find_best_model(dates, revenue_values)

    return jsonify({
        'dates': dates,
        'real_revenue': revenue_values,
        'models': all_models,
        'best_model': best_model,
        'best_fit': best_fit
    })



@app.route('/api/sales_overview')
def sales_overview():
    if 'user_id' not in session or session.get('role') != 'manager':
        return jsonify({'error': 'Unauthorized'}), 403

    from sqlalchemy import func

    last_30_days = datetime.utcnow() - timedelta(days=30)

    total_sales = db.session.query(func.sum(UserPurchaseHistory.total_amount_spent)).scalar() or 0
    total_orders = db.session.query(func.count(UserPurchaseHistory.id)).scalar() or 0
    last_30_days_revenue = (
        db.session.query(func.sum(UserPurchaseHistory.total_amount_spent))
        .filter(UserPurchaseHistory.timestamp >= last_30_days)
        .scalar() or 0
    )
    forecast_next_month = last_30_days_revenue * 1.1  # Assume 10% growth

    return jsonify({
        "total_sales": total_sales,
        "total_orders": total_orders,
        "last_30_days": last_30_days_revenue,
        "forecast_next_month": forecast_next_month
    })
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/stock_management')
def stock_management():
    return render_template('stock_management.html')

@app.route('/orders')
def orders():
    return render_template('orders.html')

@app.route('/customer_insights')
def customer_insights():
    return render_template('customer_insights.html')  # Placeholder, will be added

from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta

# ingredient sales history from client
ingredient_sales_history = {
    "Chilli Powder": [5, 7, 6, 8, 10, 9, 11],
    "Garlic": [2, 3, 2.5, 2.8, 3.2, 2.9, 3.5],
    "Vinegar": [1, 1.5, 1.2, 1.8, 2, 1.7, 2.1],
    "Mustard Seeds": [0.8, 1, 0.9, 1.2, 1.3, 1.1, 1.4],
    "Oil": [10, 9.5, 9, 8.5, 8, 7.8, 7.5],
    "Salt": [3, 2.8, 3.2, 2.7, 2.9, 3, 2.6]
}

ingredient_stock = {
    "Chilli Powder": 50,
    "Garlic": 30,
    "Vinegar": 80,
    "Mustard Seeds": 40,
    "Oil": 200,
    "Salt": 100
}

def predict_days_until_empty(ingredient):
    try:
        if ingredient not in ingredient_sales_history:
            return "Sales history not found for ingredient."

        sales_data = ingredient_sales_history[ingredient]
        print(f"Sales data for {ingredient}: {sales_data}")

        if len(sales_data) < 2:
            return "Not enough data for prediction."

        #linear regression
        X = np.array(range(len(sales_data))).reshape(-1, 1)
        y = np.array(sales_data)

        model = LinearRegression()
        model.fit(X, y)
        print(f"Model Coefficients for {ingredient}: {model.coef_}, Intercept: {model.intercept_}")
        next_day_sales = model.predict([[len(sales_data)]])[0]
        print(f"Predicted sales for next day for {ingredient}: {next_day_sales}")  # Log prediction

        if next_day_sales <= 0:
            return "Invalid prediction: predicted sales are zero or negative."
        stock = ingredient_stock.get(ingredient, 0)
        if stock == 0:
            return f"No stock available for {ingredient}."

        days_remaining = stock / next_day_sales

        if days_remaining <= 0:
            return "Error: Invalid prediction, days remaining is non-positive."

        restock_date = datetime.today() + timedelta(days=int(days_remaining))

        return round(days_remaining, 2), restock_date.strftime("%Y-%m-%d")

    except Exception as e:
        return f"Error: {str(e)}"


@app.route('/stock_forecast')
def stock_forecast_page():
    ingredient_data = []
    for ingredient in ingredient_stock:
        ingredient_data.append({
            "name": ingredient,
            "stock": ingredient_stock[ingredient]  # Sending current stock level
        })

    return render_template('stock_forecast.html', ingredient_data=ingredient_data)

@app.route('/check_ingredients')
def check_ingredients():
    ingredients = Ingredient.query.all()
    for ingredient in ingredients:
        print(f"Ingredient: {ingredient.name}, Stock: {ingredient.total_stock}")
    return "Check the console for ingredients data."


@app.route('/api/top_selling')
def top_selling():
    top_products = [
        {"name": "Cucumber Pickles", "units_sold": 150},
        {"name": "Mango Pickles", "units_sold": 120},
        {"name": "Garlic Pickles", "units_sold": 90},
    ]
    return jsonify(top_products)

@app.route('/api/stock_levels')
def stock_levels():
    ingredients = Ingredient.query.all()
    stock_data = [{"name": ing.name, "stock": ing.total_stock} for ing in ingredients]
    return jsonify(stock_data)

@app.route('/api/pickle_ingredients')
def pickle_ingredients():
    pickles = Product.query.all()
    pickle_data = []

    for pickle in pickles:
        ingredients = ProductIngredient.query.filter_by(product_id=pickle.id).all()
        ingredient_list = [
            {"name": ing.ingredient.name, "quantity_needed": ing.quantity_needed}
            for ing in ingredients
        ]
        pickle_data.append({"pickle_name": pickle.name, "ingredients": ingredient_list})

    return jsonify(pickle_data)


@app.route('/customer_insights')
def customer_insights_page():
    return render_template('customer_insights.html')

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    try:
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(200, 10, txt="Comprehensive Manager Report", ln=True, align="C")
        pdf.set_font('Arial', '', 12)
        pdf.cell(200, 10, txt="Sales & Forecast Data:", ln=True)
        sales_data = get_sales_data()  # Method to fetch sales and forecast data
        for data in sales_data:
            pdf.cell(200, 10, txt=f"{data['product']} - Sales: {data['sales']} - Revenue: {data['revenue']} - Forecast: {data['forecast']}", ln=True)

        pdf.cell(200, 10, txt="Stock Management Data:", ln=True)
        stock_data = get_stock_data()  # Method to fetch stock data
        for ingredient in stock_data:
            pdf.cell(200, 10, txt=f"{ingredient['name']} - Stock: {ingredient['stock']}"
                                  f" - Low Stock: {ingredient['low_stock']}", ln=True)

        pdf.cell(200, 10, txt="Customer Insights:", ln=True)
        customer_insights = get_customer_insights()  # Method to fetch customer data
        pdf.cell(200, 10, txt=f"Total Customers: {customer_insights['total_customers']}", ln=True)
        pdf.cell(200, 10, txt=f"Total Engagement: {customer_insights['total_engagement']}", ln=True)

        chart_image = generate_sales_chart()
        pdf.image(chart_image, x=10, y=pdf.get_y(), w=190)

        output_path = 'static/reports/manager_report.pdf'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pdf.output(output_path)

        return jsonify({"success": True, "file_path": output_path})
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return jsonify({"success": False, "error": str(e)})



def get_sales_data():
    return [
        {"product": "Cucumber Pickles", "sales": 150, "revenue": 750, "forecast": 800},
        {"product": "Mango Pickles", "sales": 120, "revenue": 600, "forecast": 650},
        {"product": "Garlic Pickles", "sales": 90, "revenue": 450, "forecast": 500},
    ]

def get_stock_data():
    return [
        {"name": "Chilli Powder", "stock": 50, "low_stock": "No"},
        {"name": "Garlic", "stock": 30, "low_stock": "Yes"},
        {"name": "Vinegar", "stock": 80, "low_stock": "No"},
    ]

def get_customer_insights():
    return {
        "total_customers": 1000,
        "total_engagement": 800
    }

def generate_sales_chart():
    products = ["Cucumber Pickles", "Mango Pickles", "Garlic Pickles"]
    sales = [150, 120, 90]

    fig, ax = plt.subplots()
    ax.bar(products, sales)
    ax.set_title('Sales Data')
    ax.set_xlabel('Product')
    ax.set_ylabel('Sales')
    chart_path = 'static/reports/sales_chart.png'
    fig.savefig(chart_path)

    return chart_path

@app.route('/generate_report')
def generate_report_page():
    return render_template('generate_report.html')


ingredient_stock = {
    "Chilli Powder": 10,  # kg
    "Garlic": 30,
    "Vinegar": 80,  # liters
    "Mustard Seeds": 40,
    "Oil": 200,
    "Salt": 100
}

def predict_days_until_empty(ingredient):
    X = np.array(range(len(ingredient_sales_history[ingredient]))).reshape(-1, 1)
    y = np.array(ingredient_sales_history[ingredient])

    model = LinearRegression()
    model.fit(X, y)

    daily_usage_prediction = model.predict([[len(ingredient_sales_history[ingredient])]])[0]
    days_remaining = ingredient_stock[ingredient] / daily_usage_prediction if daily_usage_prediction > 0 else float("inf")

    restock_date = datetime.today() + timedelta(days=int(days_remaining))

    return round(days_remaining, 2), restock_date.strftime("%Y-%m-%d")

@app.route('/api/stock_forecast')
def stock_forecast():
    ingredient_stock = {ing.name: ing.total_stock for ing in Ingredient.query.all()}
    predictions = {ing: round(ingredient_stock[ing] / 5, 2) for ing in ingredient_stock}
    return jsonify(predictions)

@app.route('/api/restock', methods=['POST'])
def restock_ingredient():
    data = request.json
    ingredient_name = data.get("ingredient")
    amount = data.get("amount", 0)  # Default increase by 10 units

    ingredient = Ingredient.query.filter_by(name=ingredient_name).first()

    if ingredient:
        ingredient.total_stock += amount
        db.session.commit()  # Save the updated stock to the database
        return jsonify({"message": f"Successfully added {amount} units of {ingredient_name}!"})

    return jsonify({"error": "Ingredient not found"}), 400


LOW_STOCK_THRESHOLD = 20

@app.route('/api/order_more', methods=['POST'])
def order_more():
    data = request.json
    ingredient_name = data.get("ingredient")
    quantity = data.get("quantity", 10)

    if ingredient_name in ingredient_stock:
        ingredient_stock[ingredient_name] += quantity
        return jsonify({"message": f"Ordered {quantity} more units of {ingredient_name}."})

    return jsonify({"error": "Ingredient not found"}), 400

@app.route('/stock_management')
def stock_management_page():
    return render_template('stock_management.html')

WAREHOUSE_COORDINATES = (16.5305291, 81.4314064)  # Coordinates of J.P road, Bhimavaram, Andhra Pradesh, India

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    product_id = db.Column(db.Integer, db.ForeignKey('product.id'), nullable=False)
    quantity = db.Column(db.Integer, default=1)
    total_price = db.Column(db.Float, nullable=False)
    delivery_address = db.Column(db.String(255), nullable=False)
    estimated_delivery_time = db.Column(db.String(100))  # Estimated delivery time
    status = db.Column(db.String(50), default="Not Received")  # Order status

    user = db.relationship('User', backref='orders', lazy=True)
    product = db.relationship('Product', backref='orders', lazy=True)

    def __repr__(self):
        return f"<Order {self.id}, User {self.user_id}, Product {self.product_id}>"

def calculate_delivery_time(user_address_coordinates):
    lat1, lon1 = WAREHOUSE_COORDINATES
    lat2, lon2 = user_address_coordinates

    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    speed = 50 #sample based on average from google search
    delivery_time_in_hours = distance / speed

    hours = int(delivery_time_in_hours)
    minutes = int((delivery_time_in_hours - hours) * 60)

    return f"{hours} hours {minutes} minutes"

@app.route('/place_order', methods=['POST'])
def place_order():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session['user_id']
    product_id = request.form['product_id']
    quantity = int(request.form['quantity'])
    total_price = float(request.form['total_price'])

    delivery_address = request.form['delivery_address']
    delivery_coordinates = tuple(map(float, request.form['delivery_coordinates'].split(',')))  # lat,lng in form "lat,lng"
    new_order = Order(user_id=user_id, product_id=product_id, quantity=quantity, total_price=total_price, delivery_address=delivery_address)

    estimated_delivery_time = calculate_delivery_time(delivery_coordinates)
    new_order.estimated_delivery_time = estimated_delivery_time

    db.session.add(new_order)
    db.session.commit()

    return jsonify({"message": "Order placed successfully!", "estimated_delivery_time": estimated_delivery_time})

@app.route('/api/orders')
def get_orders():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401

    user_id = session['user_id']
    orders = Order.query.filter_by(user_id=user_id).all()

    orders_data = []
    for order in orders:
        orders_data.append({
            "id": order.id,
            "product": {
                "name": order.product.name
            },
            "quantity": order.quantity,
            "total_price": order.total_price,
            "delivery_address": order.delivery_address,
            "estimated_delivery_time": order.estimated_delivery_time,
            "status": order.status
        })

    return jsonify(orders_data)

@app.route('/orders')
def orders_page():
    return render_template('orders.html')

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
@app.route("/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.json
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"response": "Ava: Please enter a valid message."})
        GPT_MODEL = "gpt-3.5-turbo-1106"
        messages = [
                {"role": "system", "content": 'You answer question about Pickles.'
                },
                {"role": "user", "content": 'the user message'},
            ]
        response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=messages,
                temperature=0
            )
        bot_response = response.choices[0].message.content.strip()
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"response": f"Error - {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
