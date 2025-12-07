from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load Keras model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

# Step‑wise feature grouping
feature_categories = {
    1: ['mean radius', 'radius error', 'worst radius'],
    2: ['mean perimeter', 'perimeter error', 'worst perimeter'],
    3: ['mean area', 'area error', 'worst area'],
    4: ['mean texture', 'texture error', 'worst texture'],
    5: ['mean symmetry', 'symmetry error', 'worst symmetry',
        'mean fractal dimension', 'fractal dimension error', 'worst fractal dimension'],
    6: ['mean smoothness', 'smoothness error', 'worst smoothness',
        'mean compactness', 'compactness error', 'worst compactness'],
    7: ['mean concavity', 'concavity error', 'worst concavity',
        'mean concave points', 'concave points error', 'worst concave points']
}

# Flatten list of all features
FEATURE_NAMES = [feat for feats in feature_categories.values() for feat in feats]

def save_step_data(step):
    for feature in feature_categories[step]:
        key = feature.replace(' ', '_')
        session[key] = request.form.get(key)

@app.route('/')
def home():
    session.clear()
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/docs')
def docs():
    return render_template('docs.html')

@app.route('/step/<int:step>', methods=['GET', 'POST'])
def step_handler(step):
    total_steps = len(feature_categories) + 1
    if step < 1 or step > total_steps:
        return redirect(url_for('home'))

    # Data entry steps
    if step <= len(feature_categories):
        if request.method == 'POST':
            save_step_data(step)
            return redirect(url_for('step_handler', step=step+1))

        templates = {
            1: 'step1_radius.html',
            2: 'step2_perimeter.html',
            3: 'step3_area.html',
            4: 'step4_texture.html',
            5: 'step5_symmetry.html',
            6: 'step6_smoothness_compactness.html',
            7: 'step7_concavity.html'
        }
        return render_template(
            templates[step],
            features=feature_categories[step],
            current_step=step,
            total_steps=total_steps
        )

    # Step 8: Prediction & summary
    try:
        values = [
            float(session.get(feat.replace(' ', '_'), 0))
            for feat in FEATURE_NAMES
        ]
    except ValueError:
        return "Invalid input detected."

    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    prob = model.predict(scaled)[0][0]

    print("🔍 Raw model probability of Benign:", prob)

    # Interpret: prob = P(Benign)
    if prob >= 0.5:
        pred_class = 'Benign'
        confidence = prob
    else:
        pred_class = 'Malignant'
        confidence = 1 - prob

    # Decide bar color & warning
    if confidence >= 0.8:
        bar_color = '#22c55e'       # green
    elif confidence >= 0.5:
        bar_color = '#f59e0b'       # yellow
    else:
        bar_color = '#dc2626'       # red

    warning = ''
    if confidence < 0.7:
        warning = "⚠️ Low confidence—please consult a specialist."

    pairs = list(zip(FEATURE_NAMES, values))

    return render_template(
        'step8_summary.html',
        pairs=pairs,
        prediction=pred_class,
        probability=confidence,
        bar_color=bar_color,
        warning=warning,
        current_step=total_steps,
        total_steps=total_steps
    )

if __name__ == '__main__':
    app.run(debug=True)
