from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the trained models
model_path = 'D:\\mini\\srm\\models\\Gd_all_models.pkl'
if os.path.exists(model_path):
    models = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load column names from training data
train_df = pd.read_excel('D:\\mini\\srm\\data\\Station_data_NL_1987_2023.xlsx')
feature_columns = train_df.drop(["pr", "tas", "tmax", "tmin"], axis=1).columns.tolist()

# Ensure the 'static' directory exists
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        year = int(request.form['year'])
        doy = int(request.form['doy'])

        # Prepare input DataFrame
        input_data = pd.DataFrame([[year, doy]], columns=['YEAR', 'DOY'])

        # Ensure the order of features is consistent
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        # Make predictions using the loaded models
        results = {}
        scatter_plots = {}
        for target in ['pr', 'tas', 'tmax', 'tmin']:
            model = models[target]
            prediction = model.predict(input_data)
            results[target] = prediction[0]

            # Create scatter plot
            df = pd.read_excel('D:\\mini\\srm\\data\\Station_data_NL_1987_2023.xlsx')
            df = df.dropna()
            data_X = df[feature_columns]
            target_values = df[target]
            predictions = model.predict(data_X)

            plt.figure(figsize=(8, 6))
            plt.scatter(target_values, predictions, color='blue', marker='o')
            plt.xlabel('Observed')
            plt.ylabel('Predicted')
            plt.title(f'Scatter Plot for {target}')
            plt.plot([min(target_values), max(target_values)], [min(target_values), max(target_values)], linestyle='--', color='gray', label='1:1 line')
            plt.legend()

            scatter_plot_path = os.path.join(static_dir, f'{target}_scatter_plot.png')
            plt.savefig(scatter_plot_path)
            plt.close()
            
            scatter_plots[target] = f'/static/{target}_scatter_plot.png'

        return render_template('output.html', results=results, scatter_plots=scatter_plots)

    return render_template('user.html')

@app.route('/static/<filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
