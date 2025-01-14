import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for compatibility
import matplotlib.pyplot as plt

app = Flask(__name__)

# Define the dataset paths for different categories
datasets = {
    'vlogs': 'dataset/CAvideos.csv',
    'travel': 'dataset/USvideos.csv',
    'food': 'dataset/GBvideos.csv',
    'entertainment': 'dataset/xAvideos.csv',
    'songs': 'dataset/CAvideos.csv'
}

# Define category titles
category_titles = {
    'vlogs': 'Vlogs: Explore Personal Stories',
    'travel': 'Travel: Discover New Places',
    'food': 'Food: Culinary Delights',
    'entertainment': 'Entertainment: Fun and Laughter',
    'songs': 'Songs: The Power of Music'
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
def analyze():
    category = request.args.get('category')  # Get the selected category from the query parameter

    if category in datasets:  # Check if the category exists
        dataset_path = datasets[category]
        
        # Load the dataset
        data = pd.read_csv(dataset_path)

        # Check if necessary columns exist
        if not all(col in data.columns for col in ['views', 'likes']):
            return "Data does not contain the required columns: 'views' and 'likes'.", 400

        # Prepare data for Linear Regression
        X = data[['views']].values  # Feature: views
        y = data['likes'].values     # Target: likes

        # Perform Linear Regression
        model = LinearRegression()
        model.fit(X, y)

        # Generate predictions
        predictions = model.predict(X)

        # Create a scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(data['views'], data['likes'], color='blue', label='Actual Likes', alpha=0.5)
        plt.plot(data['views'], predictions, color='red', label='Predicted Likes', linewidth=2)
        plt.xlabel('Views')
        plt.ylabel('Likes')
        plt.title(f'Likes vs Views for {category_titles[category]}')
        plt.legend()
        
        # Save the plot
        plot_filename = f'static/{category}_views_likes_plot.png'
        plt.savefig(plot_filename)
        plt.close()

        # Insights
        predicted_likes_for_100k_views = model.predict([[100000]])[0]
        insights = {
            'predicted_likes_for_100k_views': round(predicted_likes_for_100k_views, 2),
            'slope': round(model.coef_[0], 2),  # Access the first coefficient directly
            'intercept': round(model.intercept_, 2),  # Access intercept directly
            'total_videos': data.shape[0],
            'average_views': round(data['views'].mean(), 2),
            'average_likes': round(data['likes'].mean(), 2),
            
        }

        return render_template('results.html', plot_url=plot_filename, insights=insights, category_title=category_titles[category])
    else:
        return redirect(url_for('index'))  # Redirect to index if category is not found

if __name__ == '__main__':
    app.run(debug=True)
