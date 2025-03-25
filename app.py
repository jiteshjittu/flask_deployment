from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load and prepare data
def load_and_prepare_data():
    df = pd.read_csv('Amazon Beauty Recommendation system.csv')
    
    # Reduce dataset size by filtering popular products and active users
    popular_products = df['ProductId'].value_counts().nlargest(5000).index
    active_users = df['UserId'].value_counts().nlargest(5000).index
    
    df_filtered = df[df['ProductId'].isin(popular_products) & df['UserId'].isin(active_users)]
    
    # Create pivot table
    pivot_table = df_filtered.pivot_table(
        index='UserId', 
        columns='ProductId', 
        values='Rating', 
        aggfunc='mean', 
        fill_value=0
    )
    
    # Compute item similarity matrix
    item_similarity = cosine_similarity(pivot_table.T)
    item_similarity_df = pd.DataFrame(
        item_similarity, 
        index=pivot_table.columns, 
        columns=pivot_table.columns
    )
    
    return item_similarity_df, pivot_table, df_filtered

# Load data when starting the application
item_similarity_df, pivot_table, df_filtered = load_and_prepare_data()

def get_similar_items_with_ratings(product_id, top_n=5):
    if product_id not in item_similarity_df.index:
        return None
    
    similar_items = item_similarity_df[product_id].sort_values(ascending=False).iloc[1:top_n+1]
    similar_items_df = pd.DataFrame(similar_items)
    similar_items_df.columns = ['similarity']
    
    # Get average ratings for similar items
    ratings = df_filtered[df_filtered['ProductId'].isin(similar_items_df.index)].groupby('ProductId')['Rating'].mean()
    
    # Combine similarity scores with ratings
    result = []
    for idx in similar_items_df.index:
        result.append({
            'product_id': idx,
            'rating': round(ratings[idx], 2)
        })
    
    return result

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    product_id = request.form['product_id']
    similar_items = get_similar_items_with_ratings(product_id)
    
    if similar_items is None:
        return render_template('result.html', 
                             error="Product ID not found. Please try another product ID.",
                             product_id=product_id)
    
    return render_template('result.html', 
                         recommendations=similar_items,
                         product_id=product_id)

if __name__ == '__main__':
    app.run(debug=True) 