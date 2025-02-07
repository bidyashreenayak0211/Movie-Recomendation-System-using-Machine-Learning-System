# Movie Recommendation System 

## Overview
This project implements a **Movie Recommendation System** using **Python** and **Google Colab**. The system leverages **TF-IDF vectorization** and **cosine similarity** for content-based recommendations and includes **interactive visualizations** for data exploration.

---

## Steps and Code Explanation

### 1. **Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ipywidgets as widgets
from IPython.display import display
```
**Explanation:**
- `pandas` and `numpy`: Handle data manipulation.
- `matplotlib`, `seaborn`, and `plotly.express`: Create visualizations.
- `wordcloud`: Generate a word cloud from movie overviews.
- `sklearn.feature_extraction.text`: Implement TF-IDF for text analysis.
- `ipywidgets`: Create interactive elements.
- `IPython.display`: Display UI elements in Colab.

---

### 2. **Loading Dataset**
```python
df = pd.read_csv("/mnt/data/dataset.csv")
```
**Explanation:**
- Reads the movie dataset into a Pandas DataFrame.

---

### 3. **Data Cleaning**
```python
df.dropna(subset=['genre', 'overview'], inplace=True)
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df['genre'] = df['genre'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
```
**Explanation:**
- Drops rows where **genre** or **overview** is missing.
- Extracts **release year** from the date.
- Converts **genre** from a string into a list.

---

### 4. **Building the Recommendation Engine**
```python
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['overview'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```
**Explanation:**
- Uses **TF-IDF vectorization** to transform movie overviews into numerical form.
- Computes **cosine similarity** to measure similarity between movies.

#### **Function to Recommend Movies**
```python
def recommend_movies(title, num_recommendations=5):
    idx = df[df['title'].str.lower() == title.lower()].index
    if len(idx) == 0:
        return "Movie not found. Try another title."
    idx = idx[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df.iloc[movie_indices][['title', 'genre', 'release_year']]
```
**Explanation:**
- Searches for the movie title.
- Retrieves the most similar movies based on cosine similarity.
- Returns a list of recommended movies.

---

### 5. **Interactive Movie Selection**
```python
def on_movie_selection(change):
    movie_title = change['new']
    display(recommend_movies(movie_title))

movie_dropdown = widgets.Dropdown(
    options=df['title'].unique(),
    description='Movie:',
    continuous_update=False
)
movie_dropdown.observe(on_movie_selection, names='value')
display(movie_dropdown)
```
**Explanation:**
- Creates an **interactive dropdown menu** for selecting a movie.
- Calls the recommendation function when a movie is chosen.

---

### 6. **Data Visualization**
#### **Genre Distribution**
```python
genre_counts = df.explode('genre')['genre'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Count']
fig = px.bar(genre_counts, x='Genre', y='Count', title='Movie Genre Distribution', width=1000, height=500)
fig.show()
```
**Explanation:**
- Expands the **genre** column (since movies have multiple genres).
- Counts the occurrences of each genre.
- Displays an **interactive bar chart**.

#### **Word Cloud of Overviews**
```python
text = " ".join(df['overview'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
```
**Explanation:**
- Combines all movie overviews into a single text block.
- Generates a **word cloud** to highlight frequently used words.

---

## **Conclusion**
This project provides a **content-based movie recommendation system** with interactive visualizations and user interaction features. The system allows users to explore **movie genres**, **word frequencies**, and dynamically **receive movie recommendations**. 🚀

---

## **Future Enhancements**
- Implement **collaborative filtering** for better recommendations.
- Add **sentiment analysis** on movie overviews.
- Integrate **user reviews and ratings** for better insights.

Let me know if you need further modifications! 🎬

