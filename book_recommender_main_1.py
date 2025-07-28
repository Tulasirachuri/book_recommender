import pandas as pd # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.neighbors import NearestNeighbors # type: ignore
import gradio as gr # type: ignore

# 📥 Load Data
books = pd.read_csv('Books.csv', encoding='latin-1', low_memory=False)
ratings = pd.read_csv('Ratings.csv', encoding='latin-1', low_memory=False)

# 📚 Clean & Prepare
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Publisher', 'Year-Of-Publication', 'Image-URL-L']]
books.columns = ['isbn', 'title', 'author', 'publisher', 'year', 'image']
ratings.columns = ['user_id', 'isbn', 'rating']
ratings = ratings[ratings['rating'] > 0]

# 🔗 Merge & Aggregate
book_ratings = pd.merge(ratings, books, on='isbn')
avg_rating_df = book_ratings.groupby('title').agg({'rating': ['mean', 'count']}).reset_index()
avg_rating_df.columns = ['title', 'avg_rating', 'rating_count']
books_final = pd.merge(books, avg_rating_df, on='title').drop_duplicates('title').reset_index(drop=True)
books_final = books_final[books_final['rating_count'] > 50].reset_index(drop=True)

# 🧠 TF-IDF & Nearest Neighbors
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books_final['title'])
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)
indices = pd.Series(books_final.index, index=books_final['title'].str.lower()).drop_duplicates()
book_titles = books_final['title'].tolist()

# 🔍 Recommendation Logic
def recommend(book_title):
    book_title = book_title.lower().strip()
    matched_titles = [t for t in indices.index if book_title in t]

    if not matched_titles:
        suggestions = [title for title in book_titles if book_title in title.lower()]
        if suggestions:
            return f"No exact match found. Suggestions: {', '.join(suggestions[:5])}", None
        return f"No matches found for '{book_title}'", None

    idx = indices[matched_titles[0]]
    distances, idxs = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=15)
    book_indices = idxs.flatten()
    results = books_final.iloc[book_indices].sort_values(by='avg_rating', ascending=False)

    # 🖼️ Format as grid HTML
    grid_html = '<div class="grid">'
    for _, row in results.iterrows():
        grid_html += f"""
        <div class=\"card\" onclick=\"window.open('{row['image']}', '_blank')\">
            <img src="{row['image']}" alt="{row['title']}">
            <div class="card-content">
                <h4>{row['title']}</h4>
                <p><strong>Author:</strong> {row['author']}</p>
                <p><strong>Publisher:</strong> {row['publisher']} ({row['year']})</p>
                <p><strong>Rating:</strong> ⭐ {row['avg_rating']:.2f} ({row['rating_count']} ratings)</p>
                <p><em>Description not available.</em></p>
            </div>
        </div>
        """
    grid_html += '</div>'
    return f"{len(results)} recommendations found for '{matched_titles[0].title()}'", grid_html

# 🌟 Interface with Styling & Banner Image
with gr.Blocks(css="""
body {
    font-family: 'Segoe UI', sans-serif;
    background: #fefefe;
    animation: fadeIn 0.8s ease;
}
img.banner {
    width: 100%;
    max-height: 300px;
    object-fit: cover;
    border-radius: 16px;
    margin-bottom: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
}
h1.title {
    text-align: center;
    font-size: 2.5em;
    font-weight: bold;
    color: #ff6600;
    margin: 10px 0;
}
input[type="text"] {
    padding: 12px;
    font-size: 16px;
    border-radius: 12px;
    border: 2px solid #ff8800;
    width: 100%;
    box-shadow: 0px 2px 8px rgba(255,136,0,0.2);
    transition: all 0.3s ease;
}
input[type="text"]:hover {
    border-color: #ff6600;
    box-shadow: 0px 4px 10px rgba(255,102,0,0.3);
}
.grid {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    justify-content: center;
    padding-top: 10px;
}
.card {
    width: 220px;
    background: white;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    cursor: pointer;
}
.card:hover {
    transform: scale(1.08);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.25);
}
.card img {
    width: 100%;
    height: 250px;
    object-fit: cover;
}
.card-content {
    padding: 10px;
    text-align: left;
    font-size: 0.9em;
    color:black;
}
@keyframes fadeIn {
    from { opacity: 0;pip transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}
""") as demo:

    gr.HTML('<img class="banner" src="https://images.unsplash.com/photo-1521587760476-6c12a4b040da?auto=format&fit=crop&w=1350&q=80" alt="Books Banner">')
    gr.HTML('<h1 class="title">Book Recommender</h1>')

    book_input = gr.Textbox(label="Search Book Title", placeholder="e.g. Harry Potter", lines=1)
    result_count = gr.Textbox(label="Results Count", interactive=False)
    results_output = gr.HTML(label="Recommendations")

    gr.Markdown("🔍 Start typing a book title. Click on a book to enlarge its cover.")

    book_input.change(fn=recommend, inputs=book_input, outputs=[result_count, results_output])
    book_input.submit(fn=recommend, inputs=book_input, outputs=[result_count, results_output])


demo.launch()