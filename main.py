import re
import requests
import numpy as np
from bs4 import BeautifulSoup
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st
import io

# Settings
MAX_FEATURES = 10000  # Maximum vocabulary size used in training
MAXLEN = 500          # Maximum length for padding / truncating

# Define a user-agent header to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36"
}

def fetch_reviews(url):
    """
    Fetch the reviews page and extract individual review texts.
    """
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove unwanted tags
        for tag in soup(["script", "style"]):
            tag.decompose()
        
        # Extract review texts from review-container divs
        review_texts = []
        review_containers = soup.find_all("div", class_="review-container")
        for container in review_containers:
            text_div = container.find("div", class_="text")
            if text_div:
                review_text = text_div.get_text(separator=" ", strip=True)
                cleaned_text = re.sub(r"\s+", " ", review_text).strip()
                if len(cleaned_text) > 50:  # Ensure it’s substantial
                    review_texts.append(cleaned_text)
        
        return review_texts if review_texts else None
    except Exception as e:
        st.error(f"Error fetching reviews: {e}")
        return None

def search_movie_review_page(movie_name):
    """
    Search for the movie on IMDb and construct the reviews page URL.
    """
    query = movie_name.replace(" ", "+")
    search_url = f"https://www.imdb.com/find/?q={query}&s=tt"
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Get the first result link
    result = soup.find("a", class_="ipc-metadata-list-summary-item__t")
    if not result:
        raise Exception("Movie not found.")
    
    # Build full link and append "/reviews"
    movie_link = "https://www.imdb.com" + result["href"]
    reviews_url = movie_link.rstrip("/") + "/reviews"
    return reviews_url

def preprocess_text(text, word_index, maxlen=MAXLEN, max_features=MAX_FEATURES):
    """
    Preprocess text by tokenizing, encoding with the IMDb word index, and capping token indices.
    Incorporates a starting token.
    """
    words = text.lower().split()
    
    # Insert start token (index 1)
    encoded_review = [1]
    for word in words:
        # Look up the word; use 2 if not found (unknown token)
        index = word_index.get(word, 2) + 3  # Offset as in training
        # Cap the index if it exceeds the vocabulary size
        if index >= max_features:
            index = 2  # Unknown token index
        encoded_review.append(index)
    
    # Use post-padding to preserve the beginning
    padded_review = sequence.pad_sequences([encoded_review], maxlen=maxlen, padding='post', truncating='post')
    return padded_review

def load_model_from_upload(uploaded_file):
    """
    Load the model from an uploaded file in Streamlit.
    """
    try:
        # Save the uploaded file to a temporary location
        with open("temp_model.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        model = load_model("temp_model.h5")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Streamlit UI
    st.title("Movie Review Sentiment Analyzer")
    st.write("Enter a movie name to analyse the IMDb reviews.")

    model = load_model("simple_rnn_imdb.h5")
    
    # Movie name input
    movie_name = st.text_input("Enter a movie name (e.g., The Shawshank Redemption):", "")
    
    # Button to analyze
    if st.button("Analyze Reviews"):
        if not movie_name:
            st.error("Please enter a movie name.")
            return

        # Step 1: Get the reviews page URL
        with st.spinner("Fetching movie reviews..."):
            try:
                reviews_url = search_movie_review_page(movie_name)
                st.write(f"Fetched reviews page URL: {reviews_url}")
            except Exception as e:
                st.error(f"Error fetching movie: {e}")
                return

            # Step 2: Fetch individual reviews
            reviews = fetch_reviews(reviews_url)
            if not reviews:
                st.warning("No reviews found. Using mock reviews instead.")
                reviews = [
                    "This movie is a masterpiece, incredible performances by all.",
                    "The plot was slow, didn’t enjoy it as much as expected.",
                    "One of the best films ever, a powerful story of hope.",
                    "Felt overhyped, predictable and not that great.",
                    "Beautifully crafted, deeply moving film."
                ]
                st.write(f"Using {len(reviews)} mock reviews.")

        # Load the IMDb word index
        word_index = imdb.get_word_index()

        # Step 3: Process and predict sentiment for each review
        st.subheader(f"Top 5 Reviews for '{movie_name}':")
        sentiments = []
        scores = []
        for i, review in enumerate(reviews[:5], 1):  # Limit to top 5
            preprocessed_input = preprocess_text(review, word_index)
            preprocessed_input = preprocessed_input.astype('int32')
            try:
                prediction = model.predict(preprocessed_input, verbose=0)
                score = prediction[0][0]
                sentiment = "Positive" if score > 0.5 else "Negative"
                sentiments.append(sentiment)
                scores.append(score)
                st.write(f"**Review {i}:** {review[:100]}...")
                st.write(f"Sentiment: {sentiment}, Score: {score:.4f}")
                st.write("---")
            except Exception as e:
                st.error(f"Error predicting sentiment for review {i}: {e}")

        # Step 4: Calculate and display average sentiment
        if scores:
            avg_score = np.mean(scores)
            avg_sentiment = "Positive" if avg_score > 0.5 else "Negative"
            st.subheader("Summary")
            st.write(f"Average Sentiment for '{movie_name}' (based on top 5 reviews): **{avg_sentiment}**")
            st.write(f"Average Prediction Score: **{avg_score:.4f}**")
        else:
            st.error("No valid reviews to analyze.")

if __name__ == "__main__":
    main()