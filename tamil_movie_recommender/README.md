# Tamil Movie Recommendation System

A simple, beginner-friendly content-based movie recommendation system for Tamil movies.

## Project Overview

This system recommends similar Tamil movies based on movie metadata (genre, director, actors, description) using TF-IDF vectorization and cosine similarity.

## How It Works

```
Movie Metadata (genre, director, actors, description)
        ↓
Combined Text Features
        ↓
TF-IDF Vectorization (scikit-learn)
        ↓
Cosine Similarity Matrix
        ↓
Top 5 Similar Movies
```

## Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **scikit-learn** - TF-IDF vectorization and cosine similarity
- **Streamlit** - Web interface

## Project Structure

```
tamil_movie_recommender/
│
├── data/
│   └── tamil_movies.csv      # ~40 Tamil movies dataset
│
├── app.py                    # Streamlit web interface
├── recommender.py            # Core recommendation logic
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation (Windows)

```powershell
# 1. Navigate to project folder
cd tamil_movie_recommender

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

## How to Run

```powershell
# Make sure virtual environment is activated
venv\Scripts\activate

# Run the Streamlit app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Example Usage

1. Open the app in your browser
2. Select a Tamil movie from the dropdown (e.g., **Vikram**)
3. Click **"Recommend Movies"**
4. See top 5 similar movies with similarity scores

### Example Output

**Input:** Vikram

**Output:**
```
1. Kaithi (2019) - Action Thriller - Dir: Lokesh Kanagaraj - Similarity: 0.72
2. Master (2021) - Action Thriller - Dir: Lokesh Kanagaraj - Similarity: 0.68
3. Leo (2023) - Action Thriller - Dir: Lokesh Kanagaraj - Similarity: 0.65
4. Thani Oruvan (2015) - Action Thriller - Dir: Mohan Raja - Similarity: 0.58
5. Theeran Adhigaaram Ondru (2017) - Action Thriller - Dir: H. Vinoth - Similarity: 0.55
```

## Dataset

The dataset (`data/tamil_movies.csv`) contains ~40 well-known Tamil movies with columns:
- **title** - Movie name
- **genre** - Genre(s)
- **director** - Director name
- **actors** - Lead actors
- **description** - Brief plot description
- **year** - Release year

Movies span genres: Action, Thriller, Comedy, Romance, Drama, Crime, Family, Horror, Sci-Fi, Sports.

## Testing the Core Logic

You can test the recommender directly without the web interface:

```powershell
python recommender.py
```

This will run sample recommendations for test movies.

## Error Handling

The system handles:
- **Movie not found** - Shows available movies
- **Missing CSV file** - Clear error with path
- **Empty dataset** - Appropriate error message

## Notes

- This is an **educational project** demonstrating basic content-based filtering
- Recommendations are based on **metadata similarity**, not user ratings or collaborative filtering
- The system does **not** use deep learning, neural networks, or external APIs
- Similarity scores are **cosine similarity** between TF-IDF vectors (0.0 to 1.0)

## License

Educational project - free to use and modify.