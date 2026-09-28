# Movie Recommender System

A content-based movie recommendation system using TF-IDF vectorization and cosine similarity. Includes both a Streamlit web application and a reusable Python module.

## Project Structure

```
simple trial/
├── app.py                    # Streamlit web application
├── recommender.py            # MovieRecommender class (reusable module)
├── requirements.txt          # Python dependencies
├── tamil-movies-project/     # Tamil movies analysis subproject
│   ├── data/
│   │   └── tamil_movies.csv  # Dataset with 40 Tamil movies
│   ├── scripts/
│   │   └── explore.py        # Data exploration script
│   └── notebooks/
│       └── explore_tamil_movies.ipynb
└── README.md                 # This file
```

## Features

- **Content-Based Filtering**: Uses TF-IDF on movie features (genres, keywords, overview, tagline, cast, crew)
- **Cosine Similarity**: Computes pairwise similarity between all movies
- **Streamlit Web App**: Interactive UI for getting movie recommendations
- **Reusable Module**: `MovieRecommender` class for integration into other projects
- **Model Persistence**: Save/load fitted models with pickle
- **Tamil Movies Analysis**: Separate data analysis project for Tamil cinema

## Installation

### Prerequisites

- **Python 3.8 or higher** (recommended: Python 3.10+)
- **pip** (Python package manager)

### Windows Setup

1. **Open Command Prompt or PowerShell** as Administrator (recommended)

2. **Create a virtual environment** (recommended):
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Upgrade pip**:
   ```cmd
   python -m pip install --upgrade pip
   ```

4. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

## Running the Application

### Streamlit Web App (Main App)

```cmd
# From the project root directory
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Recommender Module (Python)

```python
from recommender import MovieRecommender

# Initialize
recommender = MovieRecommender(max_features=5000, stop_words='english')

# Load your dataset (CSV with columns: title, genres, keywords, overview, tagline, cast, crew)
recommender.load_dataset('your_movies.csv')

# Fit the model
recommender.fit()

# Get recommendations
recommendations = recommender.recommend_movies('The Dark Knight', top_n=5)
print(recommendations)
```

### Running the Demo (Built-in Sample Data)

```cmd
python recommender.py
```

This creates a sample dataset and demonstrates the recommender.

## Tamil Movies Analysis Subproject

Navigate to the subproject directory:
```cmd
cd tamil-movies-project
```

### Run Exploration Script
```cmd
python scripts\explore.py
```

### Open Jupyter Notebook
```cmd
jupyter notebook notebooks\explore_tamil_movies.ipynb
```

### Quick Data View (Command Line)
```cmd
python -c "
import pandas as pd
df = pd.read_csv('data\tamil_movies.csv')
print(df.head())
print(f'\nTotal movies: {len(df)}')
print(f'Year range: {df.year.min()} - {df.year.max()}')
print(f'Avg rating: {df.rating.mean():.2f}')
print(f'Total box office: ₹{df.box_office_crores.sum():.1f} crores')
"
```

## Dataset Format

The recommender expects a CSV with at least a `title` column. For best results, include these columns:

| Column | Description |
|--------|-------------|
| `title` | Movie title (required) |
| `genres` | Genres (e.g., "Action, Adventure, Sci-Fi") |
| `keywords` | Keywords/tags |
| `overview` | Plot summary |
| `tagline` | Movie tagline |
| `cast` | Main cast names |
| `crew` | Director/writer names |
| `vote_average` | Rating (optional) |
| `release_date` | Release date (optional) |

## Saving/Loading Models

```python
# Save fitted model
recommender.save_model('movie_recommender.pkl')

# Load saved model
loaded_recommender = MovieRecommender.load_model('movie_recommender.pkl')
recommendations = loaded_recommender.recommend_movies('Inception', top_n=5)
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure virtual environment is activated and dependencies installed:
   ```cmd
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Streamlit not opening browser**: Run with explicit browser flag:
   ```cmd
   streamlit run app.py --server.headless=false
   ```

3. **Port already in use**: Specify a different port:
   ```cmd
   streamlit run app.py --server.port 8502
   ```

4. **Encoding issues with CSV**: Ensure CSV is UTF-8 encoded, or specify encoding:
   ```python
   recommender.load_dataset('movies.csv', encoding='utf-8')
   ```

### Windows-Specific Notes

- Use backslashes `\` in paths for Command Prompt, forward slashes `/` work in PowerShell
- Virtual environment activation: `venv\Scripts\activate` (cmd) or `venv\Scripts\Activate.ps1` (PowerShell)
- If PowerShell scripts are disabled, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=2.0.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical operations |
| matplotlib | >=3.7.0 | Plotting |
| seaborn | >=0.12.0 | Statistical visualization |
| scikit-learn | >=1.3.0 | TF-IDF, cosine similarity |
| streamlit | >=1.28.0 | Web app framework |
| jupyter | >=1.0.0 | Notebook support |
| notebook | >=7.0.0 | Jupyter notebook server |

## License

This project is for educational/demo purposes. The sample movie data is fictional. The Tamil movies dataset uses approximate box office figures from public sources.