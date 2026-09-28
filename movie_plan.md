# MASTER PROMPT — SIMPLE TAMIL MOVIE RECOMMENDATION SYSTEM

You are a beginner-friendly software developer and ML engineer.

Build a **very simple and basic Tamil Movie Recommendation System** using Python.

The goal is to create a small, working project that recommends Tamil movies similar to a movie selected by the user.

Do NOT over-engineer this project.

I want this to be simple enough for a beginner to understand, run, modify, and explain in a college project/demo.

---

## 1. PROJECT GOAL

Create a movie recommendation system containing **ONLY Tamil movies**.

The user should be able to:

1. Open the application.
2. Select or enter the name of a Tamil movie.
3. Click "Recommend".
4. Get a list of similar Tamil movies.

Example:

User selects:

> Vikram

System returns something like:

> Kaithi
> Master
> Leo
> Thani Oruvan
> Theeran Adhigaaram Ondru

The recommendations do not need to be perfect.

The main goal is to demonstrate the basic concept of a **movie recommendation system**.

---

# 2. RECOMMENDATION METHOD

Use a simple **content-based filtering approach**.

Do NOT implement:

* Deep learning
* Neural networks
* Collaborative filtering
* User accounts
* Login
* Complex databases
* External APIs
* LLMs
* Real-time learning
* Complex cloud infrastructure

Use movie metadata to determine similarity.

---

# 3. DATASET

Create a simple CSV dataset containing around **30–50 Tamil movies**.

The dataset should contain columns such as:

```text
title
genre
director
actors
description
year
```

Example:

```text
title,genre,director,actors,description,year
Vikram,Action Thriller,Shankar,Kamal Haasan Vijay Sethupathi Fahadh Faasil,An action thriller involving a special agent and a dangerous crime network,2022
Kaithi,Action Thriller,Lokesh Kanagaraj,Karthi,Night-time action thriller involving a prisoner and police officers,2019
```

Use **only Tamil-language movies**.

Include a reasonable mixture of:

* Action
* Thriller
* Comedy
* Romance
* Drama
* Crime
* Family
* Horror

Use well-known Tamil movies so the recommendations make intuitive sense.

Do not use movies from other Indian languages just because they are popular.

---

# 4. FEATURE ENGINEERING

Create a combined text feature from:

```text
genre
director
actors
description
```

For example:

```python
combined_features = genre + director + actors + description
```

Use:

**TF-IDF Vectorizer**

from scikit-learn.

Then calculate similarity using:

**Cosine Similarity**

The basic pipeline should be:

```text
Movie Dataset
      ↓
Combine movie features
      ↓
TF-IDF Vectorization
      ↓
Cosine Similarity
      ↓
Find similar movies
      ↓
Display Top 5 Recommendations
```

---

# 5. CORE RECOMMENDATION FUNCTION

Create a function similar to:

```python
recommend_movies(movie_title, num_recommendations=5)
```

It should:

1. Find the selected movie.
2. Find its TF-IDF vector.
3. Calculate cosine similarity with all other movies.
4. Sort movies by similarity.
5. Exclude the selected movie itself.
6. Return the top 5 similar Tamil movies.

Example:

```python
recommend_movies("Vikram")
```

should return a list of similar movies.

Handle the case where the movie does not exist.

Example:

```text
Movie not found. Please choose a movie from the list.
```

---

# 6. SIMPLE USER INTERFACE

Create a very simple web interface using **Streamlit**.

The UI should contain:

### Title

```text
🎬 Tamil Movie Recommendation System
```

### Description

```text
Select a Tamil movie and discover similar movies!
```

### Movie selection

Use a dropdown containing all movie titles.

Example:

```text
Select a movie:
[ Vikram ▼ ]
```

### Button

```text
Recommend Movies
```

When the user clicks the button, display:

```text
🎬 Recommended Movies
```

Then show the top 5 recommendations.

For each recommendation display:

```text
Movie Name
Genre
Director
Year
```

Keep the UI clean and simple.

---

# 7. PROJECT STRUCTURE

Create the following structure:

```text
tamil_movie_recommender/
│
├── data/
│   └── tamil_movies.csv
│
├── app.py
├── recommender.py
├── requirements.txt
└── README.md
```

### app.py

Contains the Streamlit interface.

### recommender.py

Contains:

* Dataset loading
* Feature creation
* TF-IDF
* Cosine similarity
* Recommendation function

### tamil_movies.csv

Contains the Tamil movie dataset.

### requirements.txt

Include only the required packages, such as:

```text
pandas
scikit-learn
streamlit
```

### README.md

Explain:

* Project overview
* How the recommendation system works
* Technologies used
* Installation
* How to run
* Example usage

---

# 8. KEEP THE CODE BEGINNER-FRIENDLY

This is extremely important.

Write simple Python code.

Avoid unnecessary:

* Classes
* Design patterns
* Complex abstractions
* APIs
* Databases
* Docker
* Authentication
* Environment variables
* Microservices

The entire project should be understandable by a beginner who knows basic Python.

Add comments explaining important sections.

---

# 9. ERROR HANDLING

Handle simple cases such as:

### Movie not found

Display:

```text
Movie not found.
Please select a movie from the dropdown.
```

### Empty dataset

Display an appropriate error.

### Missing CSV

Give a clear error explaining that the dataset file is missing.

---

# 10. RECOMMENDATION EXPLANATION

For each recommendation, display the similarity score if possible.

Example:

```text
1. Kaithi
   Genre: Action Thriller
   Director: Lokesh Kanagaraj
   Year: 2019
   Similarity: 0.72
```

Round the similarity score to 2 decimal places.

Also add a simple explanation above the results:

```text
These movies are recommended because their genre,
director, actors, and description are similar to the selected movie.
```

---

# 11. DATASET REQUIREMENTS

Create approximately **40 Tamil movies**.

Make sure the dataset is diverse.

Include movies from different years and genres.

Example movies can include:

* Baasha
* Annamalai
* Muthu
* Padayappa
* Ghilli
* Vaaranam Aayiram
* Vinnaithaandi Varuvaayaa
* Kaithi
* Vikram
* Master
* Leo
* Thani Oruvan
* Theeran Adhigaaram Ondru
* Ratsasan
* Soorarai Pottru
* Jai Bhim
* 96
* OK Kanmani
* Soodhu Kavvum
* Doctor
* Good Night
* Parking
* Maanaadu
* Maanagaram
* Asuran
* Pariyerum Perumal

You may choose other well-known Tamil movies as needed.

Ensure the metadata is internally consistent and clearly identifies the movies as Tamil films.

---

# 12. IMPORTANT — DO NOT FABRICATE COMPLEX RESULTS

This is a basic educational recommendation system.

Do not claim:

> "AI understands the user's taste."

Instead explain accurately:

> "The system recommends movies based on similarity between their metadata using TF-IDF and cosine similarity."

Do not invent accuracy percentages.

If you evaluate the system, clearly explain that recommendation systems of this type are generally evaluated differently from ordinary classification problems.

---

# 13. RUNNING THE PROJECT

Make the project work on **Windows**.

The README must contain exact commands.

For example:

```powershell
cd tamil_movie_recommender

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

The application should then open in the browser.

---

# 14. TEST THE PROJECT

After creating the files:

1. Check that all imports work.
2. Check that the CSV loads correctly.
3. Check that TF-IDF is generated.
4. Check that cosine similarity works.
5. Test at least 3 different Tamil movies.
6. Test an invalid movie name.
7. Check that the Streamlit app starts successfully.
8. Fix any errors you encounter.

Do not simply create the files and assume they work.

---

# 15. FINAL OUTPUT

When you finish, show me:

### 1. Project structure

```text
tamil_movie_recommender/
...
```

### 2. Explanation of how it works

Explain this very simply:

```text
Movie metadata
      ↓
Combined text
      ↓
TF-IDF
      ↓
Cosine Similarity
      ↓
Top 5 similar movies
```

### 3. Files created

Tell me exactly which files you created.

### 4. Installation commands

Give the exact Windows commands.

### 5. Run command

Give:

```powershell
streamlit run app.py
```

### 6. Example

Show an example such as:

```text
Input:
Vikram

Output:
Kaithi
Master
Leo
Maanagaram
Theeran Adhigaaram Ondru
```

The exact recommendations should come from the actual model rather than being hard-coded.

---

# MOST IMPORTANT INSTRUCTION

Keep this project **VERY SIMPLE**.

I am intentionally asking for a basic Tamil movie recommendation system.

Do not turn it into a production-level Netflix-style recommendation engine.

The final project should be:

**Simple + Functional + Easy to Understand + Easy to Demonstrate + Actually Runnable.**

Start by creating the project structure and files, then implement the recommendation system and test it.
