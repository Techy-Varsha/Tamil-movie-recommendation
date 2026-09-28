# Tamil Movies Project

A data analysis project exploring Tamil cinema with a dataset of ~40 Tamil movies.

## Project Structure

```
tamil-movies-project/
├── data/
│   └── tamil_movies.csv      # Main dataset with 40 Tamil movies
├── scripts/
│   └── (analysis scripts will go here)
├── notebooks/
│   └── (Jupyter notebooks for exploration)
├── docs/
│   └── (documentation)
└── README.md
```

## Dataset: tamil_movies.csv

The dataset contains 40 Tamil movies spanning from 1992 to 2023, featuring movies from major stars like Rajinikanth and Vijay.

### Columns

| Column | Description |
|--------|-------------|
| `id` | Unique identifier |
| `title` | Movie title |
| `year` | Release year |
| `genre` | Primary genre |
| `director` | Director name |
| `lead_actor` | Lead actor |
| `lead_actress` | Lead actress |
| `music_director` | Music director |
| `runtime_minutes` | Movie runtime in minutes |
| `box_office_crores` | Box office collection in crores (₹) |
| `rating` | IMDb-style rating (out of 10) |
| `language` | Language (all Tamil) |

### Sample Movies Included

**Rajinikanth Movies (8):**
- Annamalai (1992), Baashha (1995), Muthu (1995), Padayappa (1999)
- Sivaji (2007), Enthiran (2010), 2.0 (2018), Jailer (2023)

**Vijay Movies (32):**
- From Naalaiya Theerpu (1992) to Varisu (2023)
- Notable: Ghilli, Thuppakki, Kaththi, Mersal, Master, Leo, Jailer

## Quick Start

```bash
# View the dataset
cat data/tamil_movies.csv

# Or with Python
python3 -c "
import pandas as pd
df = pd.read_csv('data/tamil_movies.csv')
print(df.head())
print(f'\nTotal movies: {len(df)}')
print(f'Year range: {df.year.min()} - {df.year.max()}')
print(f'Avg rating: {df.rating.mean():.2f}')
print(f'Total box office: ₹{df.box_office_crores.sum():.1f} crores')
"
```

## Potential Analyses

- Box office trends over decades
- Director/actor collaborations
- Genre distribution
- Rating vs box office correlation
- Music director impact
- Decade-wise comparison

## License

This dataset is for educational/analysis purposes. Box office figures are approximate estimates from public sources.