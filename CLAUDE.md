# Portakal Fiyatı Tahmini (Orange Price Prediction)

## Project Overview
Machine learning project for predicting orange prices. Python-based data science pipeline.

## Tech Stack
- **Language:** Python 3.x
- **ML/Data:** pandas, scikit-learn, numpy
- **Visualization:** matplotlib, seaborn
- **Environment:** Anaconda/Miniconda

## Project Structure
```
portakalfiyatitahmini/
├── data/              # Raw and processed datasets
│   ├── raw/           # Original data files
│   └── processed/     # Cleaned/transformed data
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data/          # Data loading and preprocessing
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and evaluation
│   └── utils/         # Utility functions
├── tests/             # Unit tests
├── models/            # Saved model artifacts
├── reports/           # Generated analysis and figures
├── requirements.txt   # Python dependencies
└── CLAUDE.md          # This file
```

## Iron Loop Methodology
This project follows the CTOC Iron Loop:
1. **Plan** → Define what to build
2. **Code** → Implement with quality
3. **Test** → Validate correctness
4. **Review** → Check quality gates
5. **Ship** → Deploy or deliver

## Quality Gates
- All code must have docstrings for public functions
- Tests must pass before merging
- Data pipeline steps must be reproducible
- Model metrics must be logged and tracked

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run notebooks
jupyter notebook notebooks/
```

## Conventions
- Use snake_case for all Python files and functions
- Keep notebooks clean: restart kernel and run all before committing
- Never commit raw data files larger than 10MB to git
- Use .gitignore for data/, models/, and __pycache__/
