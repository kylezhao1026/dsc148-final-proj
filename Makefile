DATA := data/games_march2025_cleaned.csv
TARGET := num_reviews_total

.PHONY: eda baseline experiments errors demo paper

eda:
	python3 -m src.eda --data $(DATA) --target-column $(TARGET)

baseline:
	python3 -m src.train --data $(DATA) --target-column $(TARGET) --models majority logistic_regression linear_svm_sgd

experiments:
	python3 -m src.experiments --data $(DATA) --target-column $(TARGET)

errors:
	python3 -m src.error_analysis --data $(DATA) --model models/best_experiment_model.joblib --target-column $(TARGET)

demo:
	python3 -m streamlit run app/streamlit_app.py --server.port 8501

paper:
	latexmk -pdf -interaction=nonstopmode -halt-on-error -cd reports/paper.tex
