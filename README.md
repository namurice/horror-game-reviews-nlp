# The Psychology of Fear in Horror Games: NLP & Sentiment Analysis of Player Reviews

Bachelor's thesis project, BSc Computer Science, Caucasus University (2025).

**Question:** Which horror game techniques (jump scares, psychological horror, tension-building, etc.) actually create fear, and how do players describe it?

## Pipeline
1. **Data collection**: Steam player reviews of horror games, game descriptions and Steam tags, plus Metacritic scores (`metacritic.py` tests the Metacritic API).
2. **Feature engineering** (`tags.py`): keeps fear-relevant tags and triggers (Survival Horror, Psychological Horror, Jump Scares, Isolation, Gore and more) and encodes them per game.
3. **Weak labelling** (`auto_label.py`): labels each review automatically using **VADER** sentiment, **spaCy** word-vector similarity to horror and positive vocabulary, and intensity patterns ("aaaa", "omfg"), so "terrifying, loved it" counts as a positive horror reaction.
4. **Model training** (`finetuning.py`): fine-tunes **RoBERTa-base** (Hugging Face Transformers, PyTorch) on the auto-labelled reviews for binary sentiment classification. Trained on Google Colab GPU.
5. **Inference** (`inference.py`, `afterml.py`): runs the fine-tuned model over all reviews to get predicted sentiment and confidence scores.
6. **Analysis & visualisation** (`group_and_visualize.py`, `visualization*.py`): aggregates per game and compares auto-labels vs. model predictions, game tags and fear elements vs. review sentiment, and review sentiment vs. Metacritic score (matplotlib).

## Tech
Python · pandas · NumPy · spaCy · VADER · PyTorch · Hugging Face Transformers (RoBERTa) · scikit-learn · matplotlib · Google Colab

## Note
Datasets and the trained model checkpoint are not included because of their size. Paths in the scripts point to the original local or Colab folders.
