Suicide Behavior Detection in Social Media-

Project Overview-
This project focuses on detecting suicidal behavior in social media posts.
The motivation behind this work is to support early identification of individuals at risk, providing opportunities for timely intervention and potentially saving lives.
In the future, such a model could be integrated into organizations and platforms that provide emotional support and crisis assistance.

Objectives-
Detect suicidal intent in posts from online communities.
Help identify individuals in distress and enable organizations to provide real-time support.
Explore how Natural Language Processing (NLP) and Machine Learning can be applied for mental health–related tasks.

Dataset-
Data collected from Reddit:
Subreddits: SuicideWatch, offmychest, and Anxiety.
Additional data generated using GPT-based text generation:
A sample of real posts was used as input to create synthetic but realistic new posts.
Final dataset: 1,800 labeled posts
Balanced classes: 900 suicidal (positive class) and 900 non-suicidal (negative class).

Methods & Technologies-
Programming: Python
Libraries: PyTorch, HuggingFace Transformers, Scikit-learn, Pandas, NumPy
Models Used:
BERT, DistilBERT, RoBERTa, XLNet, MentalBERT (transformer-based models), SVM (optimized baseline)
Techniques:
Data preprocessing & cleaning (tokenization, normalization, deduplication)
Train/Validation/Test split with fixed folds
Hyperparameter optimization with Optuna
Ensemble methods (majority voting)
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

Result-
The best performing model was DistilBERT, selected based on achieving the highest Recall score – the most important metric for this task, since it prioritizes correctly identifying posts with suicidal intent.


Suicide_Behavior_Detection_Project/
│── data/
│ ├── ensemble/ # Data prepared for ensemble models
│ ├── interim/ # Intermediate datasets
│ ├── processed/ # Cleaned and processed datasets
│ │ └── splits/ # Train/validation/test splits and fold assignments
│ └── raw/ # Raw data (original Reddit dumps)
│
│── reports/
│ └── figures/ # figures, plots, visualizations
│
│── src/
│ ├── data_acquisition/ # Scripts for data collection (e.g., Reddit API)
│ ├── data_prep/ # Scripts for cleaning, merging, balancing datasets
│ ├── eda/ # Exploratory Data Analysis (EDA) notebooks & scripts
│ └── modeling/
│ ├── ensemble/ # Ensemble method
│ ├── evaluation/ # Model evaluation scripts & metrics
│ ├── final/ # Final training runs for all 6 models
│ └── optimization/ # Hyperparameter tuning scripts (Optuna, etc.)
│
│── .env # Environment variables (not committed to GitHub)
│── .gitignore # Git ignore file
│── README.md # Project documentation
│── requirements.txt # Python dependencies

Future Work-
Expand dataset with more sources and languages.
Incorporate multimodal analysis (images + text).
Explore deployment as an API/dashboard for real-time monitoring.

Ethical Considerations-
This project deals with sensitive content.
It is intended for research and educational purposes only.

Authors-
Veronika Seman (Yezreel Valley College, B.Sc. Information Systems, Data Science specialization)
Project as part of the final capstone in Data Science