from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Definations of categorical, numerical and text features

text_features = "journal_text"

categorical_features = [
    "ambience_type",
    "time_of_day",
    "previous_day_mood",
    "face_emotion_hint",
    "reflection_quality",
]

numerical_features = ["duration_min", "sleep_hours", "energy_level", "stress_level"]

# Feature Preprocessing

text_transformer = TfidfVectorizer(stop_words="english", max_features=500)

numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# explicitly labeling missing value with label "missing"
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# final preprocessor
# using remainder="drop" to drop columns not specified in the definition section(like id)

preprocessor_textmeta = ColumnTransformer(
    transformers=[
        ("text", text_transformer, text_features),
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)

preprocessor_textonly = ColumnTransformer(
    transformers=[("text", text_transformer, text_features)], remainder="drop"
)

preprocessor_nontext = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop",
)
