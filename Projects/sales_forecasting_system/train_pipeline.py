from sklearn.model_selection import train_test_split

from src.data_ingestion import load_data
from src.preprocessing import preprocess
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.utils import save_model

# Load Dataset
df = load_data()

# Preprocess
df = preprocess(df)

# Features & Target
X = df.drop("Sales", axis=1)
y = df["Sales"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = train_model(X_train, y_train)

# Evaluate
score = evaluate_model(
    model,
    X_test,
    y_test
)

print("Model R2 Score:", score)

# Save Model
save_model(model)