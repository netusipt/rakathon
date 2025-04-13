from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def main():
    # Assume X has both categorical and numeric columns
    numeric_features = ["age", "income"]
    categorical_features = ["color", "job_type"]

    # Simple pipeline for pre-processing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Plug into RandomForest
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
    )

    X_train = ...
    y_train = ...

    clf.fit(X_train, y_train)

    # Feature importances
    importances = clf.named_steps["classifier"].feature_importances_
    print(importances)


if __name__ == "__main__":
    main()
