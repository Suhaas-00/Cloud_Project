import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.model_results = {}
        self.scaler = None
        self.categorical_cols = []
        self.numeric_cols = []

    def prepare_data(self, df, target_column='mood', test_size=0.2, random_state=42):
        # Drop columns not needed for training
        drop_columns = ['name'] if 'name' in df.columns else []
        X = df.drop(columns=drop_columns + [target_column])
        y = self.label_encoder.fit_transform(df[target_column])

        # Encode categorical features
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        self.numeric_cols = X.select_dtypes(include=['number']).columns

        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Scale numeric features
        self.scaler = StandardScaler()
        X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def train_random_forest(self):
        clf = RandomForestClassifier()
        clf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = clf
        return self._evaluate_model(clf, 'Random Forest')

    def train_svm(self):
        clf = SVC(probability=True)
        clf.fit(self.X_train, self.y_train)
        self.models['SVM'] = clf
        return self._evaluate_model(clf, 'SVM')

    def _evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=self.label_encoder.classes_)
        cm = confusion_matrix(self.y_test, y_pred)
        result = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': list(self.label_encoder.classes_)
        }
        self.model_results[model_name] = result
        return result

    def predict(self, X, model_name):
        X_copy = X.copy()

        # Drop 'name' column if present
        if 'name' in X_copy.columns:
            X_copy = X_copy.drop(columns=['name'])

        # Encode categorical features
        for col in self.categorical_cols:
            le = LabelEncoder()
            X_copy[col] = le.fit_transform(X_copy[col].astype(str))

        # Scale numeric columns
        X_copy[self.numeric_cols] = self.scaler.transform(X_copy[self.numeric_cols])

        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"No trained model named '{model_name}'")

        y_pred = model.predict(X_copy)
        return self.label_encoder.inverse_transform(y_pred)
