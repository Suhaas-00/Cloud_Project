import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self, uploaded_file):
        """Load and clean the uploaded CSV file with fallback for bad formatting."""
        try:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, encoding='ISO-8859-1', header=None)

            if df_raw.empty:
                raise ValueError("Uploaded file is empty.")

            # Handle all-in-one-column issue
            if df_raw.shape[1] == 1:
                df_split = df_raw[0].str.split(",", expand=True)
                df_split.columns = df_split.iloc[0]
                df_split = df_split[1:]
                df_split = df_split.loc[:, ~df_split.columns.duplicated(keep='first')]
                df_split = df_split.dropna(axis=1, how='all')
                df = df_split
            else:
                # If multiple columns exist already, read normally
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

            # Normalize column names
            df.columns = df.columns.str.strip().str.lower()

            # Remove unnamed/duplicate/null columns
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.loc[:, df.columns.notnull()]

            # Check required 'mood' column
            if 'mood' not in df.columns:
                raise ValueError("Missing required column: 'mood'")

            return self.clean_data(df)

        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def clean_data(self, df):
        """Clean and preprocess the dataset"""
        # Drop common irrelevant columns
        columns_to_drop = ['id', 'album', 'artist', 'release_date']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Fill missing numeric values with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

        # Fill missing categorical values (except 'mood') with 'Unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'mood':
                df[col] = df[col].fillna('Unknown')

        # Drop rows where 'mood' is missing
        df = df.dropna(subset=['mood'])

        return df

    def prepare_features(self, df):
        """Separate features and target"""
        X = df.drop(columns=['name', 'mood'], errors='ignore')
        y = df['mood']
        X = X.select_dtypes(include=[np.number])
        return X, y

    def encode_target(self, y):
        """Encode mood labels to numeric"""
        return self.label_encoder.fit_transform(y)

    def scale_features(self, X_train, X_test=None):
        """Apply standard scaling to numeric features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        return X_train_scaled

    def get_feature_names(self, df):
        """Get list of features used for training"""
        return [col for col in df.columns if col not in ['name', 'mood']]
