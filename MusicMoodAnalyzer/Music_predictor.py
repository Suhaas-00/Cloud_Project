import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import random
from utils.data_processor import DataProcessor
from utils.model_trainer import ModelTrainer
from utils.visualizations import Visualizations

# Page configuration
st.set_page_config(
    page_title="Song Mood & Popularity Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model_trainer' not in st.session_state:
    st.session_state.model_trainer = None

def main():
    st.title("🎵 Song Mood & Popularity Predictor")
    st.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Data Upload & Analysis", "Model Training & Comparison", "Predictions", "Manual Feature Input"]
    )
    
    if page == "Data Upload & Analysis":
        data_upload_page()
    elif page == "Model Training & Comparison":
        model_training_page()
    elif page == "Predictions":
        predictions_page()
    elif page == "Manual Feature Input":
        manual_input_page()

def data_upload_page():
    st.header("📊 Data Upload & Analysis")
    uploaded_file = st.file_uploader("Upload your song dataset (CSV format)", type=['csv'])
    
    if uploaded_file:
        try:
            processor = DataProcessor()
            uploaded_file.seek(0)
            df = processor.load_data(uploaded_file)

            st.session_state.df = df
            st.session_state.data_loaded = True
            visualizer = Visualizations()
            
            st.success(f"✅ Data loaded successfully! {len(df)} songs found.")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Dataset Overview")
                st.write(f"**Number of songs:** {len(df)}")
                st.write(f"**Number of features:** {len(df.columns) - 1}")
                st.subheader("Sample Data")
                st.dataframe(df.head(), use_container_width=True)

            with col2:
                if 'mood' in df.columns:
                    st.subheader("Mood Distribution")
                    mood_counts = df['mood'].value_counts()
                    fig = px.bar(x=mood_counts.index, y=mood_counts.values,
                                 labels={'x': 'Mood', 'y': 'Count'},
                                 title="Distribution of Song Moods")
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature Statistics")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'mood' in numeric_cols:
                numeric_cols.remove('mood')

            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                st.subheader("Feature Correlation Analysis")
                correlation_fig = visualizer.create_correlation_heatmap(df[numeric_cols])
                st.plotly_chart(correlation_fig, use_container_width=True)

                st.subheader("Feature Distributions by Mood")
                selected_feature = st.selectbox("Select a feature to analyze:", numeric_cols)
                distribution_fig = visualizer.create_feature_distribution(df, selected_feature)
                st.plotly_chart(distribution_fig, use_container_width=True)
            else:
                st.warning("⚠️ No numeric features found for statistics or visualizations.")

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            st.info("Please make sure your CSV includes a 'mood' column and numeric features.")
    
    else:
        st.info("👆 Please upload a CSV file to begin analysis.")
        st.subheader("Expected Data Format")
        sample_data = {
            'name': ['Song 1', 'Song 2', 'Song 3'],
            'popularity': [68, 43, 60],
            'danceability': [0.866, 0.381, 0.346],
            'acousticness': [0.137, 0.019, 0.913],
            'energy': [0.730, 0.832, 0.139],
            'valence': [0.625, 0.166, 0.116],
            'mood': ['Happy', 'Sad', 'Calm']
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

def model_training_page():
    st.header("🤖 Model Training & Comparison")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data first in the 'Data Upload & Analysis' page.")
        return
    
    df = st.session_state.df

    st.subheader("Model Configuration")
    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)

    with col2:
        models_to_train = st.multiselect("Select Models to Train:", ["Random Forest", "SVM"],
                                         default=["Random Forest", "SVM"])

    if st.button("🚀 Train Models", type="primary"):
        if not models_to_train:
            st.error("Please select at least one model to train.")
            return
        
        with st.spinner("Training models..."):
            try:
                trainer = ModelTrainer()
                trainer.prepare_data(df, test_size=test_size, random_state=random_state)

                results = {}
                for model_name in models_to_train:
                    if model_name == "Random Forest":
                        results['Random Forest'] = trainer.train_random_forest()
                    elif model_name == "SVM":
                        results['SVM'] = trainer.train_svm()

                st.session_state.model_trainer = trainer
                st.session_state.models_trained = True

                st.success("✅ Models trained successfully!")
                display_model_results(results)

            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")

    if st.session_state.models_trained and st.session_state.model_trainer:
        st.subheader("Model Performance")
        display_model_results(st.session_state.model_trainer.model_results)

def display_model_results(results):
    for model_name, result in results.items():
        st.subheader(f"📈 {model_name} Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{result['accuracy']:.3f}")
            st.text("Classification Report:")
            st.text(result['classification_report'])

        with col2:
            if 'confusion_matrix' in result:
                visualizer = Visualizations()
                cm_fig = visualizer.create_confusion_matrix(
                    result['confusion_matrix'],
                    result.get('class_names', ['Calm', 'Energetic', 'Happy', 'Sad']),
                    f"{model_name} Confusion Matrix"
                )
                st.plotly_chart(cm_fig, use_container_width=True, key=f"cm_{random.randint(1, 10000)}")

        st.markdown("---")

def predictions_page():
    st.header("🔮 Song Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
        return

    trainer = st.session_state.model_trainer

    st.subheader("Upload Songs for Prediction")
    prediction_file = st.file_uploader(
        "Upload CSV file with song features (without mood labels)",
        type=['csv'], key="prediction_upload"
    )

    if prediction_file:
        try:
            pred_df = pd.read_csv(prediction_file, encoding='latin1')
            if 'mood' in pred_df.columns:
                pred_df.drop('mood', axis=1, inplace=True)

            st.write("Preview of uploaded data:")
            st.dataframe(pred_df.head(), use_container_width=True)

            model_choice = st.selectbox("Select model for prediction:", list(trainer.models.keys()))
            
            if st.button("🎯 Make Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    predictions = trainer.predict(pred_df, model_choice)
                    results_df = pred_df.copy()
                    results_df['Predicted_Mood'] = predictions

                    st.success("✅ Predictions completed!")
                    st.subheader("Prediction Results")
                    st.dataframe(results_df, use_container_width=True)

                    csv_buffer = io.StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button("📥 Download Predictions", csv_buffer.getvalue(), "predictions.csv", "text/csv")

                    pred_counts = pd.Series(predictions).value_counts()
                    fig = px.pie(names=pred_counts.index, values=pred_counts.values, title="Predicted Mood Distribution")
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing prediction file: {str(e)}")

def manual_input_page():
    st.header("🎚️ Manual Feature Input")

    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first.")
        return

    trainer = st.session_state.model_trainer
    st.subheader("Input Song Features")

    with st.form("manual_prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            popularity = st.slider("Popularity", 0, 100, 50)
            length = st.number_input("Length (ms)", 1000, 600000, 200000)
            danceability = st.slider("Danceability", 0.0, 1.0, 0.5, 0.01)
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5, 0.01)
            energy = st.slider("Energy", 0.0, 1.0, 0.5, 0.01)

        with col2:
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
            liveness = st.slider("Liveness", 0.0, 1.0, 0.1, 0.01)
            valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
            loudness = st.slider("Loudness (dB)", -30.0, 0.0, -10.0, 0.1)
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)

        with col3:
            tempo = st.slider("Tempo (BPM)", 50.0, 200.0, 120.0, 1.0)
            key = st.selectbox("Key", list(range(12)))
            time_signature = st.selectbox("Time Signature", [3, 4, 5])
            model_choice = st.selectbox("Select Model:", list(trainer.models.keys()))

        submitted = st.form_submit_button("🎵 Predict Mood", type="primary")

        if submitted:
            features = np.array([[
                popularity, length, danceability, acousticness, energy,
                instrumentalness, liveness, valence, loudness, speechiness,
                tempo, key, time_signature
            ]])
            feature_names = [
                'popularity', 'length', 'danceability', 'acousticness', 'energy',
                'instrumentalness', 'liveness', 'valence', 'loudness', 'speechiness',
                'tempo', 'key', 'time_signature'
            ]
            df_input = pd.DataFrame(features, columns=feature_names)

            try:
                prediction = trainer.predict(df_input, model_choice)[0]
                st.success(f"🎯 Predicted Mood: **{prediction}**")

                if hasattr(trainer.models[model_choice], 'predict_proba'):
                    probs = trainer.models[model_choice].predict_proba(trainer.scaler.transform(df_input))
                    mood_probs = pd.DataFrame({
                        'Mood': trainer.label_encoder.classes_,
                        'Probability': probs[0]
                    }).sort_values(by='Probability', ascending=False)

                    fig = px.bar(mood_probs, x='Mood', y='Probability', color='Probability',
                                 title="Prediction Confidence", color_continuous_scale='viridis')
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
