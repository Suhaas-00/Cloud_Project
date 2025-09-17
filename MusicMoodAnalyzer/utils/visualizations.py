import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

class Visualizations:
    def create_correlation_heatmap(self, df):
        """Creates a heatmap of feature correlations for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns available for correlation heatmap.")
        
        corr = numeric_df.corr().round(2)

        fig = px.imshow(
            corr,
            text_auto=True,
            title="Feature Correlation Heatmap",
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Features",
            margin=dict(l=40, r=40, t=60, b=40),
            coloraxis_colorbar=dict(title="Correlation")
        )
        return fig

    def create_feature_distribution(self, df, feature):
        """Creates a box plot of a feature grouped by mood."""
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame.")
        if 'mood' not in df.columns:
            raise ValueError("Column 'mood' is required for distribution plots.")

        fig = px.box(
            df,
            x="mood",
            y=feature,
            color="mood",
            title=f"Distribution of '{feature}' by Mood",
            points="all"
        )
        fig.update_layout(
            xaxis_title="Mood",
            yaxis_title=feature,
            boxmode='group',
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig

    def create_confusion_matrix(self, cm, class_names, title="Confusion Matrix"):
        """Creates a Plotly heatmap for confusion matrix with annotations."""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            hoverongaps=False,
            colorscale='Viridis',
            showscale=True
        ))

        # Add annotation values
        annotations = []
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                annotations.append(
                    dict(
                        x=class_names[j],
                        y=class_names[i],
                        text=str(cm[i][j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i][j] > np.max(cm) / 2 else "black")
                    )
                )

        fig.update_layout(
            title=title,
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            annotations=annotations,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        return fig
