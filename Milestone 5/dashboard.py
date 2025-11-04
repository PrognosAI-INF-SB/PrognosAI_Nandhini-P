import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# File paths (assumed in same folder)
thresholds_path = 'alert_thresholds.txt'
metrics_path = 'evaluation_metrics.txt'
alert_stats_path = 'alert_statistics.csv'
alerts_sample_path = 'maintenance_alerts_sample.csv'
full_predictions_path = 'complete_test_predictions.csv'
confusion_matrix_path = 'alert_confusion_matrix.csv'
training_history_path = 'training_history.csv'
system_performance_path = 'alert_system_performance.txt'
model_config_path = 'model_configuration.txt'
data_quality_path = 'data_quality_report.txt'

# =========================
# Load CSVs
alerts_sample = pd.read_csv(alerts_sample_path)
alert_stats = pd.read_csv(alert_stats_path)
full_predictions = pd.read_csv(full_predictions_path)
confusion_matrix = pd.read_csv(confusion_matrix_path)
training_history = pd.read_csv(training_history_path)

# =========================
# Load text reports
with open(thresholds_path, 'r') as f:
    alert_thresholds_text = f.read()

with open(metrics_path, 'r') as f:
    eval_metrics_text = f.read()

with open(system_performance_path, 'r') as f:
    alert_system_performance_text = f.read()

with open(model_config_path, 'r') as f:
    model_config_text = f.read()

with open(data_quality_path, 'r') as f:
    data_quality_text = f.read()

# =========================
# Streamlit Layout

st.title("Predictive Maintenance Dashboard")

section = st.sidebar.radio("Navigate:", [
    "Thresholds & Model Info",
    "Performance Metrics",
    "Alert Distribution",
    "Confusion Matrix",
    "Training Progress",
    "Actual vs Predicted",
    "Sample Alerts",
    "Full Predictions",
    "Additional Reports"
])

def color_alert_status(val):
    color_map = {'NORMAL': 'lightgreen', 'WARNING': 'orange', 'CRITICAL': 'red'}
    return f'background-color: {color_map.get(val,"white")}; font-weight: bold'

# Thresholds and Info
if section == "Thresholds & Model Info":
    st.header("Alert Thresholds")
    st.code(alert_thresholds_text)

    with st.expander("Model Configuration"):
        st.code(model_config_text)

    with st.expander("Data Quality Report"):
        st.code(data_quality_text)

# Performance Metrics
elif section == "Performance Metrics":
    st.header("Evaluation Metrics")
    st.code(eval_metrics_text)

    st.header("Alert System Performance")
    st.code(alert_system_performance_text)

# Alert Distribution
elif section == "Alert Distribution":
    st.header("Alert Status Distribution")
    fig, ax = plt.subplots()
    ax.bar(alert_stats['Status'], alert_stats['Count'], color=['green', 'orange', 'red'])
    ax.set_xlabel("Alert Status")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    st.dataframe(alert_stats)

# Confusion Matrix
elif section == "Confusion Matrix":
    st.header("Alert Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix.iloc[:, 1:], annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Training Progress
elif section == "Training Progress":
    st.header("Training History")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Loss over Epochs")
        fig, ax = plt.subplots()
        ax.plot(training_history['Epoch'], training_history['Training_Loss'], label='Training Loss')
        ax.plot(training_history['Epoch'], training_history['Validation_Loss'], label='Validation Loss')
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)
    with col2:
        st.subheader("MAE over Epochs")
        fig, ax = plt.subplots()
        ax.plot(training_history['Epoch'], training_history['Training_MAE'], label='Training MAE')
        ax.plot(training_history['Epoch'], training_history['Validation_MAE'], label='Validation MAE')
        ax.set_ylabel("MAE")
        ax.legend()
        st.pyplot(fig)

# Actual vs Predicted
elif section == "Actual vs Predicted":
    st.header("Actual vs Predicted RUL")
    show_line = st.checkbox("Show Perfect Prediction Line", value=True)
    fig, ax = plt.subplots()
    ax.scatter(full_predictions['Actual_RUL'], full_predictions['Predicted_RUL'], alpha=0.5)
    if show_line:
        ax.plot([full_predictions['Actual_RUL'].min(), full_predictions['Actual_RUL'].max()],
                [full_predictions['Actual_RUL'].min(), full_predictions['Actual_RUL'].max()],
                color='red', linestyle='--', lw=2)
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    ax.set_title("Actual vs Predicted RUL")
    st.pyplot(fig)

# Sample maintenance alerts
elif section == "Sample Alerts":
    st.header("First 20 Maintenance Alerts")
    filter_status = st.multiselect("Filter alert status",
                                   options=alerts_sample['Alert_Status'].unique(),
                                   default=alerts_sample['Alert_Status'].unique())
    filtered = alerts_sample[alerts_sample['Alert_Status'].isin(filter_status)]
    st.dataframe(filtered.style.applymap(color_alert_status, subset=['Alert_Status']))

# Full Test Predictions
elif section == "Full Predictions":
    st.header("Complete Test Predictions")
    alert_filter = st.multiselect("Filter Alert Status", options=full_predictions['Predicted_Alert'].unique(),
                                  default=full_predictions['Predicted_Alert'].unique())
    error_threshold = st.slider("Filter by Max Absolute Error", 0, int(full_predictions['Absolute_Error'].max()), int(full_predictions['Absolute_Error'].max()))
    filtered_pred = full_predictions[(full_predictions['Predicted_Alert'].isin(alert_filter)) & 
                                     (full_predictions['Absolute_Error'] <= error_threshold)]
    st.dataframe(filtered_pred)

# Additional Reports
elif section == "Additional Reports":
    st.header("Additional Details")
    with st.expander("System Performance"):
        st.code(alert_system_performance_text)
    with st.expander("Model Configuration"):
        st.code(model_config_text)
    with st.expander("Data Quality Summary"):
        st.code(data_quality_text)

# Footer
st.markdown("---")
st.write("Dashboard last updated:", pd.Timestamp.now())
st.write("Streamlit Version:", st.__version__)
