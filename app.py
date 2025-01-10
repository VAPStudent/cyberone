import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from faker import Faker
import re
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import ollama

# Step 1: Generate synthetic logs
def generate_synthetic_logs(num_logs):
    fake = Faker()
    users = ['rbinnajim.c', 'nmzuabi', 'mhassanien.c', 'sadmin']
    logs = []
    now = datetime.now()

    for _ in range(num_logs):
        timestamp = now - timedelta(minutes=random.randint(1, 10000))
        user = random.choice(users)
        ip = fake.ipv4()
        session_id = fake.uuid4()

        # Log for LDAP Authentication
        log = f"{timestamp} LDAP Authentication SUCCESS for User[{user}]. Client IP[{ip}]\n"

        if user != 'sadmin':
            log += f"{timestamp} OTP Authentication SUCCESS for User[{user}]. Client IP[{ip}]\n"

        log += f"{timestamp} Session ID[{session_id}], Authentication Done. Redirecting to Siebel\n"
        logs.append(log)

    return "\n".join(logs)

# Step 2: Parse logs
def parse_logs(log_data):
    ldap_pattern = r"(\d+-\d+-\d+ \d+:\d+:\d+\.\d+).*?LDAP Authentication SUCCESS for User\[(.*?)\]"
    otp_pattern = r"(\d+-\d+-\d+ \d+:\d+:\d+\.\d+).*?OTP Authentication SUCCESS for User\[(.*?)\]"

    ldap_matches = re.findall(ldap_pattern, log_data)
    otp_matches = re.findall(otp_pattern, log_data)

    ldap_df = pd.DataFrame(ldap_matches, columns=['timestamp', 'user'])
    ldap_df['timestamp'] = pd.to_datetime(ldap_df['timestamp'])
    otp_df = pd.DataFrame(otp_matches, columns=['timestamp', 'user'])
    otp_df['timestamp'] = pd.to_datetime(otp_df['timestamp'])

    return ldap_df, otp_df

# Step 3: Read Excel data
def read_excel_data(uploaded_file):
    try:
        # Try reading the uploaded Excel file
        return pd.read_excel(uploaded_file, sheet_name=None)  # Reads all sheets
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None

# Step 4: Visualize data
def visualize_data(ldap_df, otp_df):
    st.subheader("Login Trends Over Time")

    # Plot LDAP Authentication Trend
    ldap_df['date'] = ldap_df['timestamp'].dt.date
    ldap_counts = ldap_df.groupby('date').size()

    fig = px.line(ldap_counts, title="LDAP Authentication Success Trend")
    st.plotly_chart(fig)

    # Plot OTP Authentication Trend
    otp_df['date'] = otp_df['timestamp'].dt.date
    otp_counts = otp_df.groupby('date').size()

    fig = px.line(otp_counts, title="OTP Authentication Success Trend", color_discrete_sequence=['green'])
    st.plotly_chart(fig)

    # Identify and plot anomalies
    st.subheader("Anomalies Detection")

    ldap_df['hour'] = ldap_df['timestamp'].dt.hour
    hourly_counts = ldap_df.groupby('hour').size()

    threshold = hourly_counts.mean() + 2 * hourly_counts.std()
    anomalies = hourly_counts[hourly_counts > threshold]

    fig, ax = plt.subplots()
    hourly_counts.plot(ax=ax, color="blue", label="Hourly Counts")
    plt.axhline(y=threshold, color='orange', linestyle='--', label='Threshold')

    if not anomalies.empty:
        anomalies.plot(ax=ax, kind="bar", color="red", label="Anomalies")
    else:
        st.write("No anomalies detected in hourly counts.")

    plt.legend()
    st.pyplot(fig)
    plt.clf()

    # Prediction of LDAP authentication trend
    st.subheader("Trend Prediction")

    ldap_counts.index = pd.to_datetime(ldap_counts.index)
    ldap_counts = ldap_counts.reset_index()
    ldap_counts['date_ordinal'] = ldap_counts['date'].apply(lambda x: x.toordinal())

    X = np.array(ldap_counts['date_ordinal']).reshape(-1, 1)
    y = np.array(ldap_counts[0])

    model = LinearRegression()
    model.fit(X, y)
    ldap_counts['predicted'] = model.predict(X).flatten()

    fig = px.line(ldap_counts, x='date', y='predicted', title="Predicted LDAP Authentication Trend")
    fig.add_scatter(x=ldap_counts['date'], y=ldap_counts[0], mode='lines', name='Actual')
    st.plotly_chart(fig)

    mse = mean_squared_error(y, ldap_counts['predicted'])
    st.write(f"Prediction Mean Squared Error: {mse:.2f}")

# Step 5: Handle questions with Ollama
def handle_question_with_ollama(question, data):
    try:
        # Use Ollama for answering questions based on the uploaded data
        prompt = f"Based on the following data, please answer the question: {question}\n\nData: {data}"
        response = ollama.chat(model="llama2", messages=[{"role": "user", "content": prompt}])
        return response['text']
    except Exception as e:
        return f"Error in querying Ollama: {str(e)}"

# Step 6: Streamlit app logic
def main():
    st.title("Enhanced Login Log Analysis Tool")

    # Sidebar for synthetic log generation
    st.sidebar.header("Generate Synthetic Logs")
    num_logs = st.sidebar.slider("Number of Logs", 10, 1000, 100)
    if st.sidebar.button("Generate Logs"):
        synthetic_logs = generate_synthetic_logs(num_logs)
        st.sidebar.download_button(
            label="Download Synthetic Logs",
            data=synthetic_logs.encode("utf-8"),
            file_name="synthetic_logs.txt"
        )

    # File upload for log analysis
    st.header("Upload Log File")
    uploaded_file = st.file_uploader("Upload a log file", type=["txt", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            try:
                log_data = uploaded_file.read().decode("utf-8")
                # Display raw logs
                st.subheader("Uploaded Log Data")
                st.text_area("Logs", log_data, height=200)

                # Parse logs
                ldap_df, otp_df = parse_logs(log_data)

                # Visualize parsed data
                visualize_data(ldap_df, otp_df)

                # Interactive Q&A with Ollama
                st.subheader("Ask a Question")
                question = st.text_input("Type your question:")
                if question:
                    response = handle_question_with_ollama(question, log_data)
                    st.write(response)
            except Exception as e:
                st.error(f"Error processing the uploaded file: {str(e)}")
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # Read Excel Data
            data = read_excel_data(uploaded_file)
            if data:
                st.subheader("Uploaded Excel Data")
                st.write(data)

                # Example of handling a question about Excel data (you can customize this)
                st.subheader("Ask a Question")
                question = st.text_input("Type your question:")
                if question:
                    response = handle_question_with_ollama(question, str(data))
                    st.write(response)

if __name__ == "__main__":
    main()
