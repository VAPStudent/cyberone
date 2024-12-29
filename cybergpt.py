import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

# Step 1: Generate synthetic logs
def generate_synthetic_logs(num_logs):
    events = []
    for _ in range(num_logs):
        timestamp = datetime.datetime.now() - datetime.timedelta(seconds=random.randint(0, 100000))
        user_id = f"user{random.randint(1, 100)}"
        event_type = random.choice(["ldap_success", "ldap_failure", "otp_success", "otp_failure"])
        events.append(f"{timestamp} {user_id} {event_type}")
    return "\n".join(events)

# Step 2: Parse logs
def parse_logs(log_data):
    rows = []
    for line in log_data.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            timestamp = " ".join(parts[:2])
            user_id = parts[2]
            event_type = parts[3]
            rows.append((timestamp, user_id, event_type))

    df = pd.DataFrame(rows, columns=["timestamp", "user_id", "event_type"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    ldap_df = df[df["event_type"].str.contains("ldap")]
    otp_df = df[df["event_type"].str.contains("otp")]

    return ldap_df, otp_df

# Step 3: Visualize data
def visualize_data(ldap_df, otp_df):
    st.subheader("LDAP Events Over Time")
    if not ldap_df.empty:
        ldap_df["date"] = ldap_df["timestamp"].dt.date
        ldap_summary = ldap_df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
        st.line_chart(ldap_summary)
    else:
        st.write("No LDAP events to display.")

    st.subheader("OTP Events Over Time")
    if not otp_df.empty:
        otp_df["date"] = otp_df["timestamp"].dt.date
        otp_summary = otp_df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
        st.line_chart(otp_summary)
    else:
        st.write("No OTP events to display.")

# Step 4: Handle questions
def handle_question(question, ldap_df, otp_df):
    if "trend" in question:
        response = "The trend shows how login events have changed over time. Check the charts above for details."
    elif "anomalies" in question:
        response = "Anomalies can be identified by sudden spikes or drops in login activity."
    elif "otp" in question:
        response = "OTP trends help identify how often additional security verification steps are completed successfully, which is vital for secure logins."
    elif "ldap" in question:
        response = "LDAP trends represent the success rate of initial authentication, showing the primary login activity over time."
    else:
        response = "I'm sorry, I couldn't understand the question. Please try asking about trends, anomalies, or predictions."

    return response

# Step 5: Streamlit app logic
def main():
    st.title("Enhanced Login Log Analysis Tool")

    # Sidebar for synthetic log generation
    st.sidebar.header("Generate Synthetic Logs")
    num_logs = st.sidebar.slider("Number of Logs", 10, 1000, 100)
    if st.sidebar.button("Generate Logs"):
        synthetic_logs = generate_synthetic_logs(num_logs)
        st.sidebar.download_button(
            label="Download Synthetic Logs",
            data=synthetic_logs,
            file_name="synthetic_logs.txt"
        )

    # File upload for log analysis
    st.header("Upload Log File")
    uploaded_file = st.file_uploader("Upload a log file", type=["txt"])
    if uploaded_file is not None:
        log_data = uploaded_file.read().decode("utf-8")

        # Display raw logs
        st.subheader("Uploaded Log Data")
        st.text_area("Logs", log_data, height=200)

        # Parse logs
        ldap_df, otp_df = parse_logs(log_data)

        # Visualize parsed data
        visualize_data(ldap_df, otp_df)

        # Interactive Q&A
        st.subheader("Ask a Question")
        question = st.text_input("Type your question:")
        if question:
            response = handle_question(question, ldap_df, otp_df)
            st.write(response)

if __name__ == "__main__":
    main()
