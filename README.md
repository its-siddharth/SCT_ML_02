Customer Segmentation Web App
=============================

A visually engaging, modern Streamlit-based web application that predicts customer segments using K-Means Clustering. This tool is ideal for marketing teams, business analysts, and students interested in understanding consumer behavior based on demographic and spending data.

-------------------------------
🔧 Features
-------------------------------
- Modern, colorful UI with gradient background and bold fonts
- Predicts customer segments using trained K-Means model
- Shows distance from each cluster center and confidence score
- Responsive, clean layout (optimized for desktops and laptops)
- Streamlit-powered for fast deployment and interactivity

-------------------------------
📁 Project Structure
-------------------------------
CustomerSegmentationApp/
│
├── app.py                        # Main Streamlit app
├── customer_segmentation_model.pkl  # Trained model file (includes scaler & metadata)
├── README.txt                    # Project description (this file)
└── requirements.txt              # Required packages list

-------------------------------
🧠 Input Features
-------------------------------
1. Gender           - Male or Female
2. Age              - Between 18 to 80
3. Annual Income    - In thousands of USD (10K to 150K)
4. Spending Score   - Scale from 1 to 100

-------------------------------
🧰 Technologies Used
-------------------------------
- Python 3.8+
- Streamlit (UI Framework)
- scikit-learn (ML Model)
- numpy (Numerical processing)
- joblib (Model saving/loading)

-------------------------------
💻 How to Run
-------------------------------
1. Clone the Repository:
   

2. Install Dependencies:
   pip install -r requirements.txt

3. Launch the App:
   streamlit run app.py

-------------------------------
🧠 How It Works
-------------------------------
- A pre-trained K-Means model is loaded.
- User inputs their gender, age, income, and spending score.
- The app uses a scaler to normalize the data before prediction.
- The cluster label and cluster distances are shown with a confidence score.
