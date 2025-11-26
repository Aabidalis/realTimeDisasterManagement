Real-Time Disaster Monitoring System

A smart real-time disaster monitoring web application built using Flask, OpenCV, and modern Frontend (HTML, CSS, JavaScript). This system detects rising water levels through live camera feed and classifies disaster risk levels (Safe, Warning, Critical) to help prevent flood-related damage.

🚀 Features

✅ Real-time camera feed processing using OpenCV
✅ Water level detection using contour analysis
✅ Dynamic risk classification:

🟢 SAFE

🟡 WARNING

🔴 CRITICAL ✅ Live dashboard with updated status ✅ Visual risk meter and alerts ✅ Responsive UI design ✅ Backend processing with Flask API

🛠️ Technologies Used

Frontend:

HTML5

CSS3

JavaScript

Backend:

Python Flask

OpenCV

NumPy
 
⚙️ Installation & Setup
1️⃣ Clone the repository
 
cd disaster-monitoring-system
2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
3️⃣ Install dependencies
pip install -r backend/requirements.txt
4️⃣ Run the Flask server
cd backend
python app.py

Server will start at:

http://127.0.0.1:5000
5️⃣ Open frontend

Open the file manually:

frontend/index.html

OR host it using Live Server extension in VS Code.

📡 How It Works

Camera captures live video frames

OpenCV analyzes water area using contour detection

System calculates water level percentage

Risk level is assigned based on predefined thresholds

Data is sent to frontend via Flask API

Dashboard updates live in real-time

🎯 Risk Level Logic
Water Level	Risk Status
0% - 30%	SAFE
31% - 70%	WARNING
71% - 100%	CRITICAL
 

🔮 Future Enhancements

Email / SMS alert system

Sound alarm for critical level

Cloud database integration (MongoDB/Firebase)

AI-based predictive analysis

Mobile app version

🧑‍💻 Developer

Your Name: Aabid Ali
Project Type: Academic 
Institution: Sapthagiri NPS University


💬 Contact

For any queries or collaboration:

📧 Email: aabidali4317@gmail.com
