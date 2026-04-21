1️⃣ Clone or Download the Project

They can download the ZIP or use git:

git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2️⃣ Create a Virtual Environment
Windows:
python -m venv .venv

3️⃣ Activate the Virtual Environment
Windows PowerShell:
.\.venv\Scripts\Activate.ps1

If PowerShell blocks it:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

4️⃣ Install Required Libraries
pip install -r requirements.txt

This file should contain:

streamlit
tensorflow
pillow

(I can generate this file for you.)

5️⃣ Run the App

After activating .venv:

streamlit run app.py

This opens the web app in the browser automatically.