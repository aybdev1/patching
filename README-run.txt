AI Patch Manager - run instructions (Windows)

1) Install Python 3.10+ on Windows.

2) Create a virtual environment:
   Open PowerShell (as admin recommended) and run:
   cd C:\ai-patch-manager\backend
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

3) Install requirements:
   pip install -r requirements.txt

4) Initialize the database:
   python db_init.py

5) (Optional) Train toy model:
   python model_train.py   # creates prioritizer.joblib which you can integrate later

6) Run the backend:
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

7) Access the UI:
   Open http://localhost:8000 in your browser.

8) Run scan on a Windows host (same machine for demo):
   - Open PowerShell as Administrator
   - cd C:\ai-patch-manager\scripts
   - ./scan_updates.ps1 | Out-File -FilePath C:\temp\scan.json
   OR directly post:
   ./scan_updates.ps1 -OutputFile C:\temp\scan.json
   Then POST the JSON to backend:
   Invoke-RestMethod -Uri "http://localhost:8000/api/scan" -Method Post -Body (Get-Content C:\temp\scan.json -Raw) -ContentType "application/json"

9) View patches in UI: enter the hostname and click Fetch.

10) To install a KB, follow the instruction returned by the UI (it will provide a powershell command to run as Admin on the host).

Security & admin notes:
- Installing updates requires Administrator rights.
- For remote install automation, configure WinRM / PSRemoting securely (HTTPS, certificates, or Kerberos).
- Do not run remote code from untrusted networks without proper authentication/authorization.
