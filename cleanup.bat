@echo off
echo ================================
echo 🧹 Hackathon Cleanup Starting...
echo ================================

REM 1. Stop cloudflared
echo 🔌 Closing cloudflared tunnel...
taskkill /F /IM cloudflared.exe >nul 2>&1

REM 2. Stop FastAPI/Uvicorn server
echo 🧨 Killing FastAPI server (uvicorn)...
taskkill /F /IM uvicorn.exe >nul 2>&1

REM 3. Backup and scrub .env
IF EXIST .env (
    echo 🧼 Scrubbing .env secrets...
    copy .env .env.bak >nul
    powershell -Command "(Get-Content .env) -replace 'OPENROUTER_API_KEY=.*', 'OPENROUTER_API_KEY=REVOKED' | Set-Content .env"
    powershell -Command "(Get-Content .env) -replace 'API_KEY=.*', 'API_KEY=REVOKED' | Set-Content .env"
)

REM 4. Delete transcripts and logs
echo 🧽 Deleting transcript and log files...
del /Q /S *transcript*.json >nul 2>&1
del /Q logs.txt >nul 2>&1

REM 5. Remove __pycache__ folders
echo 🧹 Removing __pycache__ folders...
for /d /r %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo.
echo ✅ Cleanup complete. You're secure! 🔒
echo ================================
pause
