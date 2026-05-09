@echo off
setlocal

set PORT=%1
if "%PORT%"=="" set PORT=8502
set URL=http://localhost:%PORT%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$port = %PORT%; " ^
  "$client = New-Object Net.Sockets.TcpClient; " ^
  "try { $client.Connect('127.0.0.1', $port); $busy = $client.Connected } catch { $busy = $false } finally { $client.Dispose() }; " ^
  "if ($busy) { " ^
  "  Write-Host ''; " ^
  "  Write-Host 'Port %PORT% is already in use.' -ForegroundColor Yellow; " ^
  "  Write-Host 'Opening the existing app or service at:' -ForegroundColor White; " ^
  "  Write-Host '%URL%' -ForegroundColor Green; " ^
  "  Write-Host 'If this is an old dashboard session, stop it from the terminal where it is running.' -ForegroundColor Magenta; " ^
  "  Write-Host ''; " ^
  "  exit 0 " ^
  "} else { exit 1 }"

if "%ERRORLEVEL%"=="0" (
    start "" "%URL%"
    endlocal
    exit /b 0
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Write-Host ''; " ^
  "Write-Host 'SOCO Model Results Dashboard is starting...' -ForegroundColor Cyan; " ^
  "Write-Host 'Open the app in your browser:' -ForegroundColor White; " ^
  "Write-Host 'Local URL: %URL%' -ForegroundColor Green; " ^
  "Write-Host 'Opening the browser automatically now...' -ForegroundColor Yellow; " ^
  "Write-Host 'If the browser does not open automatically, copy/paste the URL above.' -ForegroundColor DarkGray; " ^
  "Write-Host 'To close the app, return to this terminal and press Ctrl+C.' -ForegroundColor Magenta; " ^
  "Write-Host 'If Windows asks Terminate batch job (Y/N)?, type Y and press Enter.' -ForegroundColor Magenta; " ^
  "Write-Host ''"

start "" "%URL%"
conda run -n energy_demand_ml_env001 streamlit run app/model_results_dashboard.py --server.port %PORT%

endlocal
