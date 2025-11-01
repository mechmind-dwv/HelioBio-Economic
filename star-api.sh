# Iniciar el servidor
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Probar endpoints
curl http://localhost:8000/api/system/health
curl http://localhost:8000/api/solar/current
curl http://localhost:8000/api/economic/indicators
curl "http://localhost:8000/api/correlation/solar-economic?economic_indicator=SP500&solar_indicator=sunspots"
