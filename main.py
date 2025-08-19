# main.py
# A FastAPI backend to snap coordinates to the nearest road using GraphHopper Route API.
# Uses Folium for interactive maps.

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import requests
import folium
import polyline
from typing import List

GRAPH_HOPPER_API_KEY = "add4c752-8787-4a6b-ae81-6c8a357504b4"

app = FastAPI()

class Coordinates(BaseModel):
    coordinates: List[List[float]]  

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Road Finder with Folium</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            #map-container {
                height: 600px;
                width: 100%;
                border-radius: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
    </head>
    <body class="bg-gray-100 p-8 flex flex-col items-center min-h-screen">
        <div class="max-w-4xl w-full bg-white p-8 rounded-2xl shadow-xl flex flex-col space-y-8">
            <h1 class="text-4xl font-bold text-center text-gray-800">coordinates approximater</h1>
            <p class="text-center text-gray-600">Enter coordinates to snap them to the nearest road using GraphHopper.</p>

            <div id="loading" class="hidden text-center text-blue-500 font-semibold">
                Finding road...
            </div>

            <div class="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
                <div class="flex-1">
                    <label for="coordinates" class="block text-gray-700 font-semibold mb-2">Coordinates (lat, lon pairs):</label>
                    <textarea id="coordinates" rows="8" class="w-full p-4 rounded-lg border-2 border-gray-300 focus:outline-none focus:border-blue-500 transition-colors duration-200" placeholder="Enter coordinates on separate lines, e.g.:&#10;52.5200, 13.4050"></textarea>
                </div>

                <div class="flex-1 flex flex-col justify-end">
                    <button id="findRoadBtn" class="bg-blue-600 text-white font-bold py-4 rounded-lg shadow-lg hover:bg-blue-700 transition-all duration-300 ease-in-out transform hover:scale-105">
                        Find Road loacation
                    </button>
                    <div id="results" class="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <h3 class="text-lg font-semibold text-gray-700">Results:</h3>
                        <p id="road-coords" class="text-sm text-gray-600 mt-2 whitespace-pre-wrap"></p>
                    </div>
                </div>
            </div>

            <div id="map-container"></div>
        </div>
        
        <script>
            const findRoadBtn = document.getElementById('findRoadBtn');
            const coordinatesInput = document.getElementById('coordinates');
            const roadCoordsResult = document.getElementById('road-coords');
            const loadingIndicator = document.getElementById('loading');
            const mapContainer = document.getElementById('map-container');

            findRoadBtn.addEventListener('click', async () => {
                const coordinatesText = coordinatesInput.value.trim();
                if (!coordinatesText) {
                    showNotification('Please enter coordinates.');
                    return;
                }

                const points = coordinatesText.split('\\n').map(line => {
                    const parts = line.split(',').map(s => parseFloat(s.trim()));
                    return [parts[0], parts[1]];
                }).filter(p => !isNaN(p[0]) && !isNaN(p[1]) && p.length === 2);

                if (points.length === 0) {
                    showNotification('Invalid coordinate format. Please use lat,lon on each line.');
                    return;
                }

                loadingIndicator.classList.remove('hidden');
                findRoadBtn.disabled = true;

                try {
                    const response = await fetch('/generate-map', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ coordinates: points }),
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Unknown error.');
                    }

                    const data = await response.json();
                    
                    if (data.map_html && data.road_coordinates.length > 0) {
                        mapContainer.innerHTML = data.map_html;
                        roadCoordsResult.textContent = data.road_coordinates
                            .map(coord => `[${coord[0].toFixed(6)}, ${coord[1].toFixed(6)}]`)
                            .join('\\n');
                    } else {
                        mapContainer.innerHTML = '';
                        roadCoordsResult.textContent = 'No road found.';
                    }
                } catch (error) {
                    console.error('Error:', error);
                    roadCoordsResult.textContent = `Error: ${error.message}`;
                    showNotification(error.message);
                } finally {
                    loadingIndicator.classList.add('hidden');
                    findRoadBtn.disabled = false;
                }
            });

            function showNotification(message) {
                const notification = document.createElement('div');
                notification.className = 'fixed bottom-4 left-1/2 -translate-x-1/2 bg-red-600 text-white p-4 rounded-lg shadow-lg z-50 transition-all duration-500 ease-in-out transform';
                notification.textContent = message;
                document.body.appendChild(notification);
                setTimeout(() => {
                    notification.classList.add('opacity-0');
                    notification.classList.add('translate-y-full');
                    setTimeout(() => document.body.removeChild(notification), 500);
                }, 5000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/generate-map", response_class=JSONResponse)
async def generate_map_from_coords(data: Coordinates):
    """
    Takes coordinates, snaps them to the nearest road via GraphHopper Route API,
    and returns Folium map + road geometry
    """
    if not GRAPH_HOPPER_API_KEY:
        raise HTTPException(status_code=400, detail="GraphHopper API key not configured.")

    try:
        lat, lon = data.coordinates[0]
        fake_lat, fake_lon = lat + 0.0001, lon + 0.0001

        response = requests.get(
            "https://graphhopper.com/api/1/route",
            params={
                "point": [f"{lat},{lon}", f"{fake_lat},{fake_lon}"],
                "vehicle": "car",
                "locale": "en",
                "points_encoded": "false",   # important: get coordinates, not polyline string
                "key": GRAPH_HOPPER_API_KEY
            },
            timeout=10
        )
        response.raise_for_status()
        graphhopper_data = response.json()

        if "paths" in graphhopper_data and graphhopper_data["paths"]:
            path = graphhopper_data["paths"][0]

            if isinstance(path["points"], dict) and "coordinates" in path["points"]:
                road_coords_list = [[lat, lon] for lon, lat in path["points"]["coordinates"]]
            else:
                road_coords_list = polyline.decode(path["points"])

            m = folium.Map(location=road_coords_list[0], zoom_start=16)
            folium.Marker([lat, lon], icon=folium.Icon(color="red")).add_to(m)
            folium.PolyLine(road_coords_list, color="blue", weight=5, opacity=0.7).add_to(m)

            m.fit_bounds(m.get_bounds())

            return JSONResponse(content={
                "map_html": m._repr_html_(),
                "road_coordinates": road_coords_list
            })
        else:
            return JSONResponse(content={"map_html": "", "road_coordinates": []})

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"GraphHopper API request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
