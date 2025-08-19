# main.py
# A FastAPI backend to find the nearest road from given coordinates using the GraphHopper API.

from fastapi import FastAPI, Request, HTTPException
from fastapi. responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import json
import os
import sys
from typing import List

GRAPH_HOPPER_API_KEY = "add4c752-8787-4a6b-ae81-6c8a357504b4"

app = FastAPI()

# Pydantic model for the incoming coordinate data.
# Each coordinate is a list of [latitude, longitude].
class Coordinates(BaseModel):
    coordinates: List[List[float]]

# Define a root endpoint to serve the HTML file.
# This makes it a self-contained application.
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """
    Serves the main HTML page for the web application.
    """
    # HTML content for the front-end. This is embedded directly for simplicity.
    # In a real-world app, you would serve this from a 'static' directory.
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Road Finder</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                font-family: 'Inter', sans-serif;
            }
            #map {
                height: 600px;
                width: 100%;
                border-radius: 1rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
        </style>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p41847585b/897538a0f28e23f46f381f280a58a9e7f41f02172772719a2872" crossorigin="" />
    </head>
    <body class="bg-gray-100 p-8 flex flex-col items-center min-h-screen">
        <div class="max-w-4xl w-full bg-white p-8 rounded-2xl shadow-xl flex flex-col space-y-8">
            <h1 class="text-4xl font-bold text-center text-gray-800">Road Finder</h1>
            <p class="text-center text-gray-600">Enter multiple coordinates to find the nearest road using GraphHopper.</p>

            <!-- Loading Indicator -->
            <div id="loading" class="hidden text-center text-blue-500 font-semibold">
                Finding road...
            </div>

            <div class="flex flex-col md:flex-row space-y-4 md:space-y-0 md:space-x-4">
                <div class="flex-1">
                    <label for="coordinates" class="block text-gray-700 font-semibold mb-2">Coordinates (lat, lon pairs):</label>
                    <textarea id="coordinates" rows="8" class="w-full p-4 rounded-lg border-2 border-gray-300 focus:outline-none focus:border-blue-500 transition-colors duration-200" placeholder="Enter coordinates on separate lines, e.g.:&#10;52.5200, 13.4050&#10;52.5201, 13.4051"></textarea>
                </div>

                <div class="flex-1 flex flex-col justify-end">
                    <button id="findRoadBtn" class="bg-blue-600 text-white font-bold py-4 rounded-lg shadow-lg hover:bg-blue-700 transition-all duration-300 ease-in-out transform hover:scale-105">
                        Find Nearest Road
                    </button>
                    <div id="results" class="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                        <h3 class="text-lg font-semibold text-gray-700">Results:</h3>
                        <p id="road-coords" class="text-sm text-gray-600 mt-2 whitespace-pre-wrap"></p>
                    </div>
                </div>
            </div>

            <div id="map"></div>
        </div>
        
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-2B1u8J/903e49339e083c74f5351a99a77f998492c30070a7863d047306236b208c1a" crossorigin=""></script>
        
        <script>
            // Initialize the map and set its view
            const map = L.map('map').setView([52.5200, 13.4050], 13); // Default to Berlin

            // Add a tile layer (OpenStreetMap)
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            }).addTo(map);

            let inputMarkers = [];
            let roadLine = null;

            const findRoadBtn = document.getElementById('findRoadBtn');
            const coordinatesInput = document.getElementById('coordinates');
            const roadCoordsResult = document.getElementById('road-coords');
            const loadingIndicator = document.getElementById('loading');

            findRoadBtn.addEventListener('click', async () => {
                // Clear previous markers and road line
                inputMarkers.forEach(marker => map.removeLayer(marker));
                if (roadLine) {
                    map.removeLayer(roadLine);
                }

                const coordinatesText = coordinatesInput.value.trim();
                if (!coordinatesText) {
                    alert('Please enter coordinates.');
                    return;
                }

                // Parse the coordinates from the textarea
                const points = coordinatesText.split('\\n').map(line => {
                    const parts = line.split(',').map(s => parseFloat(s.trim()));
                    return [parts[0], parts[1]];
                }).filter(p => !isNaN(p[0]) && !isNaN(p[1]));

                if (points.length === 0) {
                    alert('Invalid coordinate format.');
                    return;
                }

                // Add markers for the input points
                points.forEach(point => {
                    const marker = L.marker(point).addTo(map);
                    inputMarkers.push(marker);
                });

                // Set map view to fit all points
                const bounds = new L.LatLngBounds(points);
                map.fitBounds(bounds, { padding: [50, 50] });

                loadingIndicator.classList.remove('hidden');
                findRoadBtn.disabled = true;

                try {
                    const response = await fetch('/find-road', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ coordinates: points }),
                    });

                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'An unknown error occurred.');
                    }

                    const data = await response.json();

                    if (data.road_coordinates && data.road_coordinates.length > 0) {
                        // Draw the road on the map
                        const roadCoords = data.road_coordinates;
                        roadLine = L.polyline(roadCoords, { color: 'blue', weight: 5, opacity: 0.7 }).addTo(map);
                        map.fitBounds(roadLine.getBounds(), { padding: [50, 50] });
                        
                        // Display the road coordinates in the results box
                        const formattedCoords = roadCoords.map(coord => `[${coord[0].toFixed(6)}, ${coord[1].toFixed(6)}]`).join('\\n');
                        roadCoordsResult.textContent = formattedCoords;

                    } else {
                        roadCoordsResult.textContent = 'No road found for the given coordinates.';
                    }

                } catch (error) {
                    console.error('Error:', error);
                    roadCoordsResult.textContent = `Error: ${error.message}`;
                    // Use a temporary modal or alert-like UI element instead of browser alert()
                    const errorMessage = `An error occurred: ${error.message}. Please check your GraphHopper API key and coordinate format.`;
                    showNotification(errorMessage);
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

@app.post("/find-road", response_class=JSONResponse)
async def find_road_from_coords(data: Coordinates):
    """
    Takes a list of coordinates, makes a request to the GraphHopper API
    to find the nearest road, and returns the result.
    """
    if not GRAPH_HOPPER_API_KEY or GRAPH_HOPPER_API_KEY == "YOUR_GRAPH_HOPPER_API_KEY_HERE":
        raise HTTPException(
            status_code=400,
            detail="GraphHopper API key not configured. Please add your key to main.py."
        )

    # Convert the list of [lat, lon] to GraphHopper's required format (lon, lat)
    # The 'route' matching endpoint expects [lon, lat]
    points_for_api = [[lon, lat] for lat, lon in data.coordinates]

    # GraphHopper's 'match' endpoint is perfect for this. It finds the best
    # route that snaps to the road network.
    graphhopper_url = f"https://graphhopper.com/api/1/match?key={GRAPH_HOPPER_API_KEY}"

    try:
        # Construct the request payload for the GraphHopper API
        payload = {
            "points": points_for_api,
            "vehicle": "car",  # Or 'bike', 'foot' etc.
            "locale": "en",
            "points_encoded": False # We are sending raw coordinates
        }

        # Make the POST request to the GraphHopper API
        response = requests.post(graphhopper_url, json=payload, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes

        graphhopper_data = response.json()

        if "paths" in graphhopper_data and graphhopper_data["paths"]:
            # Extract the points of the matched path (the road).
            # The 'points' field contains the geometry of the road.
            path_points = graphhopper_data["paths"][0]["points"]["coordinates"]
            
            # GraphHopper returns coordinates as [lon, lat]. Convert them back to [lat, lon]
            road_coordinates = [[lat, lon] for lon, lat in path_points]
            
            return JSONResponse(content={"road_coordinates": road_coordinates})
        else:
            return JSONResponse(content={"road_coordinates": []})

    except requests.exceptions.HTTPError as e:
        # Handle HTTP errors from the GraphHopper API
        print(f"HTTP Error: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"GraphHopper API error: {e.response.text}"
        )
    except requests.exceptions.RequestException as e:
        # Handle other request errors (e.g., connection issues)
        print(f"Request Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while connecting to the GraphHopper API."
        )
    except Exception as e:
        # Catch any other unexpected errors
        print(f"Unexpected Error: {e}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred on the server."
        )
