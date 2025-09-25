import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List
import urllib.parse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, initial_state):
        self.state_estimate = np.array(initial_state, dtype=np.float64)
        self.P = np.eye(4) * 1000
        self.Q = np.eye(4) * process_variance
        self.R = np.eye(2) * measurement_variance
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self.state_estimate = F @ self.state_estimate
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state_estimate = self.state_estimate + K @ (measurement - self.H @ self.state_estimate)
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P

    def get_location(self):
        return self.state_estimate[0], self.state_estimate[1]

class Measurement(BaseModel):
    latitude: float
    longitude: float
    timestamp: int

class MeasurementsData(BaseModel):
    measurements: List[Measurement] = Field(..., min_items=2)

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Kalman Filter Tracker</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            #map { height: 400px; width: 100%; }
            .leaflet-container { border-radius: 0.5rem; }
            .custom-marker { background: transparent; border: none; }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
        <div class="bg-white rounded-xl shadow-lg p-8 w-full max-w-4xl">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-2">Kalman Filter Location Tracker</h1>
            <p class="text-center text-gray-500 mb-6">Enter a series of location measurements to see the filter's estimation and plot it on a map.</p>
            
            <div class="flex items-end space-x-4 mb-4">
                <div class="flex-1">
                    <label for="latitude" class="block text-sm font-medium text-gray-700">Latitude</label>
                    <input type="number" id="latitude" placeholder="e.g., 28.560833" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 p-2 border">
                </div>
                <div class="flex-1">
                    <label for="longitude" class="block text-sm font-medium text-gray-700">Longitude</label>
                    <input type="number" id="longitude" placeholder="e.g., 77.367222" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 p-2 border">
                </div>
                <div class="flex-1">
                    <label for="timestamp" class="block text-sm font-medium text-gray-700">Timestamp (ms)</label>
                    <input type="number" id="timestamp" placeholder="e.g., 1721721000000" step="1" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50 p-2 border">
                </div>
                <button id="addMeasurementBtn" class="bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300">
                    Add
                </button>
            </div>
            
            <div id="measurementsList" class="mb-4 bg-gray-50 rounded-lg p-4 max-h-40 overflow-y-auto">
                <p class="text-sm text-gray-500 text-center">No measurements added yet.</p>
            </div>
            
            <button id="calculateBtn" class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300">
                Calculate & Plot
            </button>
            
            <div id="resultsSection" class="mt-8">
                <h2 class="text-xl font-bold text-gray-800 mb-2">Final Estimated Location</h2>
                <div class="bg-gray-50 p-4 rounded-lg shadow-inner">
                    <p class="text-gray-700 font-medium text-lg mb-2">
                        Final Coordinates: <span id="finalCoords">-</span>
                    </p>
                    <p class="text-gray-500 italic">
                        Address: <span id="finalAddress">Add measurements and click Calculate to see results</span>
                    </p>
                </div>
                
                <h2 class="text-xl font-bold text-gray-800 mt-6 mb-2">Location Map</h2>
                <div id="mapContainer" class="w-full rounded-lg shadow-inner">
                    <div id="map"></div>
                    <div id="mapLoading" class="text-gray-500 text-center py-20 bg-gray-100 rounded-lg">
                        Map is loading... Add measurements and click Calculate to see plotted points.
                    </div>
                </div>
                
                <div class="mt-4 bg-blue-50 p-3 rounded-lg">
                    <p class="text-sm text-blue-700">
                        <span class="font-semibold">Map Legend:</span><br>
                        <span style="color: red">● Red markers</span> - Input measurements (numbered)<br>
                        <span style="color: green">● Green marker</span> - Final estimated location<br>
                        <span style="color: gray">━ Gray line</span> - Path between input measurements<br>
                        <span style="color: blue">━ Blue dashed line</span> - Connection to final estimate
                    </p>
                </div>
            </div>
            
            <div id="alertModal" class="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto h-full w-full hidden z-50">
                <div class="relative top-20 mx-auto p-5 border w-96 shadow-lg rounded-md bg-white">
                    <div class="mt-3 text-center">
                        <h3 class="text-lg leading-6 font-medium text-gray-900" id="alertTitle"></h3>
                        <div class="mt-2 px-7 py-3">
                            <p class="text-sm text-gray-500" id="alertMessage"></p>
                        </div>
                        <div class="items-center px-4 py-3">
                            <button id="alertCloseBtn" class="px-4 py-2 bg-blue-500 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-300">
                                OK
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const addMeasurementBtn = document.getElementById('addMeasurementBtn');
            const calculateBtn = document.getElementById('calculateBtn');
            const latitudeInput = document.getElementById('latitude');
            const longitudeInput = document.getElementById('longitude');
            const timestampInput = document.getElementById('timestamp');
            const measurementsList = document.getElementById('measurementsList');
            const resultsSection = document.getElementById('resultsSection');
            const finalCoordsSpan = document.getElementById('finalCoords');
            const finalAddressSpan = document.getElementById('finalAddress');
            const mapContainer = document.getElementById('map');
            const mapLoading = document.getElementById('mapLoading');
            const alertModal = document.getElementById('alertModal');
            const alertTitle = document.getElementById('alertTitle');
            const alertMessage = document.getElementById('alertMessage');
            const alertCloseBtn = document.getElementById('alertCloseBtn');
            
            let measurements = [];
            let map = null;
            let markers = [];
            let polylines = [];

            // Initialize map on page load with a default location (New Delhi)
            function initializeMap() {
                // Default center (New Delhi)
                const defaultCenter = [28.6139, 77.2090];
                
                map = L.map('map').setView(defaultCenter, 10);
                
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 18
                }).addTo(map);
                
                // Add a default marker to show the map is working
                addMarker(defaultCenter[0], defaultCenter[1], 'blue', '?', 'Default location: New Delhi<br>Add measurements to see actual data');
                
                mapLoading.style.display = 'none';
                mapContainer.style.display = 'block';
            }

            // Initialize map when page loads
            document.addEventListener('DOMContentLoaded', function() {
                initializeMap();
            });

            function showAlert(title, message) {
                alertTitle.textContent = title;
                alertMessage.textContent = message;
                alertModal.classList.remove('hidden');
            }

            alertCloseBtn.addEventListener('click', () => {
                alertModal.classList.add('hidden');
            });

            addMeasurementBtn.addEventListener('click', () => {
                const latitude = parseFloat(latitudeInput.value);
                const longitude = parseFloat(longitudeInput.value);
                const timestamp = parseInt(timestampInput.value);
                
                if (isNaN(latitude) || isNaN(longitude) || isNaN(timestamp)) {
                    showAlert("Invalid Input", "Please enter valid numbers for all fields.");
                    return;
                }
                
                if (latitude < -90 || latitude > 90) {
                    showAlert("Invalid Latitude", "Latitude must be between -90 and 90 degrees.");
                    return;
                }
                
                if (longitude < -180 || longitude > 180) {
                    showAlert("Invalid Longitude", "Longitude must be between -180 and 180 degrees.");
                    return;
                }
                
                measurements.push({ latitude, longitude, timestamp });
                
                if (measurementsList.firstElementChild.tagName === 'P') {
                    measurementsList.innerHTML = '';
                }
                
                const listItem = document.createElement('div');
                listItem.className = "text-gray-700 text-sm p-2 bg-white rounded-md mb-2 shadow-sm";
                listItem.textContent = `Measurement ${measurements.length}: Lat: ${latitude.toFixed(6)}, Lon: ${longitude.toFixed(6)}, Time: ${timestamp}`;
                measurementsList.appendChild(listItem);
                
                latitudeInput.value = '';
                longitudeInput.value = '';
                timestampInput.value = '';
                
                // If this is the first measurement, update the map center
                if (measurements.length === 1) {
                    map.setView([latitude, longitude], 13);
                    clearMap();
                }
            });

            function clearMap() {
                markers.forEach(marker => {
                    if (map.hasLayer(marker)) {
                        map.removeLayer(marker);
                    }
                });
                polylines.forEach(polyline => {
                    if (map.hasLayer(polyline)) {
                        map.removeLayer(polyline);
                    }
                });
                markers = [];
                polylines = [];
            }

            function addMarker(lat, lng, color, label, popupText) {
                const icon = L.divIcon({
                    html: `<div style="background-color: ${color}; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">${label}</div>`,
                    className: 'custom-marker',
                    iconSize: [30, 30],
                    iconAnchor: [15, 15]
                });
                
                const marker = L.marker([lat, lng], { icon: icon })
                    .addTo(map)
                    .bindPopup(popupText);
                
                markers.push(marker);
                return marker;
            }

            function addPolyline(points, color, weight, dashArray = null) {
                const polyline = L.polyline(points, {
                    color: color,
                    weight: weight,
                    opacity: 0.8,
                    dashArray: dashArray
                }).addTo(map);
                
                polylines.push(polyline);
                return polyline;
            }

            function plotPointsOnMap(measurements, finalCoords, finalAddress) {
                clearMap();
                
                // Add input measurement markers and polyline
                const inputPoints = [];
                measurements.forEach((measurement, index) => {
                    const point = [measurement.latitude, measurement.longitude];
                    inputPoints.push(point);
                    addMarker(
                        measurement.latitude, 
                        measurement.longitude, 
                        'red', 
                        index + 1,
                        `Measurement ${index + 1}<br>Lat: ${measurement.latitude.toFixed(6)}<br>Lon: ${measurement.longitude.toFixed(6)}<br>Time: ${measurement.timestamp}`
                    );
                });
                
                // Add polyline connecting input measurements
                if (inputPoints.length > 1) {
                    addPolyline(inputPoints, 'gray', 5);
                }
                
                // Add final estimated location marker
                const finalPoint = [finalCoords[1], finalCoords[0]];
                addMarker(
                    finalCoords[1], 
                    finalCoords[0], 
                    'green', 
                    'F',
                    `Final Estimate<br>Lat: ${finalCoords[1].toFixed(6)}<br>Lon: ${finalCoords[0].toFixed(6)}<br>${finalAddress}`
                );
                
                // Add dashed line from last measurement to final estimate
                if (inputPoints.length > 0) {
                    const lastPoint = inputPoints[inputPoints.length - 1];
                    addPolyline([lastPoint, finalPoint], 'blue', 3, '5, 10');
                }
                
                // Fit map to show all points with padding
                const allPoints = [...inputPoints, finalPoint];
                const group = new L.featureGroup(markers.concat(polylines));
                map.fitBounds(group.getBounds().pad(0.1));
                
                // Update results display
                finalCoordsSpan.textContent = `(${finalCoords[0].toFixed(6)}, ${finalCoords[1].toFixed(6)})`;
                finalAddressSpan.textContent = finalAddress;
            }

            calculateBtn.addEventListener('click', async () => {
                if (measurements.length < 2) {
                    showAlert("Not Enough Data", "Please add at least two measurements to run the filter.");
                    return;
                }
                
                finalCoordsSpan.textContent = 'Calculating...';
                finalAddressSpan.textContent = 'Calculating address...';
                mapLoading.style.display = 'block';
                mapContainer.style.display = 'none';
                
                try {
                    const response = await fetch('/calculate', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ measurements })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`Server error: ${response.status} - ${errorText}`);
                    }
                    
                    const data = await response.json();
                    const finalCoords = data.final_estimated_location;
                    const finalAddress = data.final_address;

                    // Plot all points on the map
                    plotPointsOnMap(measurements, finalCoords, finalAddress);
                    
                    mapLoading.style.display = 'none';
                    mapContainer.style.display = 'block';
                    
                } catch (error) {
                    showAlert("Calculation Error", error.message);
                    finalCoordsSpan.textContent = 'Error';
                    finalAddressSpan.textContent = 'Error';
                    mapLoading.style.display = 'block';
                    mapContainer.style.display = 'none';
                }
            });

            // Handle Enter key in input fields
            [latitudeInput, longitudeInput, timestampInput].forEach(input => {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        addMeasurementBtn.click();
                    }
                });
            });

            // Add sample data button for testing
            const sampleDataBtn = document.createElement('button');
            sampleDataBtn.textContent = 'Load Sample Data';
            sampleDataBtn.className = 'w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300 mt-2';
            sampleDataBtn.onclick = loadSampleData;
            calculateBtn.parentNode.insertBefore(sampleDataBtn, calculateBtn.nextSibling);

            function loadSampleData() {
                // Clear existing measurements
                measurements = [];
                measurementsList.innerHTML = '<p class="text-sm text-gray-500 text-center">No measurements added yet.</p>';
                
                // Sample data around New Delhi
                const sampleMeasurements = [
                    { latitude: 28.6139, longitude: 77.2090, timestamp: 1721721000000 },
                    { latitude: 28.6200, longitude: 77.2150, timestamp: 1721721001000 },
                    { latitude: 28.6250, longitude: 77.2200, timestamp: 1721721002000 },
                    { latitude: 28.6300, longitude: 77.2250, timestamp: 1721721003000 },
                    { latitude: 28.6350, longitude: 77.2300, timestamp: 1721721004000 }
                ];
                
                // Add sample measurements
                sampleMeasurements.forEach(measurement => {
                    measurements.push(measurement);
                    
                    if (measurementsList.firstElementChild.tagName === 'P') {
                        measurementsList.innerHTML = '';
                    }
                    
                    const listItem = document.createElement('div');
                    listItem.className = "text-gray-700 text-sm p-2 bg-white rounded-md mb-2 shadow-sm";
                    listItem.textContent = `Measurement ${measurements.length}: Lat: ${measurement.latitude.toFixed(6)}, Lon: ${measurement.longitude.toFixed(6)}, Time: ${measurement.timestamp}`;
                    measurementsList.appendChild(listItem);
                });
                
                // Auto-calculate after loading sample data
                setTimeout(() => {
                    calculateBtn.click();
                }, 500);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/calculate")
async def calculate_location(data: MeasurementsData):
    try:
        measurements = data.measurements
        
        # Sort measurements by timestamp to ensure proper order
        measurements.sort(key=lambda x: x.timestamp)
        
        initial_lat = measurements[0].latitude
        initial_lon = measurements[0].longitude
        initial_state = np.array([initial_lon, initial_lat, 0, 0], dtype=np.float64)
        
        kf = KalmanFilter(process_variance=0.1, measurement_variance=10.0, initial_state=initial_state)
        last_timestamp = measurements[0].timestamp / 1000.0
        
        for i in range(1, len(measurements)):
            current_time = measurements[i].timestamp / 1000.0
            dt = current_time - last_timestamp
            
            if dt > 0:
                kf.predict(dt)
            
            measured_location = np.array([measurements[i].longitude, measurements[i].latitude], dtype=np.float64)
            kf.update(measured_location)
            last_timestamp = current_time
        
        final_estimated_location = kf.get_location()
        lat = float(final_estimated_location[1])
        lon = float(final_estimated_location[0])
        
        # Get address
        final_address = await get_address(lat, lon)
        
        return {
            "final_estimated_location": [lon, lat],
            "final_address": final_address
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

async def get_address(lat: float, lon: float) -> str:
    """Get address from coordinates using Nominatim"""
    try:
        geocode_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=18"
        headers = {
            'User-Agent': 'Kalman-Filter-Tracker/1.0'
        }
        
        response = requests.get(geocode_url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        address = data.get('display_name', 'Address not found')
        return address
        
    except Exception as e:
        return f"Address lookup failed: {str(e)}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)