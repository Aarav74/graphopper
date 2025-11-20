import numpy as np
import math
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize
import json
from datetime import datetime
import logging
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Location System", version="1.0.0")

class CarLocationSystem:
    def __init__(self):
        self.initialize_database()
        self.accuracy_history = []
        
    def initialize_database(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect('car_location_system.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_lat REAL NOT NULL,
                car_lon REAL NOT NULL,
                calculated_lat REAL NOT NULL,
                calculated_lon REAL NOT NULL,
                error_meters REAL NOT NULL,
                accuracy_grade TEXT NOT NULL,
                calculation_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.commit()
    
    def calculate_tower_position(self, car_lat: float, car_lon: float, 
                               distance: float, azimuth: float) -> Tuple[float, float]:
        """
        Calculate tower position based on car location and measurements
        Using destination point given distance and bearing formula
        """
        # Earth's radius in meters
        R = 6371000.0
        
        # Convert to radians
        car_lat_rad = math.radians(car_lat)
        car_lon_rad = math.radians(car_lon)
        azimuth_rad = math.radians(azimuth)
        
        # Calculate tower latitude
        tower_lat_rad = math.asin(
            math.sin(car_lat_rad) * math.cos(distance / R) +
            math.cos(car_lat_rad) * math.sin(distance / R) * math.cos(azimuth_rad)
        )
        
        # Calculate tower longitude
        tower_lon_rad = car_lon_rad + math.atan2(
            math.sin(azimuth_rad) * math.sin(distance / R) * math.cos(car_lat_rad),
            math.cos(distance / R) - math.sin(car_lat_rad) * math.sin(tower_lat_rad)
        )
        
        # Convert back to degrees
        tower_lat = math.degrees(tower_lat_rad)
        tower_lon = math.degrees(tower_lon_rad)
        
        return float(tower_lat), float(tower_lon)
    
    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates"""
        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return 6371000 * c
    
    def calculate_car_from_towers(self, tower_data: List[Dict]) -> Dict:
        """
        Calculate car position from multiple tower positions
        Using trilateration with optimization
        """
        try:
            tower_positions = []
            distances = []
            
            for tower in tower_data:
                tower_positions.append((tower['latitude'], tower['longitude']))
                distances.append(tower['distance'])
            
            if len(tower_positions) < 3:
                raise ValueError("Need at least 3 towers for accurate calculation")
            
            # Initial guess - centroid of towers weighted by distance
            weights = [1.0 / d for d in distances]
            total_weight = sum(weights)
            
            initial_lat = sum(pos[0] * weight for pos, weight in zip(tower_positions, weights)) / total_weight
            initial_lon = sum(pos[1] * weight for pos, weight in zip(tower_positions, weights)) / total_weight
            
            def objective_function(point):
                lat, lon = point
                total_error = 0.0
                for i, (tower_pos, measured_dist) in enumerate(zip(tower_positions, distances)):
                    calculated_dist = self.haversine_distance((lat, lon), tower_pos)
                    error = (calculated_dist - measured_dist) ** 2
                    total_error += error
                return total_error
            
            # Optimize car position
            bounds = [
                (initial_lat - 0.005, initial_lat + 0.005),  # ~550m bounds
                (initial_lon - 0.005, initial_lon + 0.005)
            ]
            
            result = minimize(
                objective_function,
                [initial_lat, initial_lon],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                calculated_lat, calculated_lon = float(result.x[0]), float(result.x[1])
            else:
                calculated_lat, calculated_lon = initial_lat, initial_lon
            
            # Calculate errors for each tower
            tower_errors = []
            for i, (tower_pos, measured_dist) in enumerate(zip(tower_positions, distances)):
                calculated_dist = self.haversine_distance((calculated_lat, calculated_lon), tower_pos)
                error = calculated_dist - measured_dist
                tower_errors.append({
                    'tower_id': f"Tower_{i+1}",
                    'measured_distance': float(measured_dist),
                    'calculated_distance': float(calculated_dist),
                    'error_m': float(error)
                })
            
            return {
                'calculated_location': {
                    'latitude': calculated_lat,
                    'longitude': calculated_lon
                },
                'tower_errors': tower_errors,
                'optimization_method': getattr(result, 'method', 'Initial Guess')
            }
            
        except Exception as e:
            logger.error(f"Car calculation from towers failed: {e}")
            raise ValueError(f"Calculation error: {str(e)}")
    
    async def calculate_complete_analysis(self, car_lat: float, car_lon: float, 
                                        tower_measurements: List[Dict]) -> Dict:
        """
        Complete analysis:
        1. Calculate tower positions from car location and measurements
        2. Calculate car position back from these towers
        3. Compare with original car position
        """
        try:
            # Step 1: Calculate tower positions from car location
            calculated_towers = []
            for i, measurement in enumerate(tower_measurements):
                tower_lat, tower_lon = self.calculate_tower_position(
                    float(car_lat),
                    float(car_lon),
                    float(measurement['distance']),
                    float(measurement['azimuth'])
                )
                
                calculated_towers.append({
                    'tower_id': f"Tower_{i+1}",
                    'latitude': tower_lat,
                    'longitude': tower_lon,
                    'distance': float(measurement['distance']),
                    'azimuth': int(measurement['azimuth']),
                    'signal_strength': float(measurement.get('signal_strength', -85))
                })
            
            # Step 2: Calculate car position from these towers
            car_calculation_result = self.calculate_car_from_towers(calculated_towers)
            calculated_car_lat = car_calculation_result['calculated_location']['latitude']
            calculated_car_lon = car_calculation_result['calculated_location']['longitude']
            
            # Step 3: Calculate error
            distance_error = float(self.haversine_distance(
                (float(car_lat), float(car_lon)), 
                (calculated_car_lat, calculated_car_lon)
            ))
            
            # Store in database
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO calculations 
                (car_lat, car_lon, calculated_lat, calculated_lon, error_meters, accuracy_grade)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                float(car_lat), float(car_lon), calculated_car_lat, calculated_car_lon, 
                distance_error, self.get_accuracy_grade(distance_error)
            ))
            self.conn.commit()
            
            # Prepare results
            analysis_result = {
                'actual_car_location': {
                    'latitude': float(car_lat),
                    'longitude': float(car_lon),
                    'description': 'Actual Car Position'
                },
                'calculated_car_location': {
                    'latitude': calculated_car_lat,
                    'longitude': calculated_car_lon
                },
                'calculated_towers': calculated_towers,
                'accuracy_metrics': {
                    'distance_error_meters': distance_error,
                    'accuracy_grade': self.get_accuracy_grade(distance_error),
                    'target_achieved': distance_error <= 50.0
                },
                'calculation_details': {
                    'towers_used': len(tower_measurements),
                    'optimization_method': car_calculation_result['optimization_method'],
                    'calculation_id': int(cursor.lastrowid) if cursor.lastrowid else 0
                },
                'tower_analysis': car_calculation_result['tower_errors']
            }
            
            # Store in history
            self.accuracy_history.append({
                'timestamp': datetime.now().isoformat(),
                'error_meters': distance_error,
                'accuracy_grade': self.get_accuracy_grade(distance_error)
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Complete analysis failed: {e}")
            raise ValueError(f"Analysis error: {str(e)}")
    
    def get_accuracy_grade(self, error_meters: float) -> str:
        """Accuracy classification with 50m target"""
        error_meters = float(error_meters)
        if error_meters <= 10:
            return "Excellent 🎯"
        elif error_meters <= 25:
            return "Very Good ✅"
        elif error_meters <= 50:
            return "Good 👍 (Target Achieved)"
        elif error_meters <= 100:
            return "Marginal ⚠️"
        else:
            return "Unacceptable 🚨"

# FastAPI Models
class CarAnalysisRequest(BaseModel):
    car_lat: float
    car_lon: float
    tower_measurements: List[Dict]

# Initialize system
car_system = CarLocationSystem()

@app.get("/")
async def serve_car_analysis_interface():
    """Serve the car analysis interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Car Location Analysis System</title>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            :root {
                --primary-color: #2c3e50;
                --success-color: #27ae60;
                --warning-color: #f39c12;
                --danger-color: #e74c3c;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }
            
            .container {
                max-width: 100%;
                margin: 0 auto;
                padding: 20px;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                padding: 25px;
                border-radius: 15px;
                margin-bottom: 20px;
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }
            
            .header h1 {
                color: var(--primary-color);
                margin-bottom: 10px;
                font-size: 2em;
            }
            
            .target-badge {
                background: linear-gradient(135deg, #27ae60, #2ecc71);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
            }
            
            .main-grid {
                display: grid;
                grid-template-columns: 450px 1fr;
                gap: 20px;
                height: calc(100vh - 180px);
            }
            
            .control-panel {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                overflow-y: auto;
            }
            
            .maps-container {
                display: grid;
                grid-template-rows: 1fr 1fr;
                gap: 15px;
            }
            
            .map-section {
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                position: relative;
            }
            
            .map-title {
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255, 255, 255, 0.95);
                padding: 8px 15px;
                border-radius: 20px;
                font-weight: bold;
                z-index: 1000;
                font-size: 0.9em;
            }
            
            .map {
                height: 100%;
                width: 100%;
            }
            
            .card {
                background: white;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 15px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.08);
            }
            
            .card h3 {
                color: var(--primary-color);
                margin-bottom: 12px;
                font-size: 1.1em;
            }
            
            .btn {
                background: linear-gradient(135deg, var(--primary-color), #3498db);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                width: 100%;
                font-size: 14px;
                font-weight: bold;
                margin: 8px 0;
            }
            
            .btn:hover {
                transform: translateY(-1px);
            }
            
            .input-group {
                margin-bottom: 12px;
            }
            
            .input-group label {
                display: block;
                margin-bottom: 4px;
                font-weight: 600;
                color: var(--primary-color);
                font-size: 0.85em;
            }
            
            .input-group input {
                width: 100%;
                padding: 8px 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
            }
            
            .tower-input-group {
                background: #f8f9fa;
                padding: 12px;
                border-radius: 8px;
                margin-bottom: 12px;
            }
            
            .tower-input-group h4 {
                color: var(--primary-color);
                margin-bottom: 8px;
                font-size: 0.95em;
            }
            
            .input-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 8px;
            }
            
            .results {
                background: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
            }
            
            .accuracy-badge {
                display: inline-block;
                padding: 6px 12px;
                border-radius: 15px;
                color: white;
                font-weight: bold;
                margin: 8px 0;
                text-align: center;
                width: 100%;
                font-size: 0.9em;
            }
            
            .excellent { background: linear-gradient(135deg, #27ae60, #2ecc71); }
            .very-good { background: linear-gradient(135deg, #3498db, #2980b9); }
            .good { background: linear-gradient(135deg, #f39c12, #e67e22); }
            .marginal { background: linear-gradient(135deg, #e74c3c, #c0392b); }
            
            .tower-info {
                background: #e8f4fd;
                padding: 8px;
                border-radius: 5px;
                margin: 4px 0;
                font-size: 0.8em;
            }
            
            .target-status {
                text-align: center;
                padding: 10px;
                border-radius: 8px;
                margin: 10px 0;
                font-weight: bold;
            }
            
            .target-achieved {
                background: linear-gradient(135deg, #d5f4e6, #27ae60);
                color: #155724;
            }
            
            .target-failed {
                background: linear-gradient(135deg, #f8d7da, #e74c3c);
                color: #721c24;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚗 Car Location Analysis</h1>
                <p>Calculate tower positions from car location and verify accuracy</p>
                <div class="target-badge">🎯 Target Accuracy: Under 50 meters</div>
            </div>
            
            <div class="main-grid">
                <div class="control-panel">
                    <div class="card">
                        <h3>📍 Car Location</h3>
                        <div class="input-group">
                            <label>Car Latitude</label>
                            <input type="number" id="carLat" value="28.6135" step="0.0001">
                        </div>
                        <div class="input-group">
                            <label>Car Longitude</label>
                            <input type="number" id="carLon" value="77.2088" step="0.0001">
                        </div>
                    </div>

                    <div class="card">
                        <h3>📡 Tower Measurements from Car</h3>
                        <p style="color: #666; margin-bottom: 12px; font-size: 0.8em;">Enter distance and azimuth measurements from car to each tower</p>
                        
                        <div class="tower-input-group">
                            <h4>📶 Tower 1</h4>
                            <div class="input-row">
                                <div class="input-group">
                                    <label>Distance (m)</label>
                                    <input type="number" id="tower1Distance" value="350" step="0.1">
                                </div>
                                <div class="input-group">
                                    <label>Azimuth (°)</label>
                                    <input type="number" id="tower1Azimuth" value="45" min="0" max="360">
                                </div>
                            </div>
                        </div>

                        <div class="tower-input-group">
                            <h4>📶 Tower 2</h4>
                            <div class="input-row">
                                <div class="input-group">
                                    <label>Distance (m)</label>
                                    <input type="number" id="tower2Distance" value="400" step="0.1">
                                </div>
                                <div class="input-group">
                                    <label>Azimuth (°)</label>
                                    <input type="number" id="tower2Azimuth" value="135" min="0" max="360">
                                </div>
                            </div>
                        </div>

                        <div class="tower-input-group">
                            <h4>📶 Tower 3</h4>
                            <div class="input-row">
                                <div class="input-group">
                                    <label>Distance (m)</label>
                                    <input type="number" id="tower3Distance" value="380" step="0.1">
                                </div>
                                <div class="input-group">
                                    <label>Azimuth (°)</label>
                                    <input type="number" id="tower3Azimuth" value="315" min="0" max="360">
                                </div>
                            </div>
                        </div>
                        
                        <button class="btn" onclick="calculateAnalysis()" style="background: linear-gradient(135deg, #27ae60, #2ecc71);">
                            🚀 Calculate Analysis
                        </button>
                    </div>
                    
                    <div id="resultsPanel">
                        <div class="card">
                            <h3>📊 Analysis Results</h3>
                            <div class="results">
                                <p>Enter car location and tower measurements to see accuracy analysis.</p>
                                <p style="color: #27ae60; font-weight: bold; margin-top: 8px; font-size: 0.9em;">🎯 Target: Accuracy under 50 meters</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="maps-container">
                    <div class="map-section">
                        <div class="map-title">📍 Map 1: Actual Car + Calculated Towers</div>
                        <div id="map1" class="map"></div>
                    </div>
                    <div class="map-section">
                        <div class="map-title">🎯 Map 2: Calculated Car Location</div>
                        <div id="map2" class="map"></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Initialize maps
            var map1, map2;
            var map1Markers = [], map2Markers = [];
            
            function initMaps() {
                // Map 1: Actual Car + Calculated Towers
                map1 = L.map('map1').setView([28.6135, 77.2088], 15);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map1);
                
                // Map 2: Calculated Car Location
                map2 = L.map('map2').setView([28.6135, 77.2088], 15);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map2);
            }
            
            function clearMapMarkers(map, markers) {
                markers.forEach(marker => map.removeLayer(marker));
                return [];
            }
            
            function createCarIcon(color) {
                return L.divIcon({
                    html: `<div style="background: ${color}; width: 40px; height: 40px; border-radius: 50%; border: 3px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 18px; box-shadow: 0 3px 10px rgba(0,0,0,0.3);">🚗</div>`,
                    iconSize: [46, 46],
                    iconAnchor: [23, 23]
                });
            }
            
            function createTowerIcon() {
                return L.divIcon({
                    html: '<div style="background: #e74c3c; width: 30px; height: 30px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 14px; box-shadow: 0 3px 8px rgba(0,0,0,0.3);">📡</div>',
                    iconSize: [34, 34],
                    iconAnchor: [17, 17]
                });
            }
            
            async function calculateAnalysis() {
                try {
                    const carLat = parseFloat(document.getElementById('carLat').value);
                    const carLon = parseFloat(document.getElementById('carLon').value);
                    
                    const towerMeasurements = [
                        {
                            'distance': parseFloat(document.getElementById('tower1Distance').value),
                            'azimuth': parseInt(document.getElementById('tower1Azimuth').value)
                        },
                        {
                            'distance': parseFloat(document.getElementById('tower2Distance').value),
                            'azimuth': parseInt(document.getElementById('tower2Azimuth').value)
                        },
                        {
                            'distance': parseFloat(document.getElementById('tower3Distance').value),
                            'azimuth': parseInt(document.getElementById('tower3Azimuth').value)
                        }
                    ];
                    
                    const requestData = {
                        car_lat: carLat,
                        car_lon: carLon,
                        tower_measurements: towerMeasurements
                    };
                    
                    const response = await fetch('/calculate-analysis', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(requestData)
                    });
                    
                    if (!response.ok) {
                        throw new Error('Server error: ' + response.status);
                    }
                    
                    const data = await response.json();
                    displayResults(data);
                    visualizeMaps(data);
                    
                } catch (error) {
                    console.error('Analysis error:', error);
                    alert('Analysis error: ' + error.message);
                }
            }
            
            function displayResults(data) {
                const resultsPanel = document.getElementById('resultsPanel');
                
                const accuracyClass = getAccuracyClass(data.accuracy_metrics.accuracy_grade);
                const targetAchieved = data.accuracy_metrics.target_achieved;
                const targetClass = targetAchieved ? 'target-achieved' : 'target-failed';
                const targetText = targetAchieved ? '🎯 TARGET ACHIEVED! Under 50m' : '⚠️ Target Not Met: Over 50m';
                
                let towerHtml = '';
                if (data.calculated_towers) {
                    data.calculated_towers.forEach(tower => {
                        towerHtml += `
                            <div class="tower-info">
                                <strong>${tower.tower_id}</strong><br>
                                Position: ${tower.latitude.toFixed(6)}, ${tower.longitude.toFixed(6)}<br>
                                Distance: ${tower.distance.toFixed(1)}m | Azimuth: ${tower.azimuth}°
                            </div>
                        `;
                    });
                }
                
                resultsPanel.innerHTML = `
                    <div class="card">
                        <h3>📊 Analysis Results</h3>
                        <div class="results">
                            <div class="target-status ${targetClass}">
                                ${targetText}
                            </div>
                            
                            <div style="text-align: center; margin-bottom: 12px;">
                                <div class="accuracy-badge ${accuracyClass}">
                                    ${data.accuracy_metrics.accuracy_grade}
                                </div>
                                <h2 style="margin: 8px 0; color: #2c3e50; font-size: 1.2em;">Accuracy: ${data.accuracy_metrics.distance_error_meters.toFixed(2)} meters</h2>
                            </div>
                            
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 12px 0;">
                                <div style="background: #d5f4e6; padding: 10px; border-radius: 5px; text-align: center;">
                                    <h4>🚗 Actual Car</h4>
                                    <div style="font-family: monospace; background: #2c3e50; color: white; padding: 6px; border-radius: 3px; margin: 6px 0; font-size: 0.8em;">
                                        ${data.actual_car_location.latitude.toFixed(6)}<br>${data.actual_car_location.longitude.toFixed(6)}
                                    </div>
                                </div>
                                <div style="background: #e3f2fd; padding: 10px; border-radius: 5px; text-align: center;">
                                    <h4>📡 Calculated</h4>
                                    <div style="font-family: monospace; background: #2c3e50; color: white; padding: 6px; border-radius: 3px; margin: 6px 0; font-size: 0.8em;">
                                        ${data.calculated_car_location.latitude.toFixed(6)}<br>${data.calculated_car_location.longitude.toFixed(6)}
                                    </div>
                                </div>
                            </div>
                            
                            <div style="margin: 12px 0;">
                                <h4>📶 Calculated Towers</h4>
                                ${towerHtml}
                            </div>
                        </div>
                    </div>
                `;
            }
            
            function getAccuracyClass(grade) {
                if (grade.includes('Excellent')) return 'excellent';
                if (grade.includes('Very Good')) return 'very-good';
                if (grade.includes('Good')) return 'good';
                return 'marginal';
            }
            
            function visualizeMaps(data) {
                const actualCar = data.actual_car_location;
                const calculatedCar = data.calculated_car_location;
                const towers = data.calculated_towers;
                
                // Clear previous markers
                map1Markers = clearMapMarkers(map1, map1Markers);
                map2Markers = clearMapMarkers(map2, map2Markers);
                
                // MAP 1: Actual Car + Calculated Towers
                const actualCarMarker = L.marker([actualCar.latitude, actualCar.longitude], {
                    icon: createCarIcon('#27ae60')
                }).addTo(map1);
                
                actualCarMarker.bindPopup(`
                    <div style="text-align: center;">
                        <h3 style="color: #27ae60; margin: 0;">🚗 ACTUAL CAR</h3>
                        <div>Ground Truth Position</div>
                    </div>
                `).openPopup();
                
                map1Markers.push(actualCarMarker);
                
                // Add calculated towers with distance circles
                towers.forEach(tower => {
                    const towerMarker = L.marker([tower.latitude, tower.longitude], {
                        icon: createTowerIcon()
                    }).addTo(map1);
                    
                    // Distance circle from car to tower
                    const distanceCircle = L.circle([actualCar.latitude, actualCar.longitude], {
                        color: '#e74c3c',
                        fillColor: '#e74c3c',
                        fillOpacity: 0.1,
                        radius: tower.distance
                    }).addTo(map1);
                    
                    // Connection line
                    const connectionLine = L.polyline([
                        [actualCar.latitude, actualCar.longitude],
                        [tower.latitude, tower.longitude]
                    ], {
                        color: '#3498db',
                        weight: 2,
                        opacity: 0.7
                    }).addTo(map1);
                    
                    map1Markers.push(towerMarker, distanceCircle, connectionLine);
                });
                
                // Fit map to show all points
                const map1Points = [[actualCar.latitude, actualCar.longitude]];
                towers.forEach(tower => {
                    map1Points.push([tower.latitude, tower.longitude]);
                });
                map1.fitBounds(map1Points, {padding: [20, 20]});
                
                // MAP 2: Calculated Car Location
                const calculatedCarMarker = L.marker([calculatedCar.latitude, calculatedCar.longitude], {
                    icon: createCarIcon('#3498db')
                }).addTo(map2);
                
                calculatedCarMarker.bindPopup(`
                    <div style="text-align: center;">
                        <h3 style="color: #3498db; margin: 0;">📡 CALCULATED CAR</h3>
                        <div>Accuracy: ${data.accuracy_metrics.distance_error_meters.toFixed(2)}m</div>
                    </div>
                `).openPopup();
                
                map2Markers.push(calculatedCarMarker);
                
                // Add actual car for comparison
                const actualOnMap2 = L.marker([actualCar.latitude, actualCar.longitude], {
                    icon: createCarIcon('#27ae60')
                }).addTo(map2);
                
                map2Markers.push(actualOnMap2);
                
                // Show error line
                const errorLine = L.polyline([
                    [actualCar.latitude, actualCar.longitude],
                    [calculatedCar.latitude, calculatedCar.longitude]
                ], {
                    color: '#e74c3c',
                    weight: 3,
                    opacity: 0.8,
                    dashArray: '8, 8'
                }).addTo(map2);
                
                map2Markers.push(errorLine);
                
                // Add error distance label
                const errorMidpoint = [
                    (actualCar.latitude + calculatedCar.latitude) / 2,
                    (actualCar.longitude + calculatedCar.longitude) / 2
                ];
                
                const errorLabel = L.marker(errorMidpoint, {
                    icon: L.divIcon({
                        html: `<div style="background: #e74c3c; color: white; padding: 4px 8px; border-radius: 6px; font-weight: bold; font-size: 0.8em;">${data.accuracy_metrics.distance_error_meters.toFixed(1)}m error</div>`,
                        iconSize: [70, 20],
                        iconAnchor: [35, 10]
                    })
                }).addTo(map2);
                
                map2Markers.push(errorLabel);
                
                // Add towers to map 2
                towers.forEach(tower => {
                    const towerMarker = L.marker([tower.latitude, tower.longitude], {
                        icon: createTowerIcon()
                    }).addTo(map2);
                    
                    map2Markers.push(towerMarker);
                });
                
                // Fit map 2 to show all points
                const map2Points = [[calculatedCar.latitude, calculatedCar.longitude], [actualCar.latitude, actualCar.longitude]];
                towers.forEach(tower => {
                    map2Points.push([tower.latitude, tower.longitude]);
                });
                map2.fitBounds(map2Points, {padding: [20, 20]});
            }
            
            // Initialize when page loads
            document.addEventListener('DOMContentLoaded', function() {
                initMaps();
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/calculate-analysis")
async def calculate_analysis(request: CarAnalysisRequest):
    """Calculate complete analysis"""
    try:
        result = await car_system.calculate_complete_analysis(
            float(request.car_lat),
            float(request.car_lon),
            request.tower_measurements
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Analysis calculation error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")