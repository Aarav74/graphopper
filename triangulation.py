import numpy as np
import math
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Your existing triangulation code (keep the same as before)
class HighPrecisionESIMTracker:
    def __init__(self):
        self.cell_tower_database = self.initialize_tower_database()
        self.signal_propagation_models = self.initialize_propagation_models()
    
    def initialize_tower_database(self) -> Dict:
        """Initialize database with your specific towers"""
        return {
            '24823': {
                'lat': 28.639500, 
                'lon': 77.224800, 
                'type': '4G', 
                'height': 35,
                'operator': 'JIO',
                'frequency': 1800
            },
            '27639': {
                'lat': 28.637200, 
                'lon': 77.222500, 
                'type': '4G', 
                'height': 32,
                'operator': 'JIO', 
                'frequency': 1800
            },
            '27682': {
                'lat': 28.640800, 
                'lon': 77.226200, 
                'type': '4G', 
                'height': 40,
                'operator': 'JIO',
                'frequency': 1800
            },
            '12345678': {'lat': 28.632400, 'lon': 77.218800, 'type': '4G', 'height': 35},
            '12345679': {'lat': 28.631500, 'lon': 77.217500, 'type': '4G', 'height': 32},
        }
    
    def initialize_propagation_models(self) -> Dict:
        """Initialize signal propagation models for different environments"""
        return {
            'dense_urban': {'path_loss_exp': 3.5, 'shadow_std': 12},
            'urban': {'path_loss_exp': 3.2, 'shadow_std': 10},
            'suburban': {'path_loss_exp': 2.8, 'shadow_std': 8},
            'rural': {'path_loss_exp': 2.3, 'shadow_std': 6},
            'free_space': {'path_loss_exp': 2.0, 'shadow_std': 3}
        }
    
    def haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """High precision Haversine distance calculation"""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        R = 6371000.0
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def calculate_tower_coordinates(self, reference_point: Tuple[float, float], 
                                 distance: float, azimuth: float) -> Tuple[float, float]:
        """Calculate tower coordinates from reference point"""
        ref_lat, ref_lon = reference_point
        
        lat_distance_km = distance / 1000.0
        lon_distance_km = distance / 1000.0
        
        lat_deg_per_km = 1 / 110.574
        lon_deg_per_km = 1 / (111.320 * math.cos(math.radians(ref_lat)))
        
        azimuth_rad = math.radians(azimuth)
        
        delta_lat = lat_distance_km * math.cos(azimuth_rad) * lat_deg_per_km
        delta_lon = lon_distance_km * math.sin(azimuth_rad) * lon_deg_per_km
        
        tower_lat = ref_lat + delta_lat
        tower_lon = ref_lon + delta_lon
        
        return (tower_lat, tower_lon)
    
    def advanced_triangulation(self, tower_measurements: List[Dict], 
                             reference_location: Optional[Tuple[float, float]] = None,
                             gps_location: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """Advanced triangulation using provided distances and azimuths"""
        print(f"Starting triangulation with {len(tower_measurements)} measurements...")
        
        towers = []
        valid_towers_count = 0
        
        if reference_location is None:
            ref_point = (28.632400, 77.218800)
        else:
            ref_point = reference_location
        
        # Store detailed tower information for UI
        detailed_towers = []
        
        for meas in tower_measurements:
            tower_id = meas.get('cell_id')
            
            if tower_id in self.cell_tower_database:
                tower_data = self.cell_tower_database[tower_id]
                tower_coords = (tower_data['lat'], tower_data['lon'])
                tower_source = "database"
            elif 'distance' in meas and 'azimuth' in meas:
                tower_coords = self.calculate_tower_coordinates(
                    ref_point, meas['distance'], meas['azimuth']
                )
                tower_source = "calculated"
            else:
                continue
            
            distance = meas['distance']
            signal_strength = meas.get('signal_strength', -102.495)
            confidence = self.calculate_confidence_from_signal(signal_strength)
            
            # Store detailed tower info for UI
            detailed_tower_info = {
                'id': tower_id,
                'coordinates': tower_coords,
                'distance': distance,
                'signal_strength': signal_strength,
                'azimuth': meas.get('azimuth'),
                'source': tower_source,
                'confidence': confidence,
                'range_meters': distance * 0.25  # 25% range for visualization
            }
            detailed_towers.append(detailed_tower_info)
            
            towers.append({
                'id': tower_id,
                'coordinates': tower_coords,
                'distance': distance,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'azimuth': meas.get('azimuth'),
                'weight': 1.0 / (confidence + 1e-6)
            })
            valid_towers_count += 1
        
        if len(towers) < 2:
            return {'error': 'Insufficient towers for triangulation'}
        
        # Run triangulation methods
        results = {}
        results['weighted_least_squares'] = self.weighted_least_squares(towers)
        results['circular_intersection'] = self.circular_intersection(towers)
        results['weighted_centroid'] = self.weighted_centroid(towers)
        
        if len(towers) >= 3:
            results['non_linear_opt'] = self.non_linear_optimization(towers)
        
        final_result = self.combine_triangulation_results(results, towers)
        
        # Add GPS comparison if available
        if gps_location:
            gps_accuracy = self.calculate_gps_accuracy()
            triangulation_accuracy = final_result['accuracy_meters']
            
            # Calculate distance between GPS and triangulated location
            distance_error = self.haversine_distance(
                gps_location, 
                final_result['estimated_location']
            )
            
            # Calculate improvement percentage
            improvement = ((triangulation_accuracy - distance_error) / triangulation_accuracy * 100) \
                if triangulation_accuracy > 0 else 0
            
            final_result['gps_comparison'] = {
                'gps_location': gps_location,
                'gps_accuracy_meters': gps_accuracy,
                'distance_error_meters': distance_error,
                'improvement_percentage': improvement,
                'is_improved': improvement > 0
            }
        
        # Add detailed information for UI
        final_result['detailed_towers'] = detailed_towers
        final_result['input_location'] = ref_point
        final_result['methods_results'] = {
            method: {
                'location': result['location'],
                'accuracy': result['accuracy']
            }
            for method, result in results.items()
            if result['location'] is not None
        }
        
        return final_result
    
    def calculate_gps_accuracy(self) -> float:
        """Calculate typical GPS accuracy based on conditions"""
        # Simulate GPS accuracy (typically 5-15 meters for good conditions)
        return 8.0  # meters
    
    def calculate_confidence_from_signal(self, signal_strength: float) -> float:
        if signal_strength > -70: return 0.15
        elif signal_strength > -85: return 0.20
        elif signal_strength > -100: return 0.25
        else: return 0.35
    
    def weighted_least_squares(self, towers: List[Dict]) -> Dict[str, Any]:
        try:
            lat_sum, lon_sum, total_weight = 0, 0, 0
            for tower in towers:
                weight = tower['weight']
                lat_sum += tower['coordinates'][0] * weight
                lon_sum += tower['coordinates'][1] * weight
                total_weight += weight
            
            x0 = np.array([lat_sum / total_weight, lon_sum / total_weight])
            
            def objective_function(x):
                total_error = 0
                lat, lon = x
                for tower in towers:
                    tower_lat, tower_lon = tower['coordinates']
                    calculated_distance = self.haversine_distance((lat, lon), (tower_lat, tower_lon))
                    error = (calculated_distance - tower['distance']) ** 2
                    total_error += error * tower['weight']
                return total_error
            
            lat_coords = [t['coordinates'][0] for t in towers]
            lon_coords = [t['coordinates'][1] for t in towers]
            
            bounds = [
                (min(lat_coords) - 0.01, max(lat_coords) + 0.01),
                (min(lon_coords) - 0.01, max(lon_coords) + 0.01)
            ]
            
            result = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1e-6})
            
            if result.success:
                estimated_location = tuple(result.x)
                accuracy = self.calculate_accuracy_metrics(estimated_location, towers)
                return {'location': estimated_location, 'accuracy': accuracy, 'method': 'weighted_least_squares'}
        except Exception as e:
            print(f"WLS Error: {e}")
        return {'location': None, 'accuracy': float('inf'), 'method': 'weighted_least_squares'}
    
    def circular_intersection(self, towers: List[Dict]) -> Dict[str, Any]:
        try:
            best_result = {'location': None, 'accuracy': float('inf'), 'method': 'circular_intersection'}
            for i in range(len(towers)):
                for j in range(i+1, len(towers)):
                    for k in range(j+1, len(towers)):
                        t1, t2, t3 = towers[i], towers[j], towers[k]
                        intersections = self.three_circle_intersection(t1, t2, t3)
                        if intersections:
                            best_intersection = None
                            min_error = float('inf')
                            for point in intersections:
                                error = 0
                                for tower in [t1, t2, t3]:
                                    calculated_dist = self.haversine_distance(point, tower['coordinates'])
                                    error += abs(calculated_dist - tower['distance']) * tower['weight']
                                if error < min_error:
                                    min_error = error
                                    best_intersection = point
                            if best_intersection:
                                accuracy = self.calculate_accuracy_metrics(best_intersection, [t1, t2, t3])
                                if accuracy < best_result['accuracy']:
                                    best_result = {'location': best_intersection, 'accuracy': accuracy, 'method': 'circular_intersection'}
            return best_result
        except Exception as e:
            print(f"Circular Intersection Error: {e}")
        return {'location': None, 'accuracy': float('inf'), 'method': 'circular_intersection'}
    
    def three_circle_intersection(self, t1: Dict, t2: Dict, t3: Dict) -> List[Tuple[float, float]]:
        try:
            x1, y1 = t1['coordinates'][1], t1['coordinates'][0]
            x2, y2 = t2['coordinates'][1], t2['coordinates'][0]
            x3, y3 = t3['coordinates'][1], t3['coordinates'][0]
            
            r1 = t1['distance'] / 111320
            r2 = t2['distance'] / 111320
            r3 = t3['distance'] / 111320
            
            dx = x2 - x1
            dy = y2 - y1
            d = math.sqrt(dx*dx + dy*dy)
            
            if d > (r1 + r2) or d < abs(r1 - r2):
                return []
            
            a = (r1*r1 - r2*r2 + d*d) / (2 * d)
            h = math.sqrt(r1*r1 - a*a)
            
            x0 = x1 + a * dx / d
            y0 = y1 + a * dy / d
            
            rx = -dy * (h / d)
            ry = dx * (h / d)
            
            points = [(y0 + ry, x0 + rx), (y0 - ry, x0 - rx)]
            
            valid_points = []
            for point in points:
                total_error = 0
                valid = True
                for tower in [t1, t2, t3]:
                    dist_to_tower = self.haversine_distance(point, tower['coordinates'])
                    error_ratio = abs(dist_to_tower - tower['distance']) / tower['distance']
                    if error_ratio > 1.0:
                        valid = False
                        break
                    total_error += error_ratio
                if valid and total_error < 2.0:
                    valid_points.append(point)
            return valid_points
        except Exception as e:
            print(f"Circle intersection error: {e}")
            return []
    
    def weighted_centroid(self, towers: List[Dict]) -> Dict[str, Any]:
        try:
            lat_sum, lon_sum, total_weight = 0, 0, 0
            for tower in towers:
                weight = tower['weight']
                lat_sum += tower['coordinates'][0] * weight
                lon_sum += tower['coordinates'][1] * weight
                total_weight += weight
            
            if total_weight > 0:
                estimated_location = (lat_sum / total_weight, lon_sum / total_weight)
                accuracy = self.calculate_accuracy_metrics(estimated_location, towers)
                return {'location': estimated_location, 'accuracy': accuracy, 'method': 'weighted_centroid'}
        except Exception as e:
            print(f"Centroid Error: {e}")
        return {'location': None, 'accuracy': float('inf'), 'method': 'weighted_centroid'}
    
    def non_linear_optimization(self, towers: List[Dict]) -> Dict[str, Any]:
        try:
            lat_sum, lon_sum, total_weight = 0, 0, 0
            for tower in towers:
                weight = tower['weight']
                lat_sum += tower['coordinates'][0] * weight
                lon_sum += tower['coordinates'][1] * weight
                total_weight += weight
            
            x0 = np.array([lat_sum / total_weight, lon_sum / total_weight])
            
            def objective_function(x):
                total_error = 0
                lat, lon = x
                for tower in towers:
                    calculated_distance = self.haversine_distance((lat, lon), tower['coordinates'])
                    error = math.log1p(abs(calculated_distance - tower['distance']))
                    total_error += error * tower['weight']
                return total_error
            
            lat_coords = [t['coordinates'][0] for t in towers]
            lon_coords = [t['coordinates'][1] for t in towers]
            
            bounds = [
                (min(lat_coords) - 0.01, max(lat_coords) + 0.01),
                (min(lon_coords) - 0.01, max(lon_coords) + 0.01)
            ]
            
            result = minimize(objective_function, x0, method='Nelder-Mead', bounds=bounds,
                            options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result.success:
                estimated_location = tuple(result.x)
                accuracy = self.calculate_accuracy_metrics(estimated_location, towers)
                return {'location': estimated_location, 'accuracy': accuracy, 'method': 'non_linear_opt'}
        except Exception as e:
            print(f"Non-linear optimization error: {e}")
        return {'location': None, 'accuracy': float('inf'), 'method': 'non_linear_opt'}
    
    def combine_triangulation_results(self, results: Dict, towers: List[Dict]) -> Dict[str, Any]:
        valid_results = []
        for method, result in results.items():
            if result['location'] is not None and result['accuracy'] < float('inf'):
                valid_results.append(result)
        
        if not valid_results:
            return {'error': 'All triangulation methods failed'}
        
        total_weight = 0
        weighted_lat, weighted_lon = 0, 0
        for result in valid_results:
            weight = 1.0 / (result['accuracy'] + 1e-6)
            weighted_lat += result['location'][0] * weight
            weighted_lon += result['location'][1] * weight
            total_weight += weight
        
        if total_weight > 0:
            final_location = (weighted_lat / total_weight, weighted_lon / total_weight)
            final_accuracy = self.calculate_accuracy_metrics(final_location, towers)
            
            return {
                'estimated_location': final_location,
                'accuracy_meters': final_accuracy,
                'number_of_towers': len(towers),
                'methods_used': [r['method'] for r in valid_results],
                'confidence': self.calculate_confidence(final_accuracy, len(towers)),
                'tower_details': [
                    {
                        'tower_id': t['id'],
                        'distance_estimated': t['distance'],
                        'signal_strength': t['signal_strength'],
                        'azimuth': t.get('azimuth')
                    } for t in towers
                ]
            }
        else:
            return {'error': 'Failed to combine results'}
    
    def calculate_accuracy_metrics(self, location: Tuple[float, float], towers: List[Dict]) -> float:
        errors = []
        for tower in towers:
            calculated_distance = self.haversine_distance(location, tower['coordinates'])
            error = abs(calculated_distance - tower['distance'])
            errors.append(error)
        
        if errors:
            rms_error = math.sqrt(sum(e**2 for e in errors) / len(errors))
        else:
            rms_error = 1000
        
        geometry_factor = self.calculate_geometry_factor(towers)
        return max(10, rms_error * geometry_factor)
    
    def calculate_geometry_factor(self, towers: List[Dict]) -> float:
        if len(towers) < 3:
            return 2.5
        
        lats = [t['coordinates'][0] for t in towers]
        lons = [t['coordinates'][1] for t in towers]
        
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        area = lat_span * lon_span
        
        if area < 0.00005: return 2.0
        elif area < 0.0002: return 1.5
        elif area < 0.001: return 1.2
        else: return 1.0
    
    def calculate_confidence(self, accuracy: float, num_towers: int) -> str:
        if accuracy < 25 and num_towers >= 4: return 'very_high'
        elif accuracy < 50 and num_towers >= 3: return 'high'
        elif accuracy < 100 and num_towers >= 2: return 'medium'
        elif accuracy < 200: return 'low'
        else: return 'very_low'

# Pydantic models for API
class TowerMeasurement(BaseModel):
    cell_id: str
    distance: float
    azimuth: float
    signal_strength: float = -102.495
    environment: str = "urban"

class TriangulationRequest(BaseModel):
    tower_measurements: List[TowerMeasurement]
    reference_latitude: float = 28.632400
    reference_longitude: float = 77.218800
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None

@app.get("/", response_class=HTMLResponse)
async def serve_triangulation_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESIM Triangulation Visualizer</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            #map { height: 600px; width: 100%; }
            .tower-circle { stroke-width: 2; fill-opacity: 0.1; }
            .method-marker { background: transparent; border: none; }
            .info-panel { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .fullscreen-mode { 
                position: fixed; 
                top: 0; 
                left: 0; 
                width: 100vw; 
                height: 100vh; 
                z-index: 9999; 
                background: white;
                display: none;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                padding: 20px;
            }
            .fullscreen-map-container { 
                height: 100%; 
                position: relative;
            }
            .fullscreen-results-container {
                height: 100%;
                overflow-y: auto;
                padding: 20px;
            }
            .fullscreen-controls { 
                position: absolute; 
                top: 10px; 
                right: 10px; 
                z-index: 10000; 
            }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen p-4">
        <div class="max-w-7xl mx-auto">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-2">ESIM Triangulation Visualizer</h1>
            <p class="text-center text-gray-600 mb-6">Visualize cell tower triangulation with GPS comparison</p>
            
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Input Panel -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">Tower Measurements</h2>
                    
                    <div class="space-y-4 mb-4">
                        <div class="grid grid-cols-2 gap-2">
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Reference Latitude</label>
                                <input type="number" id="refLat" value="28.632400" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Reference Longitude</label>
                                <input type="number" id="refLon" value="77.218800" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border">
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-2">
                            <div>
                                <label class="block text-sm font-medium text-gray-700">GPS Latitude (Optional)</label>
                                <input type="number" id="gpsLat" placeholder="28.632429" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">GPS Longitude (Optional)</label>
                                <input type="number" id="gpsLon" placeholder="77.218788" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border">
                            </div>
                        </div>
                    </div>

                    <div class="space-y-3 mb-4">
                        <h3 class="font-semibold text-gray-700">Add Tower Measurement</h3>
                        <div class="grid grid-cols-2 gap-2">
                            <input type="text" id="cellId" placeholder="Cell ID" class="rounded-md border-gray-300 shadow-sm p-2 border">
                            <input type="number" id="distance" placeholder="Distance (m)" step="any" class="rounded-md border-gray-300 shadow-sm p-2 border">
                        </div>
                        <div class="grid grid-cols-2 gap-2">
                            <input type="number" id="azimuth" placeholder="Azimuth (°)" step="any" class="rounded-md border-gray-300 shadow-sm p-2 border">
                            <input type="number" id="signal" placeholder="Signal (dBm)" value="-102.495" step="any" class="rounded-md border-gray-300 shadow-sm p-2 border">
                        </div>
                        <button id="addTowerBtn" class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg transition-all">
                            Add Tower
                        </button>
                    </div>

                    <div id="towersList" class="max-h-40 overflow-y-auto space-y-2 mb-4">
                        <p class="text-sm text-gray-500 text-center">No towers added yet</p>
                    </div>

                    <div class="space-y-2">
                        <button id="calculateBtn" class="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-4 rounded-lg transition-all">
                            Calculate Triangulation
                        </button>
                        <button id="clearBtn" class="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg transition-all">
                            Clear All
                        </button>
                        <button id="loadSampleBtn" class="w-full bg-purple-500 hover:bg-purple-600 text-white font-bold py-2 px-4 rounded-lg transition-all">
                            Load Sample Data
                        </button>
                        <button id="fullscreenBtn" class="w-full bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-lg transition-all">
                            Full Screen View
                        </button>
                    </div>
                </div>

                <!-- Results Panel -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">Triangulation Results</h2>
                    
                    <div id="resultsContent" class="space-y-4">
                        <div class="text-center text-gray-500">
                            <p>Calculate triangulation to see results</p>
                        </div>
                    </div>

                    <div class="mt-6">
                        <h3 class="font-semibold text-gray-700 mb-2">GPS Comparison</h3>
                        <div id="gpsComparison" class="space-y-2">
                            <!-- GPS comparison will be populated here -->
                        </div>
                    </div>

                    <div class="mt-6">
                        <h3 class="font-semibold text-gray-700 mb-2">Algorithm Results</h3>
                        <div id="methodsResults" class="space-y-2">
                            <!-- Method results will be populated here -->
                        </div>
                    </div>

                    <div class="mt-6">
                        <h3 class="font-semibold text-gray-700 mb-2">Tower Details</h3>
                        <div id="towerDetails" class="space-y-2">
                            <!-- Tower details will be populated here -->
                        </div>
                    </div>
                </div>

                <!-- Map Panel -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">Visualization Map</h2>
                    <div id="map"></div>
                    <div class="mt-4 bg-blue-50 p-3 rounded-lg">
                        <p class="text-sm text-blue-700">
                            <strong>Map Legend:</strong><br>
                            <span style="color: #FF4444">📍 Input Location</span><br>
                            <span style="color: #44FF44">📍 Calculated Location</span><br>
                            <span style="color: #4444FF">📍 GPS Location</span><br>
                            <span style="color: #FF6B00">📍 Algorithm Results</span><br>
                            <span style="color: blue">⭕ Tower Range</span><br>
                            <span style="color: purple">📡 Tower Location</span>
                        </p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Full Screen Mode -->
        <div id="fullscreenMode" class="fullscreen-mode">
            <div class="fullscreen-map-container">
                <div class="fullscreen-controls">
                    <button id="exitFullscreenBtn" class="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg transition-all">
                        Exit Full Screen
                    </button>
                </div>
                <div id="fullscreenMap" class="w-full h-full"></div>
            </div>
            <div class="fullscreen-results-container bg-gray-50 rounded-lg">
                <h2 class="text-2xl font-bold text-gray-800 mb-4">Triangulation Results - Full Screen</h2>
                <div id="fullscreenResultsContent" class="space-y-4">
                    <div class="text-center text-gray-500">
                        <p>Calculate triangulation to see results</p>
                    </div>
                </div>
                <div class="mt-6">
                    <h3 class="font-semibold text-gray-700 mb-2">GPS Comparison</h3>
                    <div id="fullscreenGpsComparison" class="space-y-2">
                        <!-- GPS comparison will be populated here -->
                    </div>
                </div>
                <div class="mt-6">
                    <h3 class="font-semibold text-gray-700 mb-2">Algorithm Results</h3>
                    <div id="fullscreenMethodsResults" class="space-y-2">
                        <!-- Method results will be populated here -->
                    </div>
                </div>
                <div class="mt-6">
                    <h3 class="font-semibold text-gray-700 mb-2">Tower Details</h3>
                    <div id="fullscreenTowerDetails" class="space-y-2">
                        <!-- Tower details will be populated here -->
                    </div>
                </div>
            </div>
        </div>

        <script>
            let map;
            let fullscreenMap;
            let markers = [];
            let fullscreenMarkers = [];
            let circles = [];
            let fullscreenCircles = [];
            let towers = [];
            let currentData = null;

            function initializeMap() {
                map = L.map('map').setView([28.6324, 77.2188], 14);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
            }

            function initializeFullscreenMap() {
                const container = document.getElementById('fullscreenMap');
                fullscreenMap = L.map(container).setView([28.6324, 77.2188], 14);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(fullscreenMap);
            }

            function clearMap(targetMap = map, clearMarkers = true, clearCircles = true) {
                if (clearMarkers) {
                    markers.forEach(marker => targetMap.removeLayer(marker));
                    markers = [];
                }
                if (clearCircles) {
                    circles.forEach(circle => targetMap.removeLayer(circle));
                    circles = [];
                }
            }

            function clearFullscreenMap() {
                if (fullscreenMap) {
                    fullscreenMarkers.forEach(marker => fullscreenMap.removeLayer(marker));
                    fullscreenCircles.forEach(circle => fullscreenMap.removeLayer(circle));
                    fullscreenMarkers = [];
                    fullscreenCircles = [];
                }
            }

            function addMarker(lat, lng, color, label, popup, targetMap = map, isFullscreen = false) {
                const icon = L.divIcon({
                    html: `<div style="background-color: ${color}; width: 24px; height: 24px; border-radius: 50%; border: 2px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">${label}</div>`,
                    className: 'custom-marker',
                    iconSize: [28, 28],
                    iconAnchor: [14, 14]
                });
                
                const marker = L.marker([lat, lng], { icon: icon })
                    .addTo(targetMap)
                    .bindPopup(popup);
                
                if (isFullscreen) {
                    fullscreenMarkers.push(marker);
                } else {
                    markers.push(marker);
                }
                return marker;
            }

            function addCircle(lat, lng, radius, color, targetMap = map, isFullscreen = false) {
                const circle = L.circle([lat, lng], {
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.1,
                    radius: radius
                }).addTo(targetMap);
                
                if (isFullscreen) {
                    fullscreenCircles.push(circle);
                } else {
                    circles.push(circle);
                }
                return circle;
            }

            function updateTowersList() {
                const list = document.getElementById('towersList');
                if (towers.length === 0) {
                    list.innerHTML = '<p class="text-sm text-gray-500 text-center">No towers added yet</p>';
                } else {
                    list.innerHTML = '';
                    towers.forEach((tower, index) => {
                        const item = document.createElement('div');
                        item.className = 'bg-gray-50 p-2 rounded text-sm flex justify-between items-center';
                        item.innerHTML = `
                            <div>
                                <strong>${tower.cell_id}</strong><br>
                                Dist: ${tower.distance.toFixed(1)}m, Az: ${tower.azimuth}°
                            </div>
                            <button onclick="removeTower(${index})" class="text-red-500 hover:text-red-700">×</button>
                        `;
                        list.appendChild(item);
                    });
                }
            }

            window.removeTower = function(index) {
                towers.splice(index, 1);
                updateTowersList();
            };

            function loadSampleData() {
                towers = [
                    { cell_id: '24823', distance: 746.32, azimuth: 217, signal_strength: -102.495 },
                    { cell_id: '27639', distance: 769.69, azimuth: 205, signal_strength: -103.0 },
                    { cell_id: '27682', distance: 876.91, azimuth: 221, signal_strength: -102.495 }
                ];
                // Set sample GPS coordinates
                document.getElementById('gpsLat').value = '28.632429';
                document.getElementById('gpsLon').value = '77.218788';
                updateTowersList();
            }

            async function calculateTriangulation() {
                if (towers.length < 2) {
                    alert('Please add at least 2 towers for triangulation');
                    return;
                }

                const refLat = parseFloat(document.getElementById('refLat').value);
                const refLon = parseFloat(document.getElementById('refLon').value);
                const gpsLat = document.getElementById('gpsLat').value ? parseFloat(document.getElementById('gpsLat').value) : null;
                const gpsLon = document.getElementById('gpsLon').value ? parseFloat(document.getElementById('gpsLon').value) : null;

                const requestData = {
                    tower_measurements: towers,
                    reference_latitude: refLat,
                    reference_longitude: refLon
                };

                // Add GPS data if provided
                if (gpsLat && gpsLon) {
                    requestData.gps_latitude = gpsLat;
                    requestData.gps_longitude = gpsLon;
                }

                try {
                    const response = await fetch('/calculate-triangulation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestData)
                    });

                    if (!response.ok) {
                        throw new Error('Server error: ' + response.status);
                    }

                    const data = await response.json();
                    currentData = data; // Store for fullscreen
                    displayResults(data);
                    displayFullscreenResults(data);
                    visualizeOnMap(data, refLat, refLon, gpsLat, gpsLon);

                } catch (error) {
                    alert('Error calculating triangulation: ' + error.message);
                }
            }

            function displayResults(data) {
                const resultsDiv = document.getElementById('resultsContent');
                const gpsDiv = document.getElementById('gpsComparison');
                const methodsDiv = document.getElementById('methodsResults');
                const towersDiv = document.getElementById('towerDetails');

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="bg-red-50 p-4 rounded-lg text-red-700">Error: ${data.error}</div>`;
                    return;
                }

                updateResultsDisplay(resultsDiv, gpsDiv, methodsDiv, towersDiv, data);
            }

            function displayFullscreenResults(data) {
                const resultsDiv = document.getElementById('fullscreenResultsContent');
                const gpsDiv = document.getElementById('fullscreenGpsComparison');
                const methodsDiv = document.getElementById('fullscreenMethodsResults');
                const towersDiv = document.getElementById('fullscreenTowerDetails');

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="bg-red-50 p-4 rounded-lg text-red-700">Error: ${data.error}</div>`;
                    return;
                }

                updateResultsDisplay(resultsDiv, gpsDiv, methodsDiv, towersDiv, data);
            }

            function updateResultsDisplay(resultsDiv, gpsDiv, methodsDiv, towersDiv, data) {
                // Main results
                resultsDiv.innerHTML = `
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-green-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-green-800">Input Location</h4>
                            <p class="text-sm">${data.input_location[0].toFixed(6)}, ${data.input_location[1].toFixed(6)}</p>
                        </div>
                        <div class="bg-blue-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-blue-800">Calculated Location</h4>
                            <p class="text-sm">${data.estimated_location[0].toFixed(6)}, ${data.estimated_location[1].toFixed(6)}</p>
                        </div>
                        <div class="bg-purple-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-purple-800">Triangulation Accuracy</h4>
                            <p class="text-sm">${data.accuracy_meters.toFixed(1)} meters</p>
                        </div>
                        <div class="bg-orange-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-orange-800">Confidence</h4>
                            <p class="text-sm">${data.confidence}</p>
                        </div>
                    </div>
                    <div class="bg-gray-50 p-3 rounded-lg">
                        <h4 class="font-semibold text-gray-800">Methods Used</h4>
                        <p class="text-sm">${data.methods_used.join(', ')}</p>
                    </div>
                `;

                // GPS Comparison
                if (data.gps_comparison) {
                    const gps = data.gps_comparison;
                    const improvementColor = gps.is_improved ? 'text-green-600' : 'text-red-600';
                    const improvementIcon = gps.is_improved ? '↗' : '↘';
                    
                    gpsDiv.innerHTML = `
                        <div class="bg-blue-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-blue-800">GPS Location</h4>
                            <p class="text-sm">${gps.gps_location[0].toFixed(6)}, ${gps.gps_location[1].toFixed(6)}</p>
                            <p class="text-sm">GPS Accuracy: ${gps.gps_accuracy_meters.toFixed(1)}m</p>
                        </div>
                        <div class="bg-${gps.is_improved ? 'green' : 'red'}-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-${gps.is_improved ? 'green' : 'red'}-800">Accuracy Comparison</h4>
                            <p class="text-sm">Distance Error: ${gps.distance_error_meters.toFixed(1)}m</p>
                            <p class="text-sm ${improvementColor}">
                                ${improvementIcon} ${Math.abs(gps.improvement_percentage).toFixed(1)}% ${gps.is_improved ? 'improvement' : 'reduction'}
                            </p>
                        </div>
                    `;
                } else {
                    gpsDiv.innerHTML = '<div class="bg-gray-50 p-3 rounded-lg text-gray-500 text-center">No GPS data provided</div>';
                }

                // Method results
                if (data.methods_results) {
                    methodsDiv.innerHTML = '';
                    Object.entries(data.methods_results).forEach(([method, result]) => {
                        const methodDiv = document.createElement('div');
                        methodDiv.className = 'bg-gray-50 p-2 rounded text-sm';
                        methodDiv.innerHTML = `
                            <strong>${method}</strong><br>
                            Location: ${result.location[0].toFixed(6)}, ${result.location[1].toFixed(6)}<br>
                            Accuracy: ${result.accuracy.toFixed(1)}m
                        `;
                        methodsDiv.appendChild(methodDiv);
                    });
                }

                // Tower details
                if (data.detailed_towers) {
                    towersDiv.innerHTML = '';
                    data.detailed_towers.forEach(tower => {
                        const towerDiv = document.createElement('div');
                        towerDiv.className = 'bg-gray-50 p-2 rounded text-sm';
                        towerDiv.innerHTML = `
                            <strong>Tower ${tower.id}</strong><br>
                            Location: ${tower.coordinates[0].toFixed(6)}, ${tower.coordinates[1].toFixed(6)}<br>
                            Distance: ${tower.distance.toFixed(1)}m, Signal: ${tower.signal_strength}dBm<br>
                            Source: ${tower.source}, Range: ±${tower.range_meters.toFixed(0)}m
                        `;
                        towersDiv.appendChild(towerDiv);
                    });
                }
            }

            function visualizeOnMap(data, refLat, refLon, gpsLat, gpsLon) {
                // Clear both maps
                clearMap(map, true, true);
                clearFullscreenMap();

                // Add input location
                addMarker(refLat, refLon, '#FF4444', 'I', 
                    `<strong>Input Reference Location</strong><br>${refLat.toFixed(6)}, ${refLon.toFixed(6)}`, map);
                if (fullscreenMap) addMarker(refLat, refLon, '#FF4444', 'I', 
                    `<strong>Input Reference Location</strong><br>${refLat.toFixed(6)}, ${refLon.toFixed(6)}`, fullscreenMap, true);

                // Add GPS location if available
                if (gpsLat && gpsLon) {
                    addMarker(gpsLat, gpsLon, '#4444FF', 'G',
                        `<strong>GPS Location</strong><br>${gpsLat.toFixed(6)}, ${gpsLon.toFixed(6)}<br>Accuracy: ${data.gps_comparison?.gps_accuracy_meters.toFixed(1)}m`, map);
                    if (fullscreenMap) addMarker(gpsLat, gpsLon, '#4444FF', 'G',
                        `<strong>GPS Location</strong><br>${gpsLat.toFixed(6)}, ${gpsLon.toFixed(6)}<br>Accuracy: ${data.gps_comparison?.gps_accuracy_meters.toFixed(1)}m`, fullscreenMap, true);
                }

                // Add calculated location
                if (data.estimated_location) {
                    addMarker(data.estimated_location[0], data.estimated_location[1], '#44FF44', 'C',
                        `<strong>Calculated Location</strong><br>${data.estimated_location[0].toFixed(6)}, ${data.estimated_location[1].toFixed(6)}<br>Accuracy: ${data.accuracy_meters.toFixed(1)}m`, map);
                    if (fullscreenMap) addMarker(data.estimated_location[0], data.estimated_location[1], '#44FF44', 'C',
                        `<strong>Calculated Location</strong><br>${data.estimated_location[0].toFixed(6)}, ${data.estimated_location[1].toFixed(6)}<br>Accuracy: ${data.accuracy_meters.toFixed(1)}m`, fullscreenMap, true);
                }

                // Add method results
                if (data.methods_results) {
                    Object.entries(data.methods_results).forEach(([method, result]) => {
                        const label = method.substring(0, 1).toUpperCase();
                        addMarker(result.location[0], result.location[1], '#FF6B00', label,
                            `<strong>${method}</strong><br>${result.location[0].toFixed(6)}, ${result.location[1].toFixed(6)}<br>Accuracy: ${result.accuracy.toFixed(1)}m`, map);
                        if (fullscreenMap) addMarker(result.location[0], result.location[1], '#FF6B00', label,
                            `<strong>${method}</strong><br>${result.location[0].toFixed(6)}, ${result.location[1].toFixed(6)}<br>Accuracy: ${result.accuracy.toFixed(1)}m`, fullscreenMap, true);
                    });
                }

                // Add towers and their ranges
                if (data.detailed_towers) {
                    data.detailed_towers.forEach(tower => {
                        // Tower location
                        addMarker(tower.coordinates[0], tower.coordinates[1], 'purple', 'T',
                            `<strong>Tower ${tower.id}</strong><br>${tower.coordinates[0].toFixed(6)}, ${tower.coordinates[1].toFixed(6)}<br>Distance: ${tower.distance.toFixed(1)}m`, map);
                        
                        // Tower range circle
                        addCircle(tower.coordinates[0], tower.coordinates[1], tower.distance, 'blue', map);

                        // For fullscreen map
                        if (fullscreenMap) {
                            addMarker(tower.coordinates[0], tower.coordinates[1], 'purple', 'T',
                                `<strong>Tower ${tower.id}</strong><br>${tower.coordinates[0].toFixed(6)}, ${tower.coordinates[1].toFixed(6)}<br>Distance: ${tower.distance.toFixed(1)}m`, fullscreenMap, true);
                            addCircle(tower.coordinates[0], tower.coordinates[1], tower.distance, 'blue', fullscreenMap, true);
                        }
                    });
                }

                // Fit maps to show all points
                const allPoints = [];
                allPoints.push([refLat, refLon]);
                if (gpsLat && gpsLon) allPoints.push([gpsLat, gpsLon]);
                if (data.estimated_location) allPoints.push(data.estimated_location);
                if (data.methods_results) {
                    Object.values(data.methods_results).forEach(result => {
                        allPoints.push(result.location);
                    });
                }
                if (data.detailed_towers) {
                    data.detailed_towers.forEach(tower => {
                        allPoints.push(tower.coordinates);
                    });
                }

                const bounds = L.latLngBounds(allPoints);
                map.fitBounds(bounds.pad(0.1));
                if (fullscreenMap) fullscreenMap.fitBounds(bounds.pad(0.1));
            }

            function toggleFullscreen() {
                const fullscreenElement = document.getElementById('fullscreenMode');
                if (fullscreenElement.style.display === 'none' || !fullscreenElement.style.display) {
                    // Enter fullscreen
                    fullscreenElement.style.display = 'grid';
                    if (!fullscreenMap) {
                        initializeFullscreenMap();
                    }
                    // Update fullscreen results with current data
                    if (currentData) {
                        displayFullscreenResults(currentData);
                        const refLat = parseFloat(document.getElementById('refLat').value);
                        const refLon = parseFloat(document.getElementById('refLon').value);
                        const gpsLat = document.getElementById('gpsLat').value ? parseFloat(document.getElementById('gpsLat').value) : null;
                        const gpsLon = document.getElementById('gpsLon').value ? parseFloat(document.getElementById('gpsLon').value) : null;
                        visualizeOnMap(currentData, refLat, refLon, gpsLat, gpsLon);
                    }
                    document.body.style.overflow = 'hidden';
                } else {
                    // Exit fullscreen
                    fullscreenElement.style.display = 'none';
                    document.body.style.overflow = 'auto';
                }
            }

            // Event listeners
            document.addEventListener('DOMContentLoaded', function() {
                initializeMap();
                
                document.getElementById('addTowerBtn').addEventListener('click', function() {
                    const cellId = document.getElementById('cellId').value;
                    const distance = parseFloat(document.getElementById('distance').value);
                    const azimuth = parseFloat(document.getElementById('azimuth').value);
                    const signal = parseFloat(document.getElementById('signal').value);

                    if (!cellId || isNaN(distance) || isNaN(azimuth)) {
                        alert('Please fill all tower fields');
                        return;
                    }

                    towers.push({
                        cell_id: cellId,
                        distance: distance,
                        azimuth: azimuth,
                        signal_strength: signal
                    });

                    updateTowersList();
                    
                    // Clear input fields
                    document.getElementById('cellId').value = '';
                    document.getElementById('distance').value = '';
                    document.getElementById('azimuth').value = '';
                });

                document.getElementById('calculateBtn').addEventListener('click', calculateTriangulation);
                document.getElementById('clearBtn').addEventListener('click', function() {
                    towers = [];
                    updateTowersList();
                    clearMap();
                    clearFullscreenMap();
                    document.getElementById('resultsContent').innerHTML = '<div class="text-center text-gray-500"><p>Calculate triangulation to see results</p></div>';
                    document.getElementById('fullscreenResultsContent').innerHTML = '<div class="text-center text-gray-500"><p>Calculate triangulation to see results</p></div>';
                    document.getElementById('gpsComparison').innerHTML = '';
                    document.getElementById('fullscreenGpsComparison').innerHTML = '';
                    document.getElementById('methodsResults').innerHTML = '';
                    document.getElementById('fullscreenMethodsResults').innerHTML = '';
                    document.getElementById('towerDetails').innerHTML = '';
                    document.getElementById('fullscreenTowerDetails').innerHTML = '';
                    document.getElementById('gpsLat').value = '';
                    document.getElementById('gpsLon').value = '';
                    currentData = null;
                });
                document.getElementById('loadSampleBtn').addEventListener('click', loadSampleData);
                document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);
                document.getElementById('exitFullscreenBtn').addEventListener('click', toggleFullscreen);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/calculate-triangulation")
async def calculate_triangulation(request: TriangulationRequest):
    try:
        tracker = HighPrecisionESIMTracker()
        
        # Convert to measurement format
        measurements = []
        for tower in request.tower_measurements:
            measurements.append({
                'cell_id': tower.cell_id,
                'distance': tower.distance,
                'azimuth': tower.azimuth,
                'signal_strength': tower.signal_strength,
                'environment': 'urban'
            })
        
        reference_location = (request.reference_latitude, request.reference_longitude)
        gps_location = None
        if request.gps_latitude and request.gps_longitude:
            gps_location = (request.gps_latitude, request.gps_longitude)
        
        # Perform triangulation
        result = tracker.advanced_triangulation(measurements, reference_location, gps_location)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Triangulation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)