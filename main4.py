import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any
import urllib.parse
from collections import defaultdict, deque
import math
import heapq

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

class RoadNetworkGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.node_coords = {}
        self.road_segments = {}
        
    def add_road(self, road_id: int, coords: List[Tuple[float, float]], road_info: Dict):
        """Add a road to the graph"""
        for i in range(len(coords) - 1):
            node1 = coords[i]
            node2 = coords[i + 1]
            
            # Create node IDs
            node1_id = f"{node1[0]:.6f}_{node1[1]:.6f}"
            node2_id = f"{node2[0]:.6f}_{node2[1]:.6f}"
            
            # Store coordinates
            self.node_coords[node1_id] = node1
            self.node_coords[node2_id] = node2
            
            # Calculate distance
            distance = self.haversine_distance(node1, node2)
            
            # Add bidirectional edges
            self.graph[node1_id].append((node2_id, distance, road_id))
            self.graph[node2_id].append((node1_id, distance, road_id))
            
            # Store road segment info
            seg_id = f"{road_id}_{i}"
            self.road_segments[seg_id] = {
                'start': node1,
                'end': node2,
                'road_info': road_info,
                'distance': distance
            }
    
    def haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two points in meters"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c #final distance in meters
    
    def find_nearest_node(self, point: Tuple[float, float]) -> Tuple[str, float]:
        """Find the nearest graph node to a point"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_coord in self.node_coords.items():
            distance = self.haversine_distance(point, node_coord)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node, min_distance
    
    def dijkstra_shortest_path(self, start_node: str, end_node: str) -> Tuple[List[str], float]:
        """Find shortest path using Dijkstra's algorithm"""
        distances = {node: float('inf') for node in self.graph}
        previous = {node: None for node in self.graph} #to track path
        distances[start_node] = 0 #start node dist is 0
        
        pq = [(0, start_node)] #priority queue starts
        
        while pq: #until queue is empty
            current_dist, current_node = heapq.heappop(pq) #get node with smallest distance
            
            if current_node == end_node: #if reached end node
                break
                
            if current_dist > distances[current_node]: #if found better path
                continue
            
            #check the neighbour node of the current node
            for neighbor, weight, road_id in self.graph[current_node]:
                distance = current_dist + weight #calculates new distance
                
                if distance < distances[neighbor]: #if new distance is better
                    distances[neighbor] = distance
                    previous[neighbor] = (current_node, road_id)
                    heapq.heappush(pq, (distance, neighbor)) #add in priority queue
        
        # Reconstruct path
        path = []
        current = end_node
        road_sequence = []
        
        while current is not None:
            path.append(current)
            if previous[current] is not None:
                prev_node, road_id = previous[current]
                road_sequence.append(road_id)
            current = previous[current][0] if previous[current] else None
        
        path.reverse() #start to end path
        road_sequence.reverse()
        
        return path, distances[end_node], road_sequence
    
    def get_path_coordinates(self, path_nodes: List[str]) -> List[Tuple[float, float]]:
        """Convert node IDs back to coordinates"""
        return [self.node_coords[node_id] for node_id in path_nodes]

class CompleteRoadTracker:
    def __init__(self):
        self.road_graph = RoadNetworkGraph() #create object of the road network
    
    #bounding box calculates the roads in the area
    def fetch_roads_in_area(self, points: List[Tuple[float, float]], padding: float = 0.02) -> List[Dict]:
        """Fetch all roads in the bounding box around the points"""
        if not points:
            return []
            
        # Calculate bounding box
        lats = [p[0] for p in points] #all latitudes
        lons = [p[1] for p in points] #all longitudes
        
        min_lat, max_lat = min(lats) - padding, max(lats) + padding
        min_lon, max_lon = min(lons) - padding, max(lons) + padding
        
        try:
            # Query Overpass API for roads in the bounding box
            overpass_query = f"""
            [out:json][timeout:45];
            (
              way({min_lat},{min_lon},{max_lat},{max_lon})["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"];
            );
            out geom;
            """
            
            encoded_query = urllib.parse.quote(overpass_query) #encodes the URL
            url = f"http://overpass-api.de/api/interpreter?data={encoded_query}"
            
            #fetch data from the API
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.json()
            
            roads = []
            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
                    #fetch road coordinates
                    road_coords = [(node['lat'], node['lon']) for node in element['geometry']]
                    road_info = {
                        'id': element['id'],
                        'coords': road_coords,
                        'tags': element.get('tags', {}),
                        'type': element.get('tags', {}).get('highway', 'road'),
                        'name': element.get('tags', {}).get('name', 'Unnamed Road')
                    }
                    roads.append(road_info)
            
            return roads
            
        except Exception as e:
            print(f"Error fetching roads: {e}")
            return []
    
    def build_road_network(self, roads: List[Dict]):
        """Build a graph from the road data"""
        for road in roads:
            self.road_graph.add_road(road['id'], road['coords'], road)
    
    #main path finding method
    def find_complete_road_path(self, measurements: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Find complete road path from first measurement (source) to last measurement (destination)"""
        if len(measurements) < 2:
            return {'error': 'Need at least 2 measurements to determine source and destination'}
        
        # First measurement is source, last measurement is destination
        source_point = measurements[0]
        destination_point = measurements[-1]
        intermediate_points = measurements[1:-1]  # Points between source and destination
        
        print(f"Source: {source_point}, Destination: {destination_point}")
        print(f"Intermediate points: {len(intermediate_points)}")
        
        # Fetch roads in the area
        all_points = measurements
        roads = self.fetch_roads_in_area(all_points, padding=0.03)
        
        if not roads:
            return {'error': 'No roads found in the area'}
        
        print(f"Found {len(roads)} roads in the area")
        
        # Build road network graph
        self.build_road_network(roads)
        
        # Find nearest nodes to source and destination points
        source_node, source_dist = self.road_graph.find_nearest_node(source_point)
        destination_node, dest_dist = self.road_graph.find_nearest_node(destination_point)
        
        if not source_node or not destination_node:
            return {'error': 'Could not find road network nodes near source/destination points'}
        
        print(f"Source node distance: {source_dist:.1f}m, Destination node distance: {dest_dist:.1f}m")
        
        # Find shortest path
        path_nodes, total_distance, road_sequence = self.road_graph.dijkstra_shortest_path(source_node, destination_node)
        path_coords = self.road_graph.get_path_coordinates(path_nodes)
        
        print(f"Found path with {len(path_nodes)} nodes, total distance: {total_distance:.1f}m")
        
        # Find which roads are used in the path
        used_roads = {}
        for road_id in set(road_sequence):
            road = next((r for r in roads if r['id'] == road_id), None)
            if road:
                used_roads[road_id] = road
        
        # Project all measurements to the path
        projected_measurements = []
        for i, measurement in enumerate(measurements):
            nearest_node, dist = self.road_graph.find_nearest_node(measurement)
            if nearest_node:
                projected_point = self.road_graph.node_coords[nearest_node]
                point_type = "source" if i == 0 else "destination" if i == len(measurements)-1 else f"point_{i}"
                projected_measurements.append({
                    'original': measurement,
                    'projected': projected_point,
                    'distance': dist,
                    'type': point_type,
                    'index': i
                })
        #return the final results
        return {
            'complete_path': path_coords,
            'total_distance': total_distance,
            'used_roads': used_roads,
            'path_nodes': len(path_nodes),
            'projected_measurements': projected_measurements,
            'source_point': source_point,
            'destination_point': destination_point,
            'source_point_projected': self.road_graph.node_coords[source_node],
            'destination_point_projected': self.road_graph.node_coords[destination_node],
            'all_roads_in_area': roads,
            'intermediate_points_count': len(intermediate_points)
        }

class Measurement(BaseModel):
    latitude: float #GPS latitude
    longitude: float #GPS longitude
    timestamp: int #timesamp when the reading is taken

class MeasurementsData(BaseModel):
    measurements: List[Measurement] = Field(..., min_items=2) #atleast 2 measurements required

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    #returns the main web page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Complete Road Path Tracker</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            #map { height: 700px; width: 100%; }
            .leaflet-container { border-radius: 0.5rem; }
            .custom-marker { background: transparent; border: none; }
            .complete-road-path { stroke-width: 8; opacity: 0.9; }
            .other-roads { stroke-width: 3; opacity: 0.3; }
            .measurement-path { stroke-width: 4; opacity: 0.7; }
            .info-panel { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
        <div class="bg-white rounded-xl shadow-lg p-8 w-full max-w-7xl">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-2">Complete Road Path Tracker</h1>
            <p class="text-center text-gray-500 mb-6">Automatically detects source (first point) and destination (last point) from your measurements.</p>
            
            <div class="space-y-4 mb-6">
                <h3 class="text-lg font-semibold text-gray-700">Car Measurements</h3>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-2">
                    <div>
                        <label for="latitude" class="block text-sm font-medium text-gray-700">Latitude</label>
                        <input type="number" id="latitude" placeholder="28.6139" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 p-2 border">
                    </div>
                    <div>
                        <label for="longitude" class="block text-sm font-medium text-gray-700">Longitude</label>
                        <input type="number" id="longitude" placeholder="77.2090" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 p-2 border">
                    </div>
                    <div>
                        <label for="timestamp" class="block text-sm font-medium text-gray-700">Timestamp (ms)</label>
                        <input type="number" id="timestamp" placeholder="1721721000000" step="1" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 p-2 border">
                    </div>
                    <div class="flex items-end">
                        <button id="addMeasurementBtn" class="w-full bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300">
                            Add Point
                        </button>
                    </div>
                </div>
            </div>

            <div class="flex space-x-4 mb-4">
                <button id="clearAllBtn" class="flex-1 bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300">
                    Clear All
                </button>
                <button id="calculateBtn" class="flex-1 bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300">
                    Find Complete Road Path
                </button>
            </div>
            
            <div id="measurementsList" class="mb-4 bg-gray-50 rounded-lg p-4 max-h-40 overflow-y-auto">
                <p class="text-sm text-gray-500 text-center">No measurements added yet. First point = Source, Last point = Destination</p>
            </div>
            
            <div id="resultsSection" class="mt-8">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div class="bg-green-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-green-800 mb-2">Source (First Point)</h3>
                        <p id="sourceInfo" class="text-sm text-green-600">Add points to see source</p>
                    </div>
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-blue-800 mb-2">Route Info</h3>
                        <p id="routeInfo" class="text-sm text-blue-600">Calculate to see route details</p>
                    </div>
                    <div class="bg-purple-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-purple-800 mb-2">Destination (Last Point)</h3>
                        <p id="destInfo" class="text-sm text-purple-600">Add points to see destination</p>
                    </div>
                </div>
                
                <h2 class="text-xl font-bold text-gray-800 mb-2">Complete Road Path Map</h2>
                <div id="mapContainer" class="w-full rounded-lg shadow-inner relative">
                    <div id="map"></div>
                    <div id="mapLoading" class="absolute inset-0 bg-gray-100 bg-opacity-90 flex items-center justify-center rounded-lg hidden">
                        <div class="text-center">
                            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                            <p class="text-gray-600">Finding complete road path from source to destination...</p>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4 bg-blue-50 p-4 rounded-lg">
                    <p class="text-sm text-blue-700">
                        <span class="font-semibold">Map Legend:</span><br>
                        <span style="color: #FF4444">🏁 Source</span> - First measurement point<br>
                        <span style="color: #44FF44">🎯 Destination</span> - Last measurement point<br>
                        <span style="color: #FF6B00">🟧 Complete Road Path</span> - Full route from source to destination<br>
                        <span style="color: red">● Red markers</span> - Car measurement points<br>
                        <span style="color: purple">● Purple markers</span> - Projected points on road<br>
                        <span style="color: blue">━ Blue line</span> - Direct path between measurements<br>
                        <span style="color: gray">━ Gray lines</span> - Other roads in area
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
            const clearAllBtn = document.getElementById('clearAllBtn');
            const calculateBtn = document.getElementById('calculateBtn');
            const latitudeInput = document.getElementById('latitude');
            const longitudeInput = document.getElementById('longitude');
            const timestampInput = document.getElementById('timestamp');
            const measurementsList = document.getElementById('measurementsList');
            const sourceInfo = document.getElementById('sourceInfo');
            const destInfo = document.getElementById('destInfo');
            const routeInfo = document.getElementById('routeInfo');
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
            let roadLayers = [];

            function initializeMap() {
                const defaultCenter = [28.6139, 77.2090];
                
                map = L.map('map').setView(defaultCenter, 13);
                
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 18
                }).addTo(map);
            }

            document.addEventListener('DOMContentLoaded', function() {
                initializeMap();
                loadSampleData();
            });

            function showAlert(title, message) {
                alertTitle.textContent = title;
                alertMessage.textContent = message;
                alertModal.classList.remove('hidden');
            }

            alertCloseBtn.addEventListener('click', () => {
                alertModal.classList.add('hidden');
            });

            function updateSourceDestinationInfo() {
                if (measurements.length > 0) {
                    const source = measurements[0];
                    const destination = measurements[measurements.length - 1];
                    
                    sourceInfo.textContent = `Point 1: ${source.latitude.toFixed(6)}, ${source.longitude.toFixed(6)}`;
                    destInfo.textContent = `Point ${measurements.length}: ${destination.latitude.toFixed(6)}, ${destination.longitude.toFixed(6)}`;
                    
                    if (measurements.length > 1) {
                        routeInfo.innerHTML = `
                            <strong>Total Points:</strong> ${measurements.length}<br>
                            <strong>Intermediate:</strong> ${measurements.length - 2} points<br>
                            <strong>Ready to calculate path</strong>
                        `;
                    }
                } else {
                    sourceInfo.textContent = 'Add points to see source';
                    destInfo.textContent = 'Add points to see destination';
                    routeInfo.textContent = 'Calculate to see route details';
                }
            }

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
                updateMeasurementsList();
                updateSourceDestinationInfo();
                
                latitudeInput.value = '';
                longitudeInput.value = '';
                timestampInput.value = '';
                
                // Auto-center map on first point
                if (measurements.length === 1) {
                    map.setView([latitude, longitude], 15);
                }
            });

            clearAllBtn.addEventListener('click', () => {
                measurements = [];
                updateMeasurementsList();
                updateSourceDestinationInfo();
                clearMap();
            });

            function updateMeasurementsList() {
                if (measurements.length === 0) {
                    measurementsList.innerHTML = '<p class="text-sm text-gray-500 text-center">No measurements added yet. First point = Source, Last point = Destination</p>';
                } else {
                    measurementsList.innerHTML = '';
                    measurements.forEach((measurement, index) => {
                        const listItem = document.createElement('div');
                        const isSource = index === 0;
                        const isDestination = index === measurements.length - 1;
                        let badge = '';
                        
                        if (isSource) badge = '<span class="bg-green-100 text-green-800 text-xs px-2 py-1 rounded ml-2">SOURCE</span>';
                        if (isDestination) badge = '<span class="bg-purple-100 text-purple-800 text-xs px-2 py-1 rounded ml-2">DESTINATION</span>';
                        
                        listItem.className = "text-gray-700 text-sm p-2 bg-white rounded-md mb-2 shadow-sm flex justify-between items-center";
                        listItem.innerHTML = `
                            <div class="flex items-center">
                                <span>📍 Point ${index + 1}: ${measurement.latitude.toFixed(6)}, ${measurement.longitude.toFixed(6)}</span>
                                ${badge}
                            </div>
                            <button onclick="removeMeasurement(${index})" class="text-red-500 hover:text-red-700 text-xs">Remove</button>
                        `;
                        measurementsList.appendChild(listItem);
                    });
                }
            }

            window.removeMeasurement = function(index) {
                measurements.splice(index, 1);
                updateMeasurementsList();
                updateSourceDestinationInfo();
            };

            function clearMap() {
                markers.forEach(marker => map.removeLayer(marker));
                polylines.forEach(polyline => map.removeLayer(polyline));
                roadLayers.forEach(layer => map.removeLayer(layer));
                
                markers = [];
                polylines = [];
                roadLayers = [];
            }

            function addMarker(lat, lng, color, label, popupText, isPermanent = false) {
                const icon = L.divIcon({
                    html: `<div style="background-color: ${color}; width: 28px; height: 28px; border-radius: 50%; border: 3px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">${label}</div>`,
                    className: 'custom-marker' + (isPermanent ? ' permanent-marker' : ''),
                    iconSize: [32, 32],
                    iconAnchor: [16, 16]
                });
                
                const marker = L.marker([lat, lng], { icon: icon })
                    .addTo(map)
                    .bindPopup(popupText);
                
                if (!isPermanent) {
                    markers.push(marker);
                }
                return marker;
            }

            function addPolyline(points, color, weight, dashArray = null, className = '') {
                const polyline = L.polyline(points, {
                    color: color,
                    weight: weight,
                    opacity: 0.8,
                    dashArray: dashArray,
                    className: className
                }).addTo(map);
                
                polylines.push(polyline);
                return polyline;
            }

            function plotCompleteRoadPath(roadData, measurements) {
                clearMap();
                
                // Plot all roads in light gray
                if (roadData.all_roads_in_area) {
                    roadData.all_roads_in_area.forEach(road => {
                        const roadPoints = road.coords.map(coord => [coord[0], coord[1]]);
                        const roadPolyline = addPolyline(roadPoints, 'gray', 2, null, 'other-roads');
                        roadLayers.push(roadPolyline);
                    });
                }
                
                // Highlight the complete road path in orange
                if (roadData.complete_path && roadData.complete_path.length > 0) {
                    const completePathPoints = roadData.complete_path.map(coord => [coord[0], coord[1]]);
                    const mainPath = addPolyline(completePathPoints, '#FF6B00', 8, null, 'complete-road-path');
                    roadLayers.push(mainPath);
                    
                    // Add popup with route info
                    mainPath.bindPopup(`
                        <strong>Complete Road Path</strong><br>
                        Distance: ${(roadData.total_distance / 1000).toFixed(2)} km<br>
                        Path Nodes: ${roadData.path_nodes}<br>
                        Roads Used: ${Object.keys(roadData.used_roads).length}
                    `);
                }
                
                // Plot source and destination points
                if (roadData.source_point) {
                    const sourceMarker = addMarker(
                        roadData.source_point[0], roadData.source_point[1], '#FF4444', 'S', 
                        `<strong>Source Point (First Measurement)</strong><br>Lat: ${roadData.source_point[0].toFixed(6)}<br>Lon: ${roadData.source_point[1].toFixed(6)}`, true
                    );
                }
                
                if (roadData.destination_point) {
                    const destMarker = addMarker(
                        roadData.destination_point[0], roadData.destination_point[1], '#44FF44', 'D', 
                        `<strong>Destination Point (Last Measurement)</strong><br>Lat: ${roadData.destination_point[0].toFixed(6)}<br>Lon: ${roadData.destination_point[1].toFixed(6)}`, true
                    );
                }
                
                // Plot projected source and destination
                if (roadData.source_point_projected) {
                    addMarker(
                        roadData.source_point_projected[0], roadData.source_point_projected[1], '#FF8888', 'SP', 
                        'Projected Source Point on Road'
                    );
                }
                
                if (roadData.destination_point_projected) {
                    addMarker(
                        roadData.destination_point_projected[0], roadData.destination_point_projected[1], '#88FF88', 'DP', 
                        'Projected Destination Point on Road'
                    );
                }
                
                // Plot measurement points and their projections
                const measurementPoints = [];
                measurements.forEach((measurement, index) => {
                    const point = [measurement.latitude, measurement.longitude];
                    measurementPoints.push(point);
                    
                    // Original measurement
                    const markerColor = index === 0 ? '#FF4444' : index === measurements.length - 1 ? '#44FF44' : 'red';
                    const markerLabel = index === 0 ? 'S' : index === measurements.length - 1 ? 'D' : (index + 1);
                    
                    addMarker(
                        measurement.latitude, measurement.longitude, markerColor, markerLabel,
                        `${index === 0 ? 'SOURCE' : index === measurements.length - 1 ? 'DESTINATION' : 'Point ' + (index + 1)}<br>Lat: ${measurement.latitude.toFixed(6)}<br>Lon: ${measurement.longitude.toFixed(6)}`
                    );
                });
                
                // Plot projected measurements
                if (roadData.projected_measurements) {
                    roadData.projected_measurements.forEach((proj) => {
                        const label = proj.type === 'source' ? 'SP' : proj.type === 'destination' ? 'DP' : 'P' + proj.index;
                        addMarker(
                            proj.projected[0], proj.projected[1], 'purple', label,
                            `Projected ${proj.type.replace('_', ' ')}<br>Distance from actual: ${proj.distance.toFixed(1)}m`
                        );
                    });
                }
                
                // Draw direct path between measurements
                if (measurementPoints.length > 1) {
                    addPolyline(measurementPoints, 'blue', 4, null, 'measurement-path');
                }
                
                // Fit map to show the complete path
                if (roadData.complete_path && roadData.complete_path.length > 0) {
                    const pathBounds = L.latLngBounds(roadData.complete_path.map(coord => [coord[0], coord[1]]));
                    map.fitBounds(pathBounds.pad(0.1));
                }
                
                // Update info panels
                if (roadData.source_point) {
                    sourceInfo.textContent = `${roadData.source_point[0].toFixed(6)}, ${roadData.source_point[1].toFixed(6)}`;
                }
                if (roadData.destination_point) {
                    destInfo.textContent = `${roadData.destination_point[0].toFixed(6)}, ${roadData.destination_point[1].toFixed(6)}`;
                }
                if (roadData.total_distance) {
                    routeInfo.innerHTML = `
                        <strong>Distance:</strong> ${(roadData.total_distance / 1000).toFixed(2)} km<br>
                        <strong>Path Nodes:</strong> ${roadData.path_nodes}<br>
                        <strong>Roads Used:</strong> ${Object.keys(roadData.used_roads).length}<br>
                        <strong>Points:</strong> ${measurements.length} total
                    `;
                }
            }

            calculateBtn.addEventListener('click', async () => {
                if (measurements.length < 2) {
                    showAlert("Not Enough Points", "Please add at least two points to determine source and destination.");
                    return;
                }
                
                mapLoading.style.display = 'flex';
                
                try {
                    const response = await fetch('/find-complete-road-path', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ measurements })
                    });
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        throw new Error(`Server error: ${response.status} - ${errorText}`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Plot the complete road path
                    plotCompleteRoadPath(data, measurements);
                    
                    mapLoading.style.display = 'none';
                    
                } catch (error) {
                    showAlert("Path Finding Error", error.message);
                    mapLoading.style.display = 'none';
                }
            });

            function loadSampleData() {
                measurements = [
                    {latitude: 28.574444, longitude: 77.384167, timestamp: 1721721000000},
                    {latitude: 28.568611, longitude: 77.391667, timestamp: 1721721000000},
                    {latitude: 28.560833, longitude: 77.367222, timestamp: 1721721000000},
                    {latitude: 28.564167, longitude: 77.365278, timestamp: 1721721000000},
                    {latitude: 28.561389, longitude: 77.343056, timestamp: 1721721000000},
                    {latitude: 28.548889, longitude: 77.306111, timestamp: 1721721000000},
                    {latitude: 28.547778, longitude: 77.264444, timestamp: 1721721000000},
                    {latitude: 28.546389, longitude: 77.224722, timestamp: 1721721000000},
                    {latitude: 28.570833, longitude: 77.165000, timestamp: 1721721000000},
                    {latitude: 28.537222, longitude: 77.111944, timestamp: 1721721000000},
                    {latitude: 28.516667, longitude: 77.093611, timestamp: 1721721000000}

                ];
                
                updateMeasurementsList();
                updateSourceDestinationInfo();
                
                // Set map view to sample area
                map.setView([28.6177, 77.2150], 14);
            }

            // Handle Enter key in input fields
            [latitudeInput, longitudeInput, timestampInput].forEach(input => {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        addMeasurementBtn.click();
                    }
                });
            });

            // Add sample data button
            const sampleDataBtn = document.createElement('button');
            sampleDataBtn.textContent = 'Load Sample Journey';
            sampleDataBtn.className = 'w-full bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300 mt-2';
            sampleDataBtn.onclick = loadSampleData;
            calculateBtn.parentNode.insertBefore(sampleDataBtn, calculateBtn.nextSibling);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/find-complete-road-path")
async def find_complete_road_path(data: MeasurementsData):
    try:
        measurements = [(m.latitude, m.longitude) for m in data.measurements]
        
        # Use road tracker to find complete path (automatically uses first as source, last as destination)
        tracker = CompleteRoadTracker()
        road_data = tracker.find_complete_road_path(measurements)
        
        return road_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
