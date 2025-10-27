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

class RoadNetworkGraph:
    #data structure to store the road coordinates
    def __init__(self):
        self.graph = defaultdict(list) #road connections
        self.node_coords = {} #store coordinates of every points
        self.road_segments = {} #road segment information
        self.segment_to_road = {} #mapping from segment to road id
        
    def add_road(self, road_id: int, coords: List[Tuple[float, float]], road_info: Dict):
        """Add a road to the graph"""
        for i in range(len(coords) - 1):
            node1 = coords[i]
            node2 = coords[i + 1]
            
            #create unique id for every coordinates
            node1_id = f"{node1[0]:.6f}_{node1[1]:.6f}"
            node2_id = f"{node2[0]:.6f}_{node2[1]:.6f}"
            
            #stores coordinates
            self.node_coords[node1_id] = node1
            self.node_coords[node2_id] = node2
            
            #calculate distance between two points
            distance = self.haversine_distance(node1, node2)
            
            #add edges to the graph
            self.graph[node1_id].append((node2_id, distance, road_id))
            self.graph[node2_id].append((node1_id, distance, road_id))
            
            #stores the info of road segment
            seg_id = f"{road_id}_{i}"
            self.road_segments[seg_id] = {
                'start': node1,
                'end': node2,
                'road_info': road_info,
                'distance': distance,
                'road_id': road_id
            }
            self.segment_to_road[seg_id] = road_id
    
    #method to calculate haversine distance between two points
    def haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Haversine distance between two points in meters"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        R = 6371000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1) #convert latitude differece to radians
        dlon = math.radians(lon2 - lon1) #convert longitude difference to radians
        
        #uses haversine formula to calculate accurate distance
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def project_point_to_segment(self, point: Tuple[float, float], seg_start: Tuple[float, float], 
                                 seg_end: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
        """
        Project a point onto a line segment and return the projected point and distance.
        Uses proper geometric projection for accuracy.
        """
        px, py = point
        ax, ay = seg_start
        bx, by = seg_end
        
        # Vector from A to B
        ab_x = bx - ax
        ab_y = by - ay
        
        # Vector from A to P
        ap_x = px - ax
        ap_y = py - ay
        
        # Calculate the parameter t (position along the segment)
        ab_len_sq = ab_x * ab_x + ab_y * ab_y
        
        if ab_len_sq == 0:
            # Segment is a point
            return seg_start, self.haversine_distance(point, seg_start)
        
        t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
        
        # Clamp t to [0, 1] to stay on the segment
        t = max(0, min(1, t))
        
        # Calculate the projected point
        proj_x = ax + t * ab_x
        proj_y = ay + t * ab_y
        projected_point = (proj_x, proj_y)
        
        # Calculate distance from point to projected point
        distance = self.haversine_distance(point, projected_point)
        
        return projected_point, distance
    
    def find_nearest_segment(self, point: Tuple[float, float]) -> Tuple[str, Tuple[float, float], float]:
        """Find the nearest road segment to a point and project the point onto it"""
        min_distance = float('inf')
        best_segment_id = None
        best_projected_point = None
        
        for seg_id, segment in self.road_segments.items():
            projected_point, distance = self.project_point_to_segment(
                point, segment['start'], segment['end']
            )
            
            if distance < min_distance:
                min_distance = distance
                best_segment_id = seg_id
                best_projected_point = projected_point
        
        return best_segment_id, best_projected_point, min_distance
    
    def find_nearest_node_to_point(self, point: Tuple[float, float]) -> Tuple[str, float]:
        """Find the nearest graph node to a point"""
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, node_coord in self.node_coords.items():
            distance = self.haversine_distance(point, node_coord)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node, min_distance
    
    def get_segment_endpoints_as_nodes(self, segment_id: str) -> Tuple[str, str]:
        """Get the node IDs for a segment's endpoints"""
        segment = self.road_segments[segment_id]
        start_coord = segment['start']
        end_coord = segment['end']
        
        start_node_id = f"{start_coord[0]:.6f}_{start_coord[1]:.6f}"
        end_node_id = f"{end_coord[0]:.6f}_{end_coord[1]:.6f}"
        
        return start_node_id, end_node_id
    
    def dijkstra_shortest_path(self, start_node: str, end_node: str) -> Tuple[List[str], float, List[int]]:
        """Find shortest path using Dijkstra's algorithm"""
        distances = {node: float('inf') for node in self.graph}
        previous = {node: None for node in self.graph}
        distances[start_node] = 0
        
        pq = [(0, start_node)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_node == end_node:
                break
                
            if current_dist > distances[current_node]:
                continue
                
            for neighbor, weight, road_id in self.graph[current_node]:
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = (current_node, road_id)
                    heapq.heappush(pq, (distance, neighbor))
        
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
        
        path.reverse()
        road_sequence.reverse()
        
        return path, distances[end_node], road_sequence
    
    def get_path_coordinates(self, path_nodes: List[str]) -> List[Tuple[float, float]]:
        """Convert node IDs back to coordinates"""
        return [self.node_coords[node_id] for node_id in path_nodes]

class CompleteRoadTracker:
    def __init__(self):
        self.road_graph = RoadNetworkGraph()
        
    def fetch_roads_in_area(self, points: List[Tuple[float, float]], padding: float = 0.02) -> List[Dict]:
        """Fetch all roads in the bounding box around the points"""
        if not points:
            return []
            
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        min_lat, max_lat = min(lats) - padding, max(lats) + padding
        min_lon, max_lon = min(lons) - padding, max(lons) + padding
        
        try:
            overpass_query = f"""
            [out:json][timeout:45];
            (
              way({min_lat},{min_lon},{max_lat},{max_lon})["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|service"];
            );
            out geom;
            """
            
            encoded_query = urllib.parse.quote(overpass_query)
            url = f"http://overpass-api.de/api/interpreter?data={encoded_query}"
            
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            data = response.json()
            
            roads = []
            for element in data.get('elements', []):
                if element['type'] == 'way' and 'geometry' in element:
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
    
    def find_complete_road_path(self, measurements: List[Tuple[float, float]]) -> Dict[str, Any]:
        """Find complete road path with accurate GPS projection"""
        if len(measurements) < 2:
            return {'error': 'Need at least 2 measurements to determine source and destination'}
        
        source_point = measurements[0]
        destination_point = measurements[-1]
        intermediate_points = measurements[1:-1]
        
        print(f"Source: {source_point}, Destination: {destination_point}")
        print(f"Intermediate points: {len(intermediate_points)}")
        
        # Fetch roads in the area
        roads = self.fetch_roads_in_area(measurements, padding=0.03)
        
        if not roads:
            return {'error': 'No roads found in the area'}
        
        print(f"Found {len(roads)} roads in the area")
        
        # Build road network graph
        self.build_road_network(roads)
        
        # Project source and destination to nearest road segments
        source_seg_id, source_projected, source_dist = self.road_graph.find_nearest_segment(source_point)
        dest_seg_id, dest_projected, dest_dist = self.road_graph.find_nearest_segment(destination_point)
        
        if not source_seg_id or not dest_seg_id:
            return {'error': 'Could not find road segments near source/destination points'}
        
        print(f"Source projection distance: {source_dist:.1f}m, Destination projection distance: {dest_dist:.1f}m")
        
        # Get the nodes for the segments containing source and destination
        source_seg_start, source_seg_end = self.road_graph.get_segment_endpoints_as_nodes(source_seg_id)
        dest_seg_start, dest_seg_end = self.road_graph.get_segment_endpoints_as_nodes(dest_seg_id)
        
        # Find shortest path between the segments (try all combinations)
        best_path = None
        best_distance = float('inf')
        best_road_sequence = None
        best_start_node = None
        best_end_node = None
        
        for start_node in [source_seg_start, source_seg_end]:
            for end_node in [dest_seg_start, dest_seg_end]:
                path_nodes, total_distance, road_sequence = self.road_graph.dijkstra_shortest_path(start_node, end_node)
                if total_distance < best_distance:
                    best_distance = total_distance
                    best_path = path_nodes
                    best_road_sequence = road_sequence
                    best_start_node = start_node
                    best_end_node = end_node
        
        if not best_path:
            return {'error': 'Could not find path between source and destination'}
        
        path_coords = self.road_graph.get_path_coordinates(best_path)
        
        # Add the projected source and destination points to the path
        complete_path = [source_projected] + path_coords + [dest_projected]
        
        print(f"Found path with {len(best_path)} nodes, total distance: {best_distance:.1f}m")
        
        # Find which roads are used in the path
        used_roads = {}
        for road_id in set(best_road_sequence):
            road = next((r for r in roads if r['id'] == road_id), None)
            if road:
                used_roads[road_id] = road
        
        # Project all measurements to nearest road segments
        projected_measurements = []
        for i, measurement in enumerate(measurements):
            seg_id, projected_point, dist = self.road_graph.find_nearest_segment(measurement)
            if seg_id:
                segment = self.road_graph.road_segments[seg_id]
                point_type = "source" if i == 0 else "destination" if i == len(measurements)-1 else f"point_{i}"
                projected_measurements.append({
                    'original': measurement,
                    'projected': projected_point,
                    'distance': dist,
                    'type': point_type,
                    'index': i,
                    'road_name': segment['road_info'].get('name', 'Unnamed Road')
                })
        
        return {
            'complete_path': complete_path,
            'total_distance': best_distance + source_dist + dest_dist,
            'used_roads': used_roads,
            'path_nodes': len(best_path),
            'projected_measurements': projected_measurements,
            'source_point': source_point,
            'destination_point': destination_point,
            'source_point_projected': source_projected,
            'destination_point_projected': dest_projected,
            'all_roads_in_area': roads,
            'intermediate_points_count': len(intermediate_points),
            'source_projection_accuracy': source_dist,
            'destination_projection_accuracy': dest_dist
        }

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
        <title>Accurate GPS Road Path Tracker</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { font-family: 'Inter', sans-serif; }
            #map { height: 700px; width: 100%; }
            .leaflet-container { border-radius: 0.5rem; }
            .custom-marker { background: transparent; border: none; }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-4">
        <div class="bg-white rounded-xl shadow-lg p-8 w-full max-w-7xl">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-2">Accurate GPS Road Path Tracker</h1>
            <p class="text-center text-gray-500 mb-6">Uses geometric projection for accurate GPS-to-road mapping</p>
            
            <div class="space-y-4 mb-6">
                <h3 class="text-lg font-semibold text-gray-700">GPS Measurements</h3>
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
                    Find Accurate Road Path
                </button>
                <button id="sampleDataBtn" class="flex-1 bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg shadow-md transition-all duration-300">
                    Load Sample Journey
                </button>
            </div>
            
            <div id="measurementsList" class="mb-4 bg-gray-50 rounded-lg p-4 max-h-40 overflow-y-auto">
                <p class="text-sm text-gray-500 text-center">No measurements added yet</p>
            </div>
            
            <div id="resultsSection" class="mt-8">
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
                    <div class="bg-green-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-green-800 mb-2">Source Accuracy</h3>
                        <p id="sourceInfo" class="text-sm text-green-600">Add points to see source</p>
                    </div>
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-blue-800 mb-2">Route Info</h3>
                        <p id="routeInfo" class="text-sm text-blue-600">Calculate to see route details</p>
                    </div>
                    <div class="bg-purple-50 p-4 rounded-lg">
                        <h3 class="font-semibold text-purple-800 mb-2">Destination Accuracy</h3>
                        <p id="destInfo" class="text-sm text-purple-600">Add points to see destination</p>
                    </div>
                </div>
                
                <h2 class="text-xl font-bold text-gray-800 mb-2">Accurate GPS Road Path Map</h2>
                <div id="mapContainer" class="w-full rounded-lg shadow-inner relative">
                    <div id="map"></div>
                    <div id="mapLoading" class="absolute inset-0 bg-gray-100 bg-opacity-90 flex items-center justify-center rounded-lg hidden">
                        <div class="text-center">
                            <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                            <p class="text-gray-600">Finding accurate road path...</p>
                        </div>
                    </div>
                </div>
                
                <div class="mt-4 bg-blue-50 p-4 rounded-lg">
                    <p class="text-sm text-blue-700">
                        <span class="font-semibold">Accuracy Features:</span><br>
                        • Geometric projection for GPS-to-road mapping<br>
                        • Shows projection accuracy for each point<br>
                        • Accurate road segment identification<br>
                        <span class="font-semibold mt-2 block">Map Legend:</span>
                        <span style="color: #FF4444">🔴 Red markers</span> - Original GPS measurements<br>
                        <span style="color: purple">🟣 Purple markers</span> - Projected points on actual roads<br>
                        <span style="color: #FF6B00">━ Orange line</span> - Complete road path<br>
                        <span style="color: blue">━ Blue line</span> - Direct GPS path
                    </p>
                </div>
            </div>
        </div>

        <script>
            const addMeasurementBtn = document.getElementById('addMeasurementBtn');
            const clearAllBtn = document.getElementById('clearAllBtn');
            const calculateBtn = document.getElementById('calculateBtn');
            const sampleDataBtn = document.getElementById('sampleDataBtn');
            const latitudeInput = document.getElementById('latitude');
            const longitudeInput = document.getElementById('longitude');
            const timestampInput = document.getElementById('timestamp');
            const measurementsList = document.getElementById('measurementsList');
            const sourceInfo = document.getElementById('sourceInfo');
            const destInfo = document.getElementById('destInfo');
            const routeInfo = document.getElementById('routeInfo');
            const mapLoading = document.getElementById('mapLoading');
            
            let measurements = [];
            let map = null;
            let markers = [];
            let polylines = [];
            let roadLayers = [];

            function initializeMap() {
                map = L.map('map').setView([28.6139, 77.2090], 13);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors',
                    maxZoom: 18
                }).addTo(map);
            }

            document.addEventListener('DOMContentLoaded', function() {
                initializeMap();
                loadSampleData();
            });

            addMeasurementBtn.addEventListener('click', () => {
                const latitude = parseFloat(latitudeInput.value);
                const longitude = parseFloat(longitudeInput.value);
                const timestamp = parseInt(timestampInput.value);
                
                if (isNaN(latitude) || isNaN(longitude) || isNaN(timestamp)) {
                    alert("Please enter valid numbers for all fields.");
                    return;
                }
                
                measurements.push({ latitude, longitude, timestamp });
                updateMeasurementsList();
                
                latitudeInput.value = '';
                longitudeInput.value = '';
                timestampInput.value = '';
                
                if (measurements.length === 1) {
                    map.setView([latitude, longitude], 15);
                }
            });

            clearAllBtn.addEventListener('click', () => {
                measurements = [];
                updateMeasurementsList();
                clearMap();
                sourceInfo.textContent = 'Add points to see source';
                destInfo.textContent = 'Add points to see destination';
                routeInfo.textContent = 'Calculate to see route details';
            });

            function updateMeasurementsList() {
                if (measurements.length === 0) {
                    measurementsList.innerHTML = '<p class="text-sm text-gray-500 text-center">No measurements added yet</p>';
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
            };

            function clearMap() {
                markers.forEach(marker => map.removeLayer(marker));
                polylines.forEach(polyline => map.removeLayer(polyline));
                roadLayers.forEach(layer => map.removeLayer(layer));
                
                markers = [];
                polylines = [];
                roadLayers = [];
            }

            function addMarker(lat, lng, color, label, popupText) {
                const icon = L.divIcon({
                    html: `<div style="background-color: ${color}; width: 28px; height: 28px; border-radius: 50%; border: 3px solid white; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.3);">${label}</div>`,
                    className: 'custom-marker',
                    iconSize: [32, 32],
                    iconAnchor: [16, 16]
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

            function plotCompleteRoadPath(roadData, measurements) {
                clearMap();
                
                // Plot all roads in light gray
                if (roadData.all_roads_in_area) {
                    roadData.all_roads_in_area.forEach(road => {
                        const roadPoints = road.coords.map(coord => [coord[0], coord[1]]);
                        const roadPolyline = addPolyline(roadPoints, 'gray', 2);
                        roadLayers.push(roadPolyline);
                    });
                }
                
                // Highlight the complete road path in orange
                if (roadData.complete_path && roadData.complete_path.length > 0) {
                    const completePathPoints = roadData.complete_path.map(coord => [coord[0], coord[1]]);
                    const mainPath = addPolyline(completePathPoints, '#FF6B00', 8);
                    roadLayers.push(mainPath);
                    
                    mainPath.bindPopup(`
                        <strong>Complete Road Path</strong><br>
                        Distance: ${(roadData.total_distance / 1000).toFixed(2)} km<br>
                        Path Nodes: ${roadData.path_nodes}
                    `);
                }
                
                // Plot measurement points
                const measurementPoints = [];
                measurements.forEach((measurement, index) => {
                    const point = [measurement.latitude, measurement.longitude];
                    measurementPoints.push(point);
                    
                    const markerColor = index === 0 ? '#FF4444' : index === measurements.length - 1 ? '#44FF44' : 'red';
                    const markerLabel = index === 0 ? 'S' : index === measurements.length - 1 ? 'D' : (index + 1);
                    
                    addMarker(
                        measurement.latitude, measurement.longitude, markerColor, markerLabel,
                        `GPS Point ${index + 1}<br>Lat: ${measurement.latitude.toFixed(6)}<br>Lon: ${measurement.longitude.toFixed(6)}`
                    );
                });
                
                // Plot projected measurements with accuracy info
                if (roadData.projected_measurements) {
                    roadData.projected_measurements.forEach((proj) => {
                        const label = proj.type === 'source' ? 'SP' : proj.type === 'destination' ? 'DP' : 'P' + proj.index;
                        addMarker(
                            proj.projected[0], proj.projected[1], 'purple', label,
                            `Projected ${proj.type.replace('_', ' ')}<br>Road: ${proj.road_name}<br>Accuracy: ${proj.distance.toFixed(1)}m from GPS`
                        );
                        
                        // Draw line from GPS point to projected point
                        addPolyline(
                            [[proj.original[0], proj.original[1]], [proj.projected[0], proj.projected[1]]],
                            'orange', 2, '5, 5'
                        );
                    });
                }
                
                // Draw direct path between measurements
                if (measurementPoints.length > 1) {
                    addPolyline(measurementPoints, 'blue', 4);
                }
                
                // Fit map to show the complete path
                if (roadData.complete_path && roadData.complete_path.length > 0) {
                    const pathBounds = L.latLngBounds(roadData.complete_path.map(coord => [coord[0], coord[1]]));
                    map.fitBounds(pathBounds.pad(0.1));
                }
                
                // Update info panels with accuracy metrics
                if (roadData.source_projection_accuracy !== undefined) {
                    sourceInfo.innerHTML = `
                        <strong>Projection Accuracy:</strong> ${roadData.source_projection_accuracy.toFixed(1)}m<br>
                        <strong>GPS:</strong> ${roadData.source_point[0].toFixed(6)}, ${roadData.source_point[1].toFixed(6)}<br>
                        <strong>Road:</strong> ${roadData.source_point_projected[0].toFixed(6)}, ${roadData.source_point_projected[1].toFixed(6)}
                    `;
                }
                if (roadData.destination_projection_accuracy !== undefined) {
                    destInfo.innerHTML = `
                        <strong>Projection Accuracy:</strong> ${roadData.destination_projection_accuracy.toFixed(1)}m<br>
                        <strong>GPS:</strong> ${roadData.destination_point[0].toFixed(6)}, ${roadData.destination_point[1].toFixed(6)}<br>
                        <strong>Road:</strong> ${roadData.destination_point_projected[0].toFixed(6)}, ${roadData.destination_point_projected[1].toFixed(6)}
                    `;
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
                    alert("Please add at least two points to determine source and destination.");
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
                    
                    plotCompleteRoadPath(data, measurements);
                    mapLoading.style.display = 'none';
                    
                } catch (error) {
                    alert("Path Finding Error: " + error.message);
                    mapLoading.style.display = 'none';
                }
            });

            function loadSampleData() {
                measurements = [
                    { latitude: 28.574444, longitude: 77.384167, timestamp: 1721721000000},
                    {latitude: 28.568611, longitude: 77.391667, timestamp: 1721721060000},
                    {latitude: 28.560833, longitude: 77.367222, timestamp: 1721721120000},
                    {latitude: 28.564167, longitude: 77.365278, timestamp: 1721721180000},
                    {latitude: 28.561389, longitude: 77.343056, timestamp: 1721721240000},
                    {latitude: 28.548889, longitude: 77.306111, timestamp: 1721721300000},
                    {latitude: 28.547778, longitude: 77.264444, timestamp: 1721721360000},
                    {latitude: 28.546389, longitude: 77.224722, timestamp: 1721721420000},
                    {latitude: 28.570833, longitude: 77.165000, timestamp: 1721721480000},
                    {latitude: 28.537222, longitude: 77.111944, timestamp: 1721721540000},
                    {latitude: 28.516667, longitude: 77.093611, timestamp: 1721721600000}
                ];
                
                updateMeasurementsList();
                map.setView([28.55, 77.25], 11);
            }

            sampleDataBtn.addEventListener('click', loadSampleData);

            [latitudeInput, longitudeInput, timestampInput].forEach(input => {
                input.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        addMeasurementBtn.click();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/find-complete-road-path")
async def find_complete_road_path(data: MeasurementsData):
    try:
        measurements = [(m.latitude, m.longitude) for m in data.measurements]
        
        tracker = CompleteRoadTracker()
        road_data = tracker.find_complete_road_path(measurements)
        
        return road_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)