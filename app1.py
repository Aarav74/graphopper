from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
import requests
import folium
import polyline
import json
import time
import math
from typing import List, Tuple, Dict, Any

# --- GraphHopper Class ---
class GraphHopper:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://graphhopper.com/api/1/route"
        self.geocode_url = "https://graphhopper.com/api/1/geocode"
        self.map_matching_url = "https://graphhopper.com/api/1/match"

    def address_to_latlong(self, address: str) -> Tuple[float, float]:
        params = {
            "q": address,
            "locale": "en",
            "limit": 1,
            "key": self.api_key
        }
        try:
            response = requests.get(self.geocode_url, params=params)
            response.raise_for_status()
            data = response.json()
            if data and 'hits' in data and len(data['hits']) > 0:
                point = data['hits'][0]['point']
                return (point['lat'], point['lng'])
            else:
                print(f"Geocoding: No hits found for address '{address}'. Full response:\n{json.dumps(data, indent=2)}")
                raise ValueError(f"Could not find coordinates for address: {address}")
        except requests.exceptions.RequestException as e:
            print(f"Geocoding API request failed for '{address}': {e}")
            raise HTTPException(status_code=500, detail=f"Geocoding API error: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _make_route_request(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        point_params = [f"{p[0]},{p[1]}" for p in points]
        params = {
            "point": point_params,
            "vehicle": "car",
            "locale": "en",
            "instructions": "true",
            "calc_points": "true",
            "points_encoded": "true",
            "key": self.api_key
        }
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Routing API request failed for points {points}: {e}")
            raise HTTPException(status_code=500, detail=f"Routing API error: {e}")

    def route(self, points: List[Tuple[float, float]]) -> Dict[str, Any]:
        return self._make_route_request(points)

    def snap_to_road(self, gps_points: List[Tuple[float, float]], timestamps: List[int] = None) -> Dict[str, Any]:
        if not gps_points:
            raise ValueError("No GPS points provided for map matching.")

        points_for_api = [[lon, lat] for lat, lon in gps_points] # API expects lon,lat

        payload = {
            "points": points_for_api,
            "vehicle": "car",
            "details": ["street_name", "time", "distance", "road_class"],
            "points_encoded": True
        }
        if timestamps:
            payload["timestamps"] = timestamps

        headers = {
            "Content-Type": "application/json"
        }
        params = {"key": self.api_key}

        try:
            response = requests.post(self.map_matching_url, headers=headers, params=params, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Map Matching API request failed: {e}")
            raise HTTPException(status_code=500, detail=f"Map Matching API error: {e}")

# --- Helper Functions ---
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters using Haversine formula"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

def trilaterate(tower_coords: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Estimate position using trilateration with 3 or more towers"""
    if len(tower_coords) < 3:
        raise ValueError("At least 3 towers required for trilateration")
    
    # Simple centroid approach for demo (replace with proper trilateration in production)
    lats = [t[0] for t in tower_coords]
    lons = [t[1] for t in tower_coords]
    return (sum(lats)/len(lats), sum(lons)/len(lons))

# --- FastAPI Application ---
app = FastAPI()

GRAPHHOPPER_API_KEY = "add4c752-8787-4a6b-ae81-6c8a357504b4"
mapper = GraphHopper(GRAPHHOPPER_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle Location Finder</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background-color: #f4f7f6; }
            .container { max-width: 900px; margin: auto; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1, h2 { color: #007bff; }
            .section { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px solid #eee; }
            input, textarea, button { width: 100%; padding: 10px; margin: 8px 0; box-sizing: border-box; }
            textarea { height: 120px; }
            button { background-color: #007bff; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #output { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            iframe { width: 100%; height: 500px; border: 1px solid #ddd; margin-top: 15px; }
            .tower-input { display: flex; gap: 10px; margin-bottom: 10px; }
            .tower-input input { flex: 1; }
        </style>
        <script>
            function addTowerField() {
                const container = document.getElementById('towerFields');
                const div = document.createElement('div');
                div.className = 'tower-input';
                div.innerHTML = `
                    <input type="text" placeholder="Latitude" class="tower-lat">
                    <input type="text" placeholder="Longitude" class="tower-lon">
                    <input type="text" placeholder="Distance (meters)" class="tower-dist">
                `;
                container.appendChild(div);
            }

            async function findCarLocation() {
                const origin = document.getElementById('origin').value;
                const destination = document.getElementById('destination').value;
                const outputDiv = document.getElementById('output');
                const mapFrame = document.getElementById('mapFrame');
                
                outputDiv.innerHTML = 'Calculating...';
                mapFrame.srcdoc = '';

                // Collect tower data
                const towers = [];
                document.querySelectorAll('.tower-input').forEach(div => {
                    const lat = div.querySelector('.tower-lat').value;
                    const lon = div.querySelector('.tower-lon').value;
                    const dist = div.querySelector('.tower-dist').value;
                    if (lat && lon && dist) {
                        towers.push(`${lat},${lon},${dist}`);
                    }
                });

                if (towers.length < 3) {
                    outputDiv.innerHTML = '<p style="color:red">Error: At least 3 towers required</p>';
                    return;
                }

                try {
                    const response = await fetch(`/find_car?origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}&towers=${towers.join('|')}`);
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Failed to calculate position');
                    }
                    const html = await response.text();
                    mapFrame.srcdoc = html;
                    outputDiv.innerHTML = '<p>Location calculated successfully!</p>';
                } catch (error) {
                    outputDiv.innerHTML = `<p style="color:red">Error: ${error.message}</p>`;
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Vehicle Location Finder</h1>
            <p>Find a car's position on the road using tower coordinates</p>

            <div class="section">
                <h2>Route Information</h2>
                <input type="text" id="origin" placeholder="Origin (e.g., India Gate)" value="India Gate">
                <input type="text" id="destination" placeholder="Destination (e.g., Delhi)" value="Delhi">
            </div>

            <div class="section">
                <h2>Tower Coordinates</h2>
                <p>Enter at least 3 tower coordinates and distances to the vehicle</p>
                <div id="towerFields">
                    <div class="tower-input">
                        <input type="text" placeholder="Latitude" class="tower-lat" value="28.6201">
                        <input type="text" placeholder="Longitude" class="tower-lon" value="77.2056">
                        <input type="text" placeholder="Distance (meters)" class="tower-dist" value="500">
                    </div>
                    <div class="tower-input">
                        <input type="text" placeholder="Latitude" class="tower-lat" value="28.6198">
                        <input type="text" placeholder="Longitude" class="tower-lon" value="77.2045">
                        <input type="text" placeholder="Distance (meters)" class="tower-dist" value="600">
                    </div>
                    <div class="tower-input">
                        <input type="text" placeholder="Latitude" class="tower-lat" value="28.619">
                        <input type="text" placeholder="Longitude" class="tower-lon" value="77.204">
                        <input type="text" placeholder="Distance (meters)" class="tower-dist" value="550">
                    </div>
                </div>
                <button onclick="addTowerField()">Add Another Tower</button>
            </div>

            <button onclick="findCarLocation()">Find Car Location</button>
            <div id="output"></div>
            <iframe id="mapFrame"></iframe>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/find_car", response_class=HTMLResponse)
async def find_car(origin: str, destination: str, towers: str):
    try:
        # Get route
        origin_coords = mapper.address_to_latlong(origin)
        dest_coords = mapper.address_to_latlong(destination)
        route_data = mapper.route([origin_coords, dest_coords])
        
        if not route_data or 'paths' not in route_data:
            raise HTTPException(status_code=404, detail="No route found")
        
        route_points = polyline.decode(route_data['paths'][0]['points'])
        
        # Parse tower data
        tower_coords = []
        for tower in towers.split('|'):
            parts = tower.split(',')
            if len(parts) != 3:
                continue
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                dist = float(parts[2])
                tower_coords.append((lat, lon, dist))
            except ValueError:
                continue
        
        if len(tower_coords) < 3:
            raise HTTPException(status_code=400, detail="At least 3 valid towers required")
        
        # Estimate car position
        estimated_lat, estimated_lon = trilaterate(tower_coords)
        
        # Snap to road
        snap_result = mapper.snap_to_road([(estimated_lat, estimated_lon)])
        if not snap_result or 'paths' not in snap_result:
            raise HTTPException(status_code=500, detail="Failed to snap to road")
        
        snapped_points = polyline.decode(snap_result['paths'][0]['points'])
        if not snapped_points:
            raise HTTPException(status_code=500, detail="No snapped points returned")
        
        car_position = snapped_points[0]  # First point is our snapped position
        
        # Create map
        m = folium.Map(location=car_position, zoom_start=15)
        
        # Draw route
        folium.PolyLine(
            locations=route_points,
            color='blue',
            weight=5,
            opacity=0.7,
            popup="Route"
        ).add_to(m)
        
        # Mark towers
        for i, (lat, lon, dist) in enumerate(tower_coords):
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color='purple',
                fill=True,
                popup=f"Tower {i+1}<br>Distance: {dist}m"
            ).add_to(m)
            
            folium.Circle(
                location=[lat, lon],
                radius=dist,
                color='purple',
                fill=False,
                opacity=0.5
            ).add_to(m)
        
        # Mark estimated and snapped positions
        folium.CircleMarker(
            location=[estimated_lat, estimated_lon],
            radius=8,
            color='red',
            fill=True,
            popup=f"Estimated Position<br>{estimated_lat:.6f}, {estimated_lon:.6f}"
        ).add_to(m)
        
        folium.Marker(
            location=car_position,
            popup=f"Car on Road<br>{car_position[0]:.6f}, {car_position[1]:.6f}",
            icon=folium.Icon(color='green', icon='car')
        ).add_to(m)
        
        # Add legend
        legend_html = """
            <div style="position: fixed; bottom: 50px; left: 50px; width: 180px; 
            background: white; padding: 10px; border: 1px solid grey; z-index: 9999;">
                <b>Legend</b><br>
                <i style="background: blue; width: 15px; height: 3px; display: inline-block;"></i> Route<br>
                <i style="background: purple; width: 15px; height: 15px; border-radius: 50%; display: inline-block;"></i> Tower<br>
                <i style="background: red; width: 15px; height: 15px; border-radius: 50%; display: inline-block;"></i> Estimated<br>
                <i class="fa fa-car" style="color: green"></i> Car on Road
            </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return HTMLResponse(content=m._repr_html_())
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)