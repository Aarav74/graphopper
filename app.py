from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import requests
import folium
import polyline
import os


class GraphHopper:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://graphhopper.com/api/1/route"
        self.geocode_url = "https://graphhopper.com/api/1/geocode" 

    def address_to_latlong(self, address):
        """Converts an address string to a (latitude, longitude) tuple."""
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
                raise ValueError(f"Could not find coordinates for address: {address}")
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Geocoding API error: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def _make_route_request(self, points, unit="km"):
        """Internal method to make the GraphHopper routing API request."""
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
            raise HTTPException(status_code=500, detail=f"Routing API error: {e}")


    def distance(self, points, unit="km"):
        """Calculates the distance of a route."""
        route_data = self._make_route_request(points)
        if route_data and 'paths' in route_data and len(route_data['paths']) > 0:
            distance_meters = route_data['paths'][0]['distance']
            if unit == "km":
                return distance_meters / 1000
            return distance_meters
        return 0

    def route(self, points):
        """Gets the full routing details."""
        route_data = self._make_route_request(points)
        return route_data

app = FastAPI()
GRAPHHOPPER_API_KEY = "add4c752-8787-4a6b-ae81-6c8a357504b4"
mapper = GraphHopper(GRAPHHOPPER_API_KEY)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>GraphHopper Route Plotter</title>
        <style>
            body { font-family: sans-serif; text-align: center; margin: 50px; }
            .container { max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            input[type="text"] { width: 70%; padding: 10px; margin: 10px 0; border: 1px solid #ccc; border-radius: 4px; }
            button { padding: 10px 20px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            #output { margin-top: 20px; text-align: left; }
            iframe { width: 100%; height: 600px; border: none; }
        </style>
        <script>
            async function getRoute() {
                const origin = document.getElementById('origin').value;
                const destination = document.getElementById('destination').value;
                const outputDiv = document.getElementById('output');
                const mapFrame = document.getElementById('mapFrame');
                outputDiv.innerHTML = 'Loading...';
                mapFrame.srcdoc = ''; // Clear previous map

                try {
                    const response = await fetch(`/plot_route?origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}`);
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || 'Failed to fetch route.');
                    }
                    const htmlContent = await response.text();
                    mapFrame.srcdoc = htmlContent;
                    outputDiv.innerHTML = '<p>Route loaded successfully!</p>';
                } catch (error) {
                    outputDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>GraphHopper Route Plotter</h1>
            <p>Enter origin and destination addresses to plot a route on the map.</p>

            <input type="text" id="origin" placeholder="Enter origin address (e.g., Connaught Place, Delhi)">
            <input type="text" id="destination" placeholder="Enter destination address (e.g., India Gate, Delhi)">
            <button onclick="getRoute()">Get & Plot Route</button>

            <div id="output"></div>
            <iframe id="mapFrame"></iframe>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/plot_route", response_class=HTMLResponse)
async def plot_route(origin: str, destination: str):
    try:
 
        origin_latlong = mapper.address_to_latlong(origin)
        destination_latlong = mapper.address_to_latlong(destination)


        routing_data = mapper.route([origin_latlong, destination_latlong])

        if not routing_data or 'paths' not in routing_data or not routing_data['paths']:
            raise HTTPException(status_code=404, detail="No route found for the given addresses.")

        path = routing_data['paths'][0]
        encoded_points = path['points']
        distance_km = path['distance'] / 1000 
        time_ms = path['time']
        time_minutes = round(time_ms / 60000)


        decoded_points = polyline.decode(encoded_points)

    
        if decoded_points:
            map_center = [decoded_points[0][0], decoded_points[0][1]]
        else:
            map_center = [28.6139, 77.2090]

        m = folium.Map(location=map_center, zoom_start=12)

        folium.PolyLine(
            locations=decoded_points,
            color='blue',
            weight=5,
            opacity=0.7,
            tooltip=f"Distance: {distance_km:.2f} km, Time: {time_minutes} min"
        ).add_to(m)

        if decoded_points:
            folium.Marker(
                location=[decoded_points[0][0], decoded_points[0][1]],
                popup=f"Origin: {origin}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            folium.Marker(
                location=[decoded_points[-1][0], decoded_points[-1][1]],
                popup=f"Destination: {destination}",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)

        map_html = m._repr_html_() 
        return HTMLResponse(content=map_html)

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while plotting the route: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)