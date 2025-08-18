from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import requests
import folium
import polyline

app = FastAPI()
GRAPHHOPPER_API_KEY = "add4c752-8787-4a6b-ae81-6c8a357504b4"
GRAPHHOPPER_ROUTE_URL = "https://graphhopper.com/api/1/route"
GRAPHHOPPER_GEOCODE_URL = "https://graphhopper.com/api/1/geocode"

async def get_lat_lon(address: str):
    """Converts an address to (latitude, longitude) using GraphHopper Geocoding API."""
    params = {"q": address, "limit": 1, "key": GRAPHHOPPER_API_KEY}
    try:
        response = requests.get(GRAPHHOPPER_GEOCODE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if data and 'hits' in data and len(data['hits']) > 0:
            point = data['hits'][0]['point']
            return (point['lat'], point['lng'])
        raise ValueError(f"Address not found: {address}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Geocoding API error: {e}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/route_map", response_class=HTMLResponse)
async def route_map(
    origin: str = "Connaught Place, Delhi",
    destination: str = "India Gate, Delhi"
):
    """
    Generates an interactive map with a route between origin and destination.
    Access at: http://localhost:8000/route_map?origin=ADDRESS1&destination=ADDRESS2
    Example: http://localhost:8000/route_map?origin=Connaught%20Place,%20Delhi&destination=India%20Gate,%20Delhi
    """
    try:
        o_lat, o_lon = await get_lat_lon(origin)
        d_lat, d_lon = await get_lat_lon(destination)
        route_params = {
            "point": [f"{o_lat},{o_lon}", f"{d_lat},{d_lon}"],
            "vehicle": "car",
            "points_encoded": "true",
            "key": GRAPHHOPPER_API_KEY
        }

        response = requests.get(GRAPHHOPPER_ROUTE_URL, params=route_params)
        response.raise_for_status()
        route_data = response.json()

        if not route_data or 'paths' not in route_data or not route_data['paths']:
            raise HTTPException(status_code=404, detail="No route found.")

        path = route_data['paths'][0]
        encoded_points = path['points']
        decoded_points = polyline.decode(encoded_points)

        m = folium.Map(location=[(o_lat + d_lat) / 2, (o_lon + d_lon) / 2], zoom_start=12)
        folium.PolyLine(
            locations=decoded_points,
            color='blue',
            weight=5,
            opacity=0.7
        ).add_to(m)

        folium.Marker([o_lat, o_lon], popup=origin, icon=folium.Icon(color='green')).add_to(m)
        folium.Marker([d_lat, d_lon], popup=destination, icon=folium.Icon(color='red')).add_to(m)

        return HTMLResponse(content=m._repr_html_())

    except HTTPException as e:
        raise e
    except Exception as e:

        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",port=8000)