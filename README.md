# 🚗 Coordinates Approximator (FastAPI + GraphHopper + Folium)

A backend service built with **FastAPI** that takes user-provided coordinates, snaps them to the nearest road using the **GraphHopper Route API**, and generates an interactive **Folium map**.  

---

## ✨ Features
- Input **latitude, longitude** pairs
- Snaps coordinates to **nearest road**
- Displays results on an **interactive Folium map**
- Lightweight **FastAPI backend**
- **CORS enabled** (ready for frontend integration)
- Styled frontend using **TailwindCSS**

---

## 🛠️ Tech Stack
- [FastAPI](https://fastapi.tiangolo.com/) – Backend framework
- [GraphHopper API](https://www.graphhopper.com/) – Road snapping service
- [Folium](https://python-visualization.github.io/folium/) – Map rendering
- [TailwindCSS](https://tailwindcss.com/) – Frontend styling
- [Requests](https://docs.python-requests.org/) – External API calls
- [Pydantic](https://docs.pydantic.dev/) – Data validation

---

## 🚀 Getting Started

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/coordinates-approximator.git
cd coordinates-approximator
```

### 2️⃣ Install dependencies
Make sure you have **Python 3.9+** installed.
```bash
pip install fastapi uvicorn requests folium polyline pydantic
```

### 3️⃣ Configure API Key
In **`main.py`**, replace:
```python
GRAPH_HOPPER_API_KEY = "your-api-key-here"
```
👉 Get a free API key from [GraphHopper](https://www.graphhopper.com/).

### 4️⃣ Run the server
```bash
uvicorn main:app --reload
```

The app will be live at 👉 **http://127.0.0.1:8000/**  

---

## 📡 API Endpoints

### `GET /`
Serves the **frontend HTML page**.  
Contains:
- A textarea to enter coordinates
- A button to fetch nearest road data
- An embedded Folium map

---

### `POST /generate-map`
Snaps input coordinates to the nearest road.  

#### ✅ Request Body
```json
{
  "coordinates": [
    [52.5200, 13.4050],
    [52.5205, 13.4055]
  ]
}
```

#### 📩 Response
```json
{
  "map_html": "<iframe>...</iframe>",
  "road_coordinates": [
    [52.520000, 13.405000],
    [52.520500, 13.405500]
  ]
}
```

---

## 🌍 Usage
1. Open the homepage (`/`)
2. Enter coordinates in the textarea (one pair per line)
3. Click **Find Road location**
4. View:
   - Snapped coordinates
   - Interactive Folium map with markers & route

---

## 📜 License
MIT License © 2025  
Built with ❤️ using FastAPI, Folium & GraphHopper
