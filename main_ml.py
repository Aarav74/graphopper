import math
import logging
import sqlite3
import joblib
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from scipy.optimize import least_squares
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("groundhog_ml_system")

app = FastAPI(title="Groundhog-Like ML Location System", version="2.0.0")

SAMPLE_MEASUREMENTS: List[Dict] = [
    {
        "id": "sample_1", "timestamp": 1721721000000,
        "car": {"latitude": 28.613939, "longitude": 77.229508},
        "towers": [
            {"id": "BTS1", "distance": 703.9, "azimuth": 286},
            {"id": "BTS2", "distance": 706.0, "azimuth": 292},
            {"id": "BTS3", "distance": 710.0, "azimuth": 284},
        ],
        "meta": {"RSRP_4G": -98.9, "RSRQ_4G": -16.1}
    },
    {
        "id": "sample_2", "timestamp": 1721721000000,
        "car": {"latitude": 28.556112, "longitude": 77.250834},
        "towers": [
            {"id": "BTS1", "distance": 201.2, "azimuth": 81},
            {"id": "BTS2", "distance": 391.7, "azimuth": 198},
            {"id": "BTS3", "distance": 426.0, "azimuth": 12},
        ],
        "meta": {"RSRP_4G": -105.8, "RSRQ_4G": -12.9}
    },
    {
        "id": "sample_3", "timestamp": 1721721000000,
        "car": {"latitude": 28.4595, "longitude": 77.0266},
        "towers": [
            {"id": "BTS1", "distance": 204.2, "azimuth": 187},
            {"id": "BTS2", "distance": 262.0, "azimuth": 236},
            {"id": "BTS3", "distance": 264.4, "azimuth": 274},
        ],
        "meta": {"RSRP_4G": -106.4, "RSRQ_4G": -13.1}
    },
    {
        "id": "sample_4", "timestamp": 1721721000000,
        "car": {"latitude": 28.521456, "longitude": 77.265789},
        "towers": [
            {"id": "BTS1", "distance": 263.1, "azimuth": 203},
            {"id": "BTS2", "distance": 281.1, "azimuth": 160},
            {"id": "BTS3", "distance": 488.2, "azimuth": 286},
        ],
        "meta": {"RSRP_4G": -100.4, "RSRQ_4G": -14.8}
    },
    {
        "id": "sample_5", "timestamp": 1721721000000,
        "car": {"latitude": 28.489123, "longitude": 77.278901},
        "towers": [
             {"id": "BTS1", "distance": 496.6, "azimuth": 69},
             {"id": "BTS2", "distance": 615.4, "azimuth": 337},
             {"id": "BTS3", "distance": 644.3, "azimuth": 79},
        ],
        "meta": {"RSRP_4G": -102.6, "RSRQ_4G": -14.1}
    },
    {
        "id": "sample_6", "timestamp": 1721721000000,
        "car": {"latitude": 28.46789, "longitude": 77.282345},
        "towers": [
            {"id": "BTS1", "distance": 910.5, "azimuth": 98},
            {"id": "BTS2", "distance": 990.2, "azimuth": 157},
            {"id": "BTS3", "distance": 1237.6, "azimuth": 227},
        ],
        "meta": {"RSRP_4G": -83.1, "RSRQ_4G": -10.0}
    }
]

class MLCorrector:
    """
    This class implements the 'Groundhog' logic:
    It uses Random Forest to learn the correlation between Signal Quality (Chaos) 
    and Geometric Error (Multipath effects).
    """
    def __init__(self, model_path="model_correction.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_trained = True
                logger.info("✅ ML Model loaded. AI Correction Enabled.")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
        else:
            logger.warning("⚠️ No ML model found. Using pure Geometry until trained.")

    def train(self, db_path):
        """
        Fetches history from DB and trains the AI to fix geometric errors.
        Features: [Geometric_Lat, Geometric_Lon, RSRP, RSRQ]
        Target:   [Lat_Error, Lon_Error]
        """
        conn = sqlite3.connect(db_path)
        # We need records where we have signal stats (RSRP/RSRQ) stored
        query = '''
            SELECT 
                car_lat, car_lon,           -- Truth
                calculated_lat, calculated_lon, -- Geometry Guess
                avg_rsrp, avg_rsrq          -- Chaos Factors (Signal Quality)
            FROM calculations
            WHERE avg_rsrp IS NOT NULL
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) < 5:
            return {"status": "error", "message": f"Not enough data to train. Have {len(df)}, need 5+."}

        # 1. Features (What the AI sees)
        X = df[['calculated_lat', 'calculated_lon', 'avg_rsrp', 'avg_rsrq']]
        
        # 2. Labels (The Error the AI must predict)
        # Error = Actual - Calculated
        y_lat_err = df['car_lat'] - df['calculated_lat']
        y_lon_err = df['car_lon'] - df['calculated_lon']
        y = pd.concat([y_lat_err, y_lon_err], axis=1)
        y.columns = ['lat_error', 'lon_error']

        # 3. Train Random Forest (Captures non-linear signal patterns)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X, y)

        # 4. Save
        joblib.dump(self.model, self.model_path)
        self.is_trained = True
        return {"status": "success", "message": f"AI Trained on {len(df)} samples."}

    def correct_position(self, calc_lat, calc_lon, rsrp, rsrq) -> Tuple[float, float, bool]:
        """
        Input: Geometric Guess + Signal Quality
        Output: AI Corrected Position
        """
        if not self.is_trained or rsrp is None:
            return calc_lat, calc_lon, False # Fallback to physics

        # Predict the error bias
        input_data = pd.DataFrame([[calc_lat, calc_lon, rsrp, rsrq]], 
                                  columns=['calculated_lat', 'calculated_lon', 'avg_rsrp', 'avg_rsrq'])
        
        predicted_error = self.model.predict(input_data)[0] # [lat_bias, lon_bias]
        
        # Apply Correction
        final_lat = calc_lat + predicted_error[0]
        final_lon = calc_lon + predicted_error[1]
        
        return final_lat, final_lon, True


class TowerMeasurementModel(BaseModel):
    distance: float = Field(..., gt=0)
    azimuth: float = Field(..., ge=0, le=360)

class CarAnalysisRequest(BaseModel):
    car_lat: float
    car_lon: float
    tower_measurements: List[TowerMeasurementModel]
    
    rsrp: Optional[float] = -100.0
    rsrq: Optional[float] = -12.0

class CarLocationSystem:
    def __init__(self, db_path: str = "car_location_system.db"):
        self.db_path = db_path
        self.conn = None
        self.initialize_database()
        # Initialize ML Engine
        self.ml_corrector = MLCorrector()

    def initialize_database(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()
        # Updated schema to include Signal Stats (Chaos variables)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                car_lat REAL, car_lon REAL,
                calculated_lat REAL, calculated_lon REAL,
                avg_rsrp REAL, avg_rsrq REAL, 
                error_meters REAL,
                accuracy_grade TEXT,
                method_used TEXT,
                calculation_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()


    def calculate_tower_position(self, car_lat, car_lon, dist, az):
        # Destination point formula
        R = 6371000.0
        φ1 = math.radians(car_lat)
        λ1 = math.radians(car_lon)
        θ = math.radians(az)
        δ = dist / R
        sinφ2 = math.sin(φ1)*math.cos(δ) + math.cos(φ1)*math.sin(δ)*math.cos(θ)
        φ2 = math.asin(sinφ2)
        y = math.sin(θ)*math.sin(δ)*math.cos(φ1)
        x = math.cos(δ) - math.sin(φ1)*math.sin(φ2)
        λ2 = λ1 + math.atan2(y, x)
        return math.degrees(φ2), (math.degrees(λ2) + 180) % 360 - 180

    @staticmethod
    def haversine_distance(c1, c2):
        lat1, lon1 = map(math.radians, c1)
        lat2, lon2 = map(math.radians, c2)
        a = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
        return 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    async def process_location(self, req: CarAnalysisRequest):
        # 1. Simulate finding Tower positions (Reverse Engineering for the demo)
        towers_calc = []
        for i, m in enumerate(req.tower_measurements):
            tlat, tlon = self.calculate_tower_position(req.car_lat, req.car_lon, m.distance, m.azimuth)
            towers_calc.append((tlat, tlon, m.distance))

        # 2. Geometric Triangulation (Least Squares) - The "Physics" Guess
        def residuals(x):
            return [self.haversine_distance((x[0], x[1]), (t[0], t[1])) - t[2] for t in towers_calc]
        
        # Start guess: avg of towers
        avg_lat = sum(t[0] for t in towers_calc)/len(towers_calc)
        avg_lon = sum(t[1] for t in towers_calc)/len(towers_calc)
        
        res = least_squares(residuals, [avg_lat, avg_lon], loss='soft_l1')
        geo_lat, geo_lon = res.x[0], res.x[1]

        # 3. AI CORRECTION STEP (The Groundhog Layer)
        final_lat, final_lon, used_ai = self.ml_corrector.correct_position(
            geo_lat, geo_lon, req.rsrp, req.rsrq
        )

        # 4. Calculate Final Error
        final_error = self.haversine_distance((req.car_lat, req.car_lon), (final_lat, final_lon))
        grade = self.get_grade(final_error)

        # 5. Save to DB for future Training
        self.save_to_db(req, geo_lat, geo_lon, req.rsrp, req.rsrq, final_error, grade, "AI" if used_ai else "Geometry")

        # 6. Response
        return {
            "actual": {"lat": req.car_lat, "lon": req.car_lon},
            "calculated": {"lat": final_lat, "lon": final_lon},
            "method": "🤖 ML Enhanced" if used_ai else "📐 Pure Geometry",
            "metrics": {
                "error_meters": final_error,
                "grade": grade,
                "signal_chaos": f"RSRP: {req.rsrp}, RSRQ: {req.rsrq}"
            }
        }

    def save_to_db(self, req, c_lat, c_lon, rsrp, rsrq, err, grade, method):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO calculations (car_lat, car_lon, calculated_lat, calculated_lon, avg_rsrp, avg_rsrq, error_meters, accuracy_grade, method_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (req.car_lat, req.car_lon, c_lat, c_lon, rsrp, rsrq, err, grade, method))
        self.conn.commit()

    @staticmethod
    def get_grade(err):
        if err <= 20: return "Excellent 🎯"
        if err <= 50: return "Good ✅"
        return "Poor ⚠️"

car_system = CarLocationSystem()


@app.get("/samples")
def get_samples():
    return {"samples": SAMPLE_MEASUREMENTS}

@app.post("/analyze")
async def analyze(req: CarAnalysisRequest):
    return await car_system.process_location(req)

@app.post("/train-ai")
async def train_ai():
    """Triggers the learning process on accumulated data"""
    try:
        res = car_system.ml_corrector.train(car_system.db_path)
        return JSONResponse(content=res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def ui():
    return r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Groundhog AI Location System</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { font-family: sans-serif; display: grid; grid-template-columns: 350px 1fr; height: 100vh; margin: 0; }
            .sidebar { background: #f4f4f4; padding: 20px; overflow-y: auto; border-right: 2px solid #ddd; }
            .map-container { height: 100vh; }
            .card { background: white; padding: 15px; margin-bottom: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            input, select, button { width: 100%; margin-bottom: 10px; padding: 8px; box-sizing: border-box; }
            button { background: #2c3e50; color: white; border: none; cursor: pointer; border-radius: 4px; }
            button:hover { background: #34495e; }
            button.ai { background: linear-gradient(135deg, #667eea, #764ba2); font-weight: bold; }
            .stat-box { font-size: 0.9em; margin-top: 5px; }
        </style>
    </head>
    <body>
        <div class="sidebar">
            <h2>📡 Groundhog AI</h2>
            
            <div class="card">
                <h3>1. Select Sample Data</h3>
                <select id="sampleSelect" onchange="loadSample()"><option>Loading...</option></select>
            </div>

            <div class="card">
                <h3>2. Run Analysis</h3>
                <label>Car Lat/Lon (Truth)</label>
                <div style="display:flex; gap:5px;">
                    <input id="carLat" placeholder="Lat">
                    <input id="carLon" placeholder="Lon">
                </div>
                <label>Signal Quality (Chaos)</label>
                <div style="display:flex; gap:5px;">
                    <input id="rsrp" placeholder="RSRP (-dbm)">
                    <input id="rsrq" placeholder="RSRQ">
                </div>
                <div id="towersArea"></div>
                <button onclick="runAnalysis()">📍 Locate Car</button>
            </div>

            <div class="card">
                <h3>3. AI Control</h3>
                <p class="stat-box">Run multiple analyses to build database, then click Train.</p>
                <button class="ai" onclick="trainAI()">🧠 Train AI Model</button>
                <div id="aiStatus" style="margin-top:10px; font-size:0.8em; color:green;"></div>
            </div>

            <div id="results" class="card" style="display:none;">
                <h3>📊 Results</h3>
                <div id="resContent"></div>
            </div>
        </div>
        <div id="map" class="map-container"></div>

        <script>
            let map = L.map('map').setView([28.6139, 77.2295], 13);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            let markers = [];

            // Load samples on start
            fetch('/samples').then(r=>r.json()).then(d => {
                let s = document.getElementById('sampleSelect');
                s.innerHTML = '<option value="">-- Select --</option>';
                d.samples.forEach((x,i) => {
                    let opt = document.createElement('option');
                    opt.value = i;
                    opt.text = `${x.id} (Towers: ${x.towers.length})`;
                    opt.data = x;
                    s.appendChild(opt);
                });
                s.samples = d.samples;
            });

            function loadSample() {
                let idx = document.getElementById('sampleSelect').value;
                if(!idx) return;
                let data = document.getElementById('sampleSelect').samples[idx];
                
                document.getElementById('carLat').value = data.car.latitude;
                document.getElementById('carLon').value = data.car.longitude;
                document.getElementById('rsrp').value = data.meta.RSRP_4G;
                document.getElementById('rsrq').value = data.meta.RSRQ_4G;
                
                window.currentTowers = data.towers;
            }

            async function runAnalysis() {
                let req = {
                    car_lat: parseFloat(document.getElementById('carLat').value),
                    car_lon: parseFloat(document.getElementById('carLon').value),
                    rsrp: parseFloat(document.getElementById('rsrp').value),
                    rsrq: parseFloat(document.getElementById('rsrq').value),
                    tower_measurements: window.currentTowers
                };

                let resp = await fetch('/analyze', {
                    method: 'POST', headers: {'Content-Type':'application/json'},
                    body: JSON.stringify(req)
                });
                let res = await resp.json();
                
                // UI Updates
                document.getElementById('results').style.display = 'block';
                document.getElementById('resContent').innerHTML = `
                    <b>Method:</b> ${res.method}<br>
                    <b>Error:</b> ${res.metrics.error_meters.toFixed(2)}m<br>
                    <b>Grade:</b> ${res.metrics.grade}
                `;

                // Map Updates
                markers.forEach(m => map.removeLayer(m));
                markers = [];
                
                // Actual (Green)
                let act = L.marker([res.actual.lat, res.actual.lon]).addTo(map).bindPopup("Actual Car");
                act._icon.style.filter = "hue-rotate(240deg)";
                markers.push(act);

                // Calculated (Blue/Red)
                let calc = L.marker([res.calculated.lat, res.calculated.lon]).addTo(map).bindPopup(res.method);
                markers.push(calc);

                L.polyline([[res.actual.lat, res.actual.lon], [res.calculated.lat, res.calculated.lon]], {color: 'red', dashArray: '5,5'}).addTo(map);
                
                map.setView([res.calculated.lat, res.calculated.lon], 15);
            }

            async function trainAI() {
                let btn = document.querySelector('button.ai');
                btn.innerText = "Training...";
                try {
                    let r = await fetch('/train-ai', {method:'POST'});
                    let d = await r.json();
                    document.getElementById('aiStatus').innerText = d.message;
                    if(d.status === 'success') alert("AI Trained! Future requests will be more accurate.");
                } catch(e) { alert("Error training"); }
                btn.innerText = "🧠 Train AI Model";
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)