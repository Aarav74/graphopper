# app.py
import math
import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError
from scipy.optimize import least_squares

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("car_location_system")

app = FastAPI(title="Car Location System", version="1.1.1")


# -----------------------
# Helper data models
# -----------------------
class TowerMeasurementModel(BaseModel):
    distance: float = Field(..., gt=0, description="Measured distance from car to tower (meters)")
    azimuth: float = Field(..., ge=0, le=360, description="Azimuth from car to tower in degrees")
    signal_strength: Optional[float] = Field(default=None, description="Optional RSSI or signal strength")


class CarAnalysisRequest(BaseModel):
    car_lat: float = Field(..., ge=-90, le=90)
    car_lon: float = Field(..., ge=-180, le=180)
    tower_measurements: List[TowerMeasurementModel]


# -----------------------
# Core system
# -----------------------
class CarLocationSystem:
    def __init__(self, db_path: str = "car_location_system.db"):
        self.db_path = db_path
        self.conn = None
        self.accuracy_history: List[Dict] = []
        self.initialize_database()

    def initialize_database(self):
        """
        Initialize SQLite database and run lightweight migrations.
        This will create the table if it doesn't exist and add missing columns (like towers_used)
        so schema changes don't break already-existing DB files.
        """
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        # Base table creation (safe: will only create if missing)
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

        # Lightweight migrations: ensure expected columns exist; if not, add them
        cursor.execute("PRAGMA table_info(calculations);")
        cols = [r[1] for r in cursor.fetchall()]  # second column is name

        # Add towers_used column if missing
        if 'towers_used' not in cols:
            logger.info("Adding 'towers_used' column to calculations table")
            cursor.execute("ALTER TABLE calculations ADD COLUMN towers_used INTEGER DEFAULT 0;")
            self.conn.commit()

        # (Future migration points can be added here similarly)

    # -- Destination point (tower from car using distance & azimuth) --
    def calculate_tower_position(self, car_lat: float, car_lon: float, distance_m: float, azimuth_deg: float) -> Tuple[float, float]:
        """
        Given car location (lat, lon in degrees), a distance in meters and azimuth (degrees from north),
        compute the tower's lat/lon using the great-circle destination formula.
        """
        R = 6371000.0  # Earth radius meters
        φ1 = math.radians(car_lat)
        λ1 = math.radians(car_lon)
        θ = math.radians(azimuth_deg)
        δ = distance_m / R

        sinφ2 = math.sin(φ1) * math.cos(δ) + math.cos(φ1) * math.sin(δ) * math.cos(θ)
        # numerical safety
        sinφ2 = max(-1.0, min(1.0, sinφ2))
        φ2 = math.asin(sinφ2)

        y = math.sin(θ) * math.sin(δ) * math.cos(φ1)
        x = math.cos(δ) - math.sin(φ1) * math.sin(φ2)
        λ2 = λ1 + math.atan2(y, x)

        lat2 = math.degrees(φ2)
        lon2 = math.degrees(λ2)
        # Normalize lon to [-180,180]
        lon2 = (lon2 + 180) % 360 - 180

        return float(lat2), float(lon2)

    # -- Haversine distance in meters between two (lat, lon) pairs --
    @staticmethod
    def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, coord1)
        lat2, lon2 = map(math.radians, coord2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return 6371000.0 * c

    # -- Trilateration using non-linear least squares (residual = calc_dist - measured_dist) --
    def calculate_car_from_towers(self, tower_data: List[Dict]) -> Dict:
        """
        Given a list of towers each with 'latitude','longitude','distance' (meters),
        find best-fit car latitude and longitude using least squares on haversine distances.
        """
        if len(tower_data) < 3:
            raise ValueError("At least 3 towers required for reasonable 2D position fix")

        tower_positions = [(float(t['latitude']), float(t['longitude'])) for t in tower_data]
        distances = [float(t['distance']) for t in tower_data]

        # initial guess: weighted centroid (weights inverse to distance)
        weights = np.array([1.0 / max(d, 1.0) for d in distances], dtype=float)
        weights /= weights.sum()
        initial_lat = float(np.sum([p[0] * w for p, w in zip(tower_positions, weights)]))
        initial_lon = float(np.sum([p[1] * w for p, w in zip(tower_positions, weights)]))

        # residual function: for a candidate lat/lon, returns vector of (calc_dist - measured_dist)
        def residuals(x):
            lat, lon = float(x[0]), float(x[1])
            res = []
            for (tlat, tlon), meas in zip(tower_positions, distances):
                calc = self.haversine_distance((lat, lon), (tlat, tlon))
                res.append(calc - meas)
            return np.array(res)

        # set reasonable bounds: +/- 0.05 deg (~5.5 km) around initial guess
        lat_bound = 0.05
        lon_bound = 0.05
        lower_bounds = [initial_lat - lat_bound, initial_lon - lon_bound]
        upper_bounds = [initial_lat + lat_bound, initial_lon + lon_bound]

        try:
            result = least_squares(
                residuals,
                x0=np.array([initial_lat, initial_lon]),
                bounds=(lower_bounds, upper_bounds),
                method='trf',
                max_nfev=2000,
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6
            )
        except Exception:
            logger.exception("Least squares failed")
            result = None

        if result is not None and result.success:
            calc_lat, calc_lon = float(result.x[0]), float(result.x[1])
            optimization_info = {
                'method': 'least_squares_trf',
                'cost': float(result.cost),
                'success': True,
                'nfev': int(result.nfev)
            }
        else:
            calc_lat, calc_lon = initial_lat, initial_lon
            optimization_info = {
                'method': 'initial_guess_fallback',
                'cost': None,
                'success': False,
                'nfev': 0
            }

        # per-tower errors
        tower_errors = []
        for idx, ((tlat, tlon), meas) in enumerate(zip(tower_positions, distances), start=1):
            calc_d = self.haversine_distance((calc_lat, calc_lon), (tlat, tlon))
            err_m = float(calc_d - meas)
            tower_errors.append({
                'tower_id': f"Tower_{idx}",
                'measured_distance': float(meas),
                'calculated_distance': float(calc_d),
                'error_m': err_m
            })

        return {
            'calculated_location': {'latitude': calc_lat, 'longitude': calc_lon},
            'tower_errors': tower_errors,
            'optimization': optimization_info
        }

    # -- full pipeline: calculate towers from car measurements, then recover car position
    async def calculate_complete_analysis(self, car_lat: float, car_lon: float, tower_measurements: List[Dict]) -> Dict:
        try:
            if not tower_measurements or len(tower_measurements) < 3:
                raise ValueError("Need at least 3 tower measurements (distance + azimuth) to proceed")

            calculated_towers = []
            for i, meas in enumerate(tower_measurements, start=1):
                dist = float(meas['distance'])
                az = float(meas['azimuth'])
                sig = meas.get('signal_strength', None)
                tlat, tlon = self.calculate_tower_position(car_lat, car_lon, dist, az)
                calculated_towers.append({
                    'tower_id': f"Tower_{i}",
                    'latitude': tlat,
                    'longitude': tlon,
                    'distance': dist,
                    'azimuth': az,
                    'signal_strength': float(sig) if sig is not None else None
                })

            # Recover car from towers
            car_calc = self.calculate_car_from_towers(calculated_towers)
            calc_lat = car_calc['calculated_location']['latitude']
            calc_lon = car_calc['calculated_location']['longitude']

            # compute error (meters)
            distance_error = float(self.haversine_distance((float(car_lat), float(car_lon)), (float(calc_lat), float(calc_lon))))

            # store in DB
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO calculations (car_lat, car_lon, calculated_lat, calculated_lon, error_meters, accuracy_grade, towers_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                float(car_lat), float(car_lon),
                float(calc_lat), float(calc_lon),
                float(distance_error),
                self.get_accuracy_grade(distance_error),
                len(calculated_towers)
            ))
            self.conn.commit()
            calc_id = int(cursor.lastrowid) if cursor.lastrowid else 0

            # push to in-memory history
            hist_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'error_meters': distance_error,
                'accuracy_grade': self.get_accuracy_grade(distance_error),
                'calculation_id': calc_id
            }
            self.accuracy_history.append(hist_entry)

            # result
            result = {
                'actual_car_location': {'latitude': float(car_lat), 'longitude': float(car_lon), 'description': 'Actual Car Position'},
                'calculated_car_location': {'latitude': float(calc_lat), 'longitude': float(calc_lon)},
                'calculated_towers': calculated_towers,
                'accuracy_metrics': {
                    'distance_error_meters': distance_error,
                    'accuracy_grade': self.get_accuracy_grade(distance_error),
                    'target_achieved': distance_error <= 50.0
                },
                'calculation_details': {
                    'towers_used': len(calculated_towers),
                    'optimization': car_calc.get('optimization', {}),
                    'calculation_id': calc_id
                },
                'tower_analysis': car_calc['tower_errors']
            }
            return result
        except Exception:
            logger.exception("Complete analysis failed")
            raise

    @staticmethod
    def get_accuracy_grade(error_meters: float) -> str:
        e = float(error_meters)
        if e <= 10:
            return "Excellent 🎯"
        elif e <= 25:
            return "Very Good ✅"
        elif e <= 50:
            return "Good 👍 (Target Achieved)"
        elif e <= 100:
            return "Marginal ⚠️"
        else:
            return "Unacceptable 🚨"

    def get_history(self, limit: int = 50) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, car_lat, car_lon, calculated_lat, calculated_lon, error_meters, accuracy_grade, towers_used, calculation_time
            FROM calculations
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        return [
            {
                'id': r[0],
                'car_lat': r[1],
                'car_lon': r[2],
                'calculated_lat': r[3],
                'calculated_lon': r[4],
                'error_meters': r[5],
                'accuracy_grade': r[6],
                'towers_used': r[7],
                'calculation_time': r[8]
            } for r in rows
        ]


# instantiate
car_system = CarLocationSystem()


# -----------------------
# FastAPI endpoints
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/history")
async def history(limit: int = 20):
    try:
        history_data = car_system.get_history(limit=limit)
        return JSONResponse(content={"history": history_data})
    except Exception as e:
        logger.exception("History fetch failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calculate-analysis")
async def calculate_analysis(request: CarAnalysisRequest):
    try:
        result = await car_system.calculate_complete_analysis(float(request.car_lat), float(request.car_lon), [m.dict() for m in request.tower_measurements])
        return JSONResponse(content=result)
    except ValidationError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Analysis calculation error")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# -----------------------
# HTML frontend (fixed)
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def serve_car_analysis_interface():
    # Note: JS improvements: fixed getAccuracyClass, added spinner, disabled button, better validation
    html_content = r"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Car Location Analysis System</title>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <style>
    :root {
      --primary-color: #2c3e50;
    }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); margin:0; padding:20px; color:#222; }
    .container { max-width:1200px; margin: 0 auto; }
    .header { background:rgba(255,255,255,0.95); padding:18px; border-radius:12px; text-align:center; margin-bottom:14px; box-shadow:0 6px 20px rgba(0,0,0,0.08); }
    .main-grid { display:grid; grid-template-columns:420px 1fr; gap:16px; height: calc(100vh - 180px); }
    .control-panel { background:rgba(255,255,255,0.95); padding:16px; border-radius:12px; overflow:auto; }
    .card { background:white; border-radius:10px; padding:12px; margin-bottom:12px; box-shadow:0 4px 12px rgba(0,0,0,0.06); }
    .input-group { margin-bottom:10px; }
    label { display:block; font-weight:600; color:var(--primary-color); margin-bottom:6px; font-size:0.9em; }
    input[type=number] { width:100%; padding:8px; border-radius:6px; border:1px solid #ddd; font-size:14px; }
    .btn { background: linear-gradient(135deg,#27ae60,#2ecc71); color:white; border:none; padding:10px 12px; font-weight:700; border-radius:8px; cursor:pointer; width:100%; }
    .btn[disabled] { opacity:0.6; cursor:not-allowed; transform:none; }
    .map { height:100%; width:100%; }
    .maps-container { display:grid; grid-template-rows:1fr 1fr; gap:12px; }
    .tower-input-group { background:#f8f9fa; padding:10px; border-radius:8px; margin-bottom:10px; }
    .small { font-size:0.85em; color:#555; }
    .results { background:#f8f9fa; padding:10px; border-radius:8px; }
    .accuracy-badge { padding:8px 12px; border-radius:999px; display:inline-block; color:white; font-weight:700; }
    .excellent { background: linear-gradient(135deg,#27ae60,#2ecc71); }
    .very-good { background: linear-gradient(135deg,#3498db,#2980b9); }
    .good { background: linear-gradient(135deg,#f39c12,#e67e22); }
    .marginal { background: linear-gradient(135deg,#e74c3c,#c0392b); }
    .spinner { display:inline-block; width:18px; height:18px; border:3px solid rgba(255,255,255,0.3); border-top-color:white; border-radius:50%; animation:spin 1s linear infinite; vertical-align:middle; margin-left:8px; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🚗 Car Location Analysis</h1>
      <div class="small">Calculate tower positions from car location and verify accuracy (improved solver)</div>
    </div>

    <div class="main-grid">
      <div class="control-panel">
        <div class="card">
          <h3>📍 Car Location</h3>
          <div class="input-group">
            <label>Car Latitude</label>
            <input id="carLat" type="number" step="0.000001" value="28.613524">
          </div>
          <div class="input-group">
            <label>Car Longitude</label>
            <input id="carLon" type="number" step="0.000001" value="77.208962">
          </div>
        </div>

        <div class="card">
          <h3>📡 Tower Measurements</h3>
          <div class="small">Provide measured distance (meters) and azimuth (degrees from North). At least 3 towers for a good fix.</div>

          <div class="tower-input-group">
            <h4>Tower 1</h4>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <div><label>Distance (m)</label><input id="t1d" type="number" value="350" step="0.1"></div>
              <div><label>Azimuth (°)</label><input id="t1a" type="number" value="45" min="0" max="360"></div>
            </div>
          </div>

          <div class="tower-input-group">
            <h4>Tower 2</h4>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <div><label>Distance (m)</label><input id="t2d" type="number" value="400" step="0.1"></div>
              <div><label>Azimuth (°)</label><input id="t2a" type="number" value="135" min="0" max="360"></div>
            </div>
          </div>

          <div class="tower-input-group">
            <h4>Tower 3</h4>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
              <div><label>Distance (m)</label><input id="t3d" type="number" value="380" step="0.1"></div>
              <div><label>Azimuth (°)</label><input id="t3a" type="number" value="315" min="0" max="360"></div>
            </div>
          </div>

          <button id="calcBtn" class="btn" onclick="calculateAnalysis()">🚀 Calculate Analysis</button>
        </div>

        <div id="resultsPanel" class="card">
          <h3>📊 Analysis Results</h3>
          <div class="results" id="resultsContent">
            Enter car location and tower measurements to see accuracy analysis.
          </div>
        </div>

      </div>

      <div class="maps-container">
        <div class="card" style="position:relative; overflow:hidden;">
          <div style="position:absolute; left:12px; top:12px; z-index:1000; background:rgba(255,255,255,0.9); padding:6px 8px; border-radius:8px; font-weight:700;">📍 Map 1: Actual Car + Calculated Towers</div>
          <div id="map1" class="map"></div>
        </div>

        <div class="card" style="position:relative; overflow:hidden;">
          <div style="position:absolute; left:12px; top:12px; z-index:1000; background:rgba(255,255,255,0.9); padding:6px 8px; border-radius:8px; font-weight:700;">🎯 Map 2: Calculated Car Location</div>
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
    const defaultLat = parseFloat(document.getElementById('carLat').value) || 28.6135;
    const defaultLon = parseFloat(document.getElementById('carLon').value) || 77.2088;
    map1 = L.map('map1').setView([defaultLat, defaultLon], 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map1);

    map2 = L.map('map2').setView([defaultLat, defaultLon], 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map2);
  }

  function clearMarkers(map, arr) {
    arr.forEach(m => {
      try { map.removeLayer(m); } catch(e){}
    });
    return [];
  }

  function createCarIcon(color) {
    return L.divIcon({
      html: `<div style="background:${color}; width:40px; height:40px; border-radius:50%; border:3px solid white; display:flex; align-items:center; justify-content:center; font-size:18px;">🚗</div>`,
      className: ''
    });
  }

  function createTowerIcon() {
    return L.divIcon({
      html: `<div style="background:#e74c3c; width:30px; height:30px; border-radius:50%; border:2px solid white; display:flex; align-items:center; justify-content:center; font-size:14px;">📡</div>`,
      className: ''
    });
  }

  function getAccuracyClass(grade) {
    if (!grade) return 'marginal';
    if (grade.includes('Excellent')) return 'excellent';
    if (grade.includes('Very Good')) return 'very-good';
    if (grade.includes('Good')) return 'good';
    return 'marginal';
  }

  async function calculateAnalysis() {
    const btn = document.getElementById('calcBtn');
    btn.disabled = true;
    const spinner = document.createElement('span');
    spinner.className = 'spinner';
    btn.appendChild(spinner);

    try {
      const carLat = parseFloat(document.getElementById('carLat').value);
      const carLon = parseFloat(document.getElementById('carLon').value);

      const towerMeasurements = [
        { distance: parseFloat(document.getElementById('t1d').value), azimuth: parseFloat(document.getElementById('t1a').value) },
        { distance: parseFloat(document.getElementById('t2d').value), azimuth: parseFloat(document.getElementById('t2a').value) },
        { distance: parseFloat(document.getElementById('t3d').value), azimuth: parseFloat(document.getElementById('t3a').value) }
      ];

      // Basic validation
      if (!Array.isArray(towerMeasurements) || towerMeasurements.length < 3) {
        alert('Please provide at least 3 tower measurements.');
        return;
      }

      const payload = { car_lat: carLat, car_lon: carLon, tower_measurements: towerMeasurements };

      const resp = await fetch('/calculate-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error('Server error: ' + resp.status + ' - ' + txt);
      }

      const data = await resp.json();
      displayResults(data);
      visualizeMaps(data);
    } catch (err) {
      console.error(err);
      alert('Analysis error: ' + (err.message || err));
    } finally {
      // cleanup spinner/button state
      try { btn.removeChild(spinner); } catch(e){}
      btn.disabled = false;
    }
  }

  function displayResults(data) {
    const el = document.getElementById('resultsContent');
    const acc = data.accuracy_metrics;
    const cls = getAccuracyClass(acc.accuracy_grade);
    const target = acc.target_achieved ? '🎯 TARGET ACHIEVED: under 50m' : '⚠️ Target not met';
    let towersHtml = '';
    (data.calculated_towers || []).forEach(t => {
      towersHtml += `<div style="background:#eef6ff;padding:8px;border-radius:6px;margin-bottom:6px;">
        <strong>${t.tower_id}</strong><br>
        Pos: ${t.latitude.toFixed(6)}, ${t.longitude.toFixed(6)}<br>
        Distance: ${t.distance.toFixed(1)}m | Azimuth: ${t.azimuth}°
      </div>`;
    });

    el.innerHTML = `
      <div style="text-align:center;margin-bottom:10px;">
        <div class="accuracy-badge ${cls}">${acc.accuracy_grade}</div>
        <div style="font-weight:700; font-size:1.1em; margin-top:8px;">Error: ${acc.distance_error_meters.toFixed(2)} meters</div>
        <div style="margin-top:6px; color:#333;">${target}</div>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:8px;">
        <div style="background:#e9fbe9;padding:8px;border-radius:6px;">
          <h4 style="margin:0 0 6px 0">Actual Car</h4>
          <div style="font-family:monospace; background:#2c3e50;color:white;padding:6px;border-radius:4px;">
            ${data.actual_car_location.latitude.toFixed(6)}<br>${data.actual_car_location.longitude.toFixed(6)}
          </div>
        </div>
        <div style="background:#e9f3ff;padding:8px;border-radius:6px;">
          <h4 style="margin:0 0 6px 0">Calculated</h4>
          <div style="font-family:monospace; background:#2c3e50;color:white;padding:6px;border-radius:4px;">
            ${data.calculated_car_location.latitude.toFixed(6)}<br>${data.calculated_car_location.longitude.toFixed(6)}
          </div>
        </div>
      </div>
      <div style="margin-top:10px;">
        <h4 style="margin:0 0 8px 0">Calculated Towers</h4>
        ${towersHtml}
      </div>
    `;
  }

  function visualizeMaps(data) {
    const actual = data.actual_car_location;
    const calc = data.calculated_car_location;
    const towers = data.calculated_towers || [];

    map1Markers = clearMarkers(map1, map1Markers);
    map2Markers = clearMarkers(map2, map2Markers);

    const actualMarker = L.marker([actual.latitude, actual.longitude], { icon: createCarIcon('#27ae60') }).addTo(map1);
    actualMarker.bindPopup('<b>Actual Car</b>').openPopup();
    map1Markers.push(actualMarker);

    const pts1 = [[actual.latitude, actual.longitude]];

    towers.forEach(t => {
      const tm = L.marker([t.latitude, t.longitude], { icon: createTowerIcon() }).addTo(map1);
      tm.bindPopup(`<b>${t.tower_id}</b><br>Distance: ${t.distance.toFixed(1)} m`).openPopup();
      map1Markers.push(tm);

      const circle = L.circle([actual.latitude, actual.longitude], { radius: t.distance, color:'#e74c3c', fillOpacity:0.05 }).addTo(map1);
      map1Markers.push(circle);

      const line = L.polyline([[actual.latitude, actual.longitude], [t.latitude, t.longitude]], { color:'#3498db' }).addTo(map1);
      map1Markers.push(line);

      pts1.push([t.latitude, t.longitude]);
    });

    try { map1.fitBounds(pts1, { padding:[40,40] }); } catch(e){}

    // Map 2
    const calcMarker = L.marker([calc.latitude, calc.longitude], { icon: createCarIcon('#3498db') }).addTo(map2);
    calcMarker.bindPopup(`<b>Calculated Car</b><br>Error: ${data.accuracy_metrics.distance_error_meters.toFixed(1)} m`).openPopup();
    map2Markers.push(calcMarker);

    const actualMarker2 = L.marker([actual.latitude, actual.longitude], { icon: createCarIcon('#27ae60') }).addTo(map2);
    map2Markers.push(actualMarker2);

    const errLine = L.polyline([[actual.latitude, actual.longitude], [calc.latitude, calc.longitude]], { color:'#e74c3c', weight:3, dashArray:'8,8' }).addTo(map2);
    map2Markers.push(errLine);

    const mid = [(actual.latitude + calc.latitude)/2, (actual.longitude + calc.longitude)/2];
    const label = L.marker(mid, {
      icon: L.divIcon({
        html: `<div style="background:#e74c3c;color:white;padding:4px 8px;border-radius:6px;font-weight:700;">${data.accuracy_metrics.distance_error_meters.toFixed(1)} m</div>`,
        className: ''
      })
    }).addTo(map2);
    map2Markers.push(label);

    towers.forEach(t => {
      const tm2 = L.marker([t.latitude, t.longitude], { icon: createTowerIcon() }).addTo(map2);
      map2Markers.push(tm2);
    });

    const pts2 = [[calc.latitude, calc.longitude], [actual.latitude, actual.longitude]];
    towers.forEach(t => pts2.push([t.latitude, t.longitude]));
    try { map2.fitBounds(pts2, { padding:[40,40] }); } catch(e){}
  }

  document.addEventListener('DOMContentLoaded', function(){ initMaps(); });
</script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# -----------------------
# Run guard for local debugging
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, log_level="info")
