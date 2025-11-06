import numpy as np
import math
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional
import json
from scipy.optimize import minimize

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Your existing triangulation code with enhanced database
class HighPrecisionESIMTracker:
    def __init__(self):
        self.cell_tower_database = self.initialize_tower_database()
        self.signal_propagation_models = self.initialize_propagation_models()
        self.highway_points = self.initialize_highway_points()
    
    def initialize_highway_points(self) -> List[Dict]:
        """Initialize all highway test points with their data"""
        return [
            {'lat': 28.613939, 'lon': 77.229508, 'name': 'Point 1: Dhaula Kuan (Delhi)'},
            {'lat': 28.556112, 'lon': 77.250834, 'name': 'Point 2: IGI Airport'},
            {'lat': 28.459500, 'lon': 77.026600, 'name': 'Point 3: HUDA City Center'},
            {'lat': 28.521456, 'lon': 77.265789, 'name': 'Point 4: Kherki Daula'},
            {'lat': 28.489123, 'lon': 77.278901, 'name': 'Point 5: IMT Manesar'},
            {'lat': 28.467890, 'lon': 77.282345, 'name': 'Point 6: Behror Border'},
            {'lat': 28.445678, 'lon': 77.293456, 'name': 'Point 7: Kotputli'},
            {'lat': 28.423456, 'lon': 77.298765, 'name': 'Point 8: Neemrana'},
            {'lat': 28.401234, 'lon': 77.306789, 'name': 'Point 9: Shahpura'},
            {'lat': 28.378901, 'lon': 77.315678, 'name': 'Point 10: Jaipur Outskirts'}
        ]
    
    def initialize_tower_database(self) -> Dict:
        """Initialize database with all your highway towers"""
        return {
            # Delhi-Jaipur Highway Towers
            
            # Point 1: 28.613939, 77.229508
            '35382': {'lat': 28.620000, 'lon': 77.235000, 'type': '4G', 'height': 35, 'operator': 'JIO'},
            '35383': {'lat': 28.621000, 'lon': 77.236000, 'type': '4G', 'height': 32, 'operator': 'JIO'},
            '10695': {'lat': 28.619500, 'lon': 77.234500, 'type': '4G', 'height': 40, 'operator': 'AIRTEL'},
            
            # Point 2: 28.556112, 77.250834
            '55454': {'lat': 28.557000, 'lon': 77.252000, 'type': '4G', 'height': 38, 'operator': 'JIO'},
            '52796': {'lat': 28.555000, 'lon': 77.249000, 'type': '4G', 'height': 42, 'operator': 'AIRTEL'},
            '55443': {'lat': 28.558000, 'lon': 77.251000, 'type': '4G', 'height': 35, 'operator': 'VI'},
            
            # Point 3: 28.459500, 77.026600
            '55136': {'lat': 28.461000, 'lon': 77.028000, 'type': '4G', 'height': 45, 'operator': 'JIO'},
            '54485': {'lat': 28.458000, 'lon': 77.025000, 'type': '4G', 'height': 38, 'operator': 'AIRTEL'},
            '55168': {'lat': 28.460000, 'lon': 77.027000, 'type': '4G', 'height': 40, 'operator': 'VI'},
            
            # Point 4: 28.521456, 77.265789
            '53154': {'lat': 28.523000, 'lon': 77.267000, 'type': '4G', 'height': 36, 'operator': 'JIO'},
            '50368': {'lat': 28.520000, 'lon': 77.264000, 'type': '4G', 'height': 42, 'operator': 'AIRTEL'},
            '55260': {'lat': 28.522000, 'lon': 77.266000, 'type': '4G', 'height': 38, 'operator': 'VI'},
            
            # Point 5: 28.489123, 77.278901
            '7359': {'lat': 28.491000, 'lon': 77.280000, 'type': '4G', 'height': 35, 'operator': 'JIO'},
            '52373': {'lat': 28.488000, 'lon': 77.277000, 'type': '4G', 'height': 44, 'operator': 'AIRTEL'},
            '3802': {'lat': 28.490000, 'lon': 77.279000, 'type': '4G', 'height': 40, 'operator': 'VI'},
            
            # Point 6: 28.467890, 77.282345
            '53285': {'lat': 28.469000, 'lon': 77.284000, 'type': '4G', 'height': 38, 'operator': 'JIO'},
            '24668': {'lat': 28.466000, 'lon': 77.281000, 'type': '4G', 'height': 42, 'operator': 'AIRTEL'},
            '53864': {'lat': 28.468000, 'lon': 77.283000, 'type': '4G', 'height': 45, 'operator': 'VI'},
            
            # Point 7: 28.445678, 77.293456
            '27915': {'lat': 28.447000, 'lon': 77.295000, 'type': '4G', 'height': 36, 'operator': 'JIO'},
            '56190': {'lat': 28.444000, 'lon': 77.292000, 'type': '4G', 'height': 40, 'operator': 'AIRTEL'},
            '55770': {'lat': 28.446000, 'lon': 77.294000, 'type': '4G', 'height': 38, 'operator': 'VI'},
            
            # Point 8: 28.423456, 77.298765
            '3570': {'lat': 28.425000, 'lon': 77.300000, 'type': '4G', 'height': 42, 'operator': 'JIO'},
            '7048': {'lat': 28.422000, 'lon': 77.297000, 'type': '4G', 'height': 35, 'operator': 'AIRTEL'},
            '50628': {'lat': 28.424000, 'lon': 77.299000, 'type': '4G', 'height': 40, 'operator': 'VI'},
            
            # Point 9: 28.401234, 77.306789
            '51288': {'lat': 28.403000, 'lon': 77.308000, 'type': '4G', 'height': 38, 'operator': 'JIO'},
            '51426': {'lat': 28.400000, 'lon': 77.305000, 'type': '4G', 'height': 44, 'operator': 'AIRTEL'},
            '56232': {'lat': 28.402000, 'lon': 77.307000, 'type': '4G', 'height': 36, 'operator': 'VI'},
            
            # Point 10: 28.378901, 77.315678
            '56209': {'lat': 28.380000, 'lon': 77.317000, 'type': '4G', 'height': 40, 'operator': 'JIO'},
            '50648': {'lat': 28.377000, 'lon': 77.314000, 'type': '4G', 'height': 38, 'operator': 'AIRTEL'},
            '3768': {'lat': 28.379000, 'lon': 77.316000, 'type': '4G', 'height': 42, 'operator': 'VI'},
            
            # Original towers
            '24823': {'lat': 28.639500, 'lon': 77.224800, 'type': '4G', 'height': 35, 'operator': 'JIO'},
            '27639': {'lat': 28.637200, 'lon': 77.222500, 'type': '4G', 'height': 32, 'operator': 'JIO'},
            '27682': {'lat': 28.640800, 'lon': 77.226200, 'type': '4G', 'height': 40, 'operator': 'JIO'},
            '12345678': {'lat': 28.632400, 'lon': 77.218800, 'type': '4G', 'height': 35, 'operator': 'JIO'},
            '12345679': {'lat': 28.631500, 'lon': 77.217500, 'type': '4G', 'height': 32, 'operator': 'JIO'},
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
    
    def get_tower_measurements_for_location(self, location: Tuple[float, float]) -> List[Dict]:
        """Get tower measurements for a specific highway location"""
        highway_data = {
            (28.613939, 77.229508): [
                {'cell_id': '35382', 'distance': 703.9025590543764, 'azimuth': 286, 'signal_strength': -98.914},
                {'cell_id': '35383', 'distance': 706.0192250487871, 'azimuth': 292, 'signal_strength': -98.914},
                {'cell_id': '10695', 'distance': 710.0020347954293, 'azimuth': 284, 'signal_strength': -98.914}
            ],
            (28.556112, 77.250834): [
                {'cell_id': '55454', 'distance': 201.2857064577612, 'azimuth': 81, 'signal_strength': -105.817},
                {'cell_id': '52796', 'distance': 391.73491761298226, 'azimuth': 198, 'signal_strength': -105.817},
                {'cell_id': '55443', 'distance': 426.0350469144528, 'azimuth': 12, 'signal_strength': -105.817}
            ],
            (28.459500, 77.026600): [
                {'cell_id': '55136', 'distance': 204.20375635832463, 'azimuth': 187, 'signal_strength': -106.442},
                {'cell_id': '54485', 'distance': 262.0960388885914, 'azimuth': 236, 'signal_strength': -106.442},
                {'cell_id': '55168', 'distance': 264.4220763163507, 'azimuth': 274, 'signal_strength': -106.442}
            ],
            (28.521456, 77.265789): [
                {'cell_id': '53154', 'distance': 263.1323098088868, 'azimuth': 203, 'signal_strength': -100.482},
                {'cell_id': '50368', 'distance': 281.12730283275584, 'azimuth': 160, 'signal_strength': -100.482},
                {'cell_id': '55260', 'distance': 488.25719642382126, 'azimuth': 286, 'signal_strength': -100.482}
            ],
            (28.489123, 77.278901): [
                {'cell_id': '7359', 'distance': 496.6772634247112, 'azimuth': 69, 'signal_strength': -102.697},
                {'cell_id': '52373', 'distance': 615.4148156096423, 'azimuth': 337, 'signal_strength': -102.697},
                {'cell_id': '3802', 'distance': 644.3205753228746, 'azimuth': 79, 'signal_strength': -102.697}
            ],
            (28.467890, 77.282345): [
                {'cell_id': '53285', 'distance': 910.5672761005306, 'azimuth': 98, 'signal_strength': -83.137},
                {'cell_id': '24668', 'distance': 990.2529396727788, 'azimuth': 157, 'signal_strength': -83.137},
                {'cell_id': '53864', 'distance': 1237.6686040470163, 'azimuth': 227, 'signal_strength': -83.137}
            ],
            (28.445678, 77.293456): [
                {'cell_id': '27915', 'distance': 999.4283137159614, 'azimuth': 302, 'signal_strength': -105.360},
                {'cell_id': '56190', 'distance': 1013.5452921669805, 'azimuth': 98, 'signal_strength': -105.360},
                {'cell_id': '55770', 'distance': 1084.1657390634791, 'azimuth': 114, 'signal_strength': -105.360}
            ],
            (28.423456, 77.298765): [
                {'cell_id': '3570', 'distance': 320.28367491904737, 'azimuth': 15, 'signal_strength': -103.790},
                {'cell_id': '7048', 'distance': 762.189837448132, 'azimuth': 342, 'signal_strength': -103.790},
                {'cell_id': '50628', 'distance': 884.244160863876, 'azimuth': 244, 'signal_strength': -103.790}
            ],
            (28.401234, 77.306789): [
                {'cell_id': '51288', 'distance': 178.2564647031069, 'azimuth': 185, 'signal_strength': -96.883},
                {'cell_id': '51426', 'distance': 330.2042043919641, 'azimuth': 347, 'signal_strength': -96.883},
                {'cell_id': '56232', 'distance': 413.1734621221455, 'azimuth': 238, 'signal_strength': -96.883}
            ],
            (28.378901, 77.315678): [
                {'cell_id': '56209', 'distance': 364.0810067859212, 'azimuth': 234, 'signal_strength': -107.335},
                {'cell_id': '50648', 'distance': 531.1432986440477, 'azimuth': 285, 'signal_strength': -107.335},
                {'cell_id': '3768', 'distance': 913.2758226775524, 'azimuth': 199, 'signal_strength': -107.335}
            ]
        }
        
        # Find the closest location in our database
        min_distance = float('inf')
        closest_location = None
        
        for loc in highway_data.keys():
            distance = self.haversine_distance(location, loc)
            if distance < min_distance:
                min_distance = distance
                closest_location = loc
        
        if closest_location and min_distance < 5000:  # Within 5km
            return highway_data[closest_location]
        else:
            return []
    
    def calculate_complete_highway_triangulation(self) -> Dict[str, Any]:
        """Calculate triangulation for all highway points and connect them"""
        print("Calculating triangulation for all highway points...")
        
        all_results = {}
        road_path = []
        
        for point in self.highway_points:
            location = (point['lat'], point['lon'])
            print(f"Processing {point['name']} at {location}")
            
            # Get tower measurements for this location
            tower_measurements = self.get_tower_measurements_for_location(location)
            
            if not tower_measurements:
                print(f"  No tower data found for {point['name']}")
                continue
            
            # Perform triangulation
            result = self.advanced_triangulation(
                tower_measurements, 
                reference_location=location,
                gps_location=location  # Use actual location as GPS for comparison
            )
            
            if 'estimated_location' in result:
                all_results[point['name']] = result
                
                # Add to road path
                road_path.append({
                    'name': point['name'],
                    'original_location': location,
                    'calculated_location': result['estimated_location'],
                    'accuracy': result['accuracy_meters'],
                    'towers_used': len(tower_measurements)
                })
                print(f"  ✓ Success - Accuracy: {result['accuracy_meters']:.1f}m")
            else:
                print(f"  ✗ Failed - {result.get('error', 'Unknown error')}")
        
        # Calculate road statistics
        total_points = len(road_path)
        if total_points > 0:
            avg_accuracy = sum(point['accuracy'] for point in road_path) / total_points
            total_distance = self.calculate_road_distance(road_path)
        else:
            avg_accuracy = 0
            total_distance = 0
        
        return {
            'complete_highway_analysis': True,
            'total_points_processed': total_points,
            'average_accuracy_meters': avg_accuracy,
            'total_road_distance_km': total_distance,
            'road_path': road_path,
            'point_results': all_results,
            'all_towers': self.get_all_towers_for_highway()
        }
    
    def calculate_road_distance(self, road_path: List[Dict]) -> float:
        """Calculate total road distance in kilometers"""
        total_distance = 0
        for i in range(len(road_path) - 1):
            point1 = road_path[i]['calculated_location']
            point2 = road_path[i + 1]['calculated_location']
            distance = self.haversine_distance(point1, point2) / 1000.0  # Convert to km
            total_distance += distance
        return total_distance
    
    def get_all_towers_for_highway(self) -> List[Dict]:
        """Get all towers used in highway analysis"""
        all_towers = []
        for point in self.highway_points:
            location = (point['lat'], point['lon'])
            tower_measurements = self.get_tower_measurements_for_location(location)
            
            for meas in tower_measurements:
                tower_id = meas['cell_id']
                if tower_id in self.cell_tower_database:
                    tower_data = self.cell_tower_database[tower_id]
                    tower_info = {
                        'id': tower_id,
                        'coordinates': (tower_data['lat'], tower_data['lon']),
                        'type': tower_data['type'],
                        'operator': tower_data['operator'],
                        'height': tower_data['height'],
                        'distance': meas['distance'],
                        'signal_strength': meas['signal_strength'],
                        'serves_point': point['name']
                    }
                    # Avoid duplicates
                    if not any(t['id'] == tower_id for t in all_towers):
                        all_towers.append(tower_info)
        
        return all_towers
    
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
        
        print(f"Valid towers processed: {valid_towers_count}")
        
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

class HighwayLocationRequest(BaseModel):
    latitude: float
    longitude: float
    use_gps: bool = False

@app.get("/", response_class=HTMLResponse)
async def serve_triangulation_ui():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESIM Triangulation Visualizer - Highway Analysis</title>
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
            .road-path { stroke-width: 6; opacity: 0.8; }
            .highway-line { 
                stroke: #e53e3e; 
                stroke-width: 4; 
                stroke-dasharray: 10, 5;
                opacity: 0.8;
            }
        </style>
    </head>
    <body class="bg-gray-100 min-h-screen p-4">
        <div class="max-w-7xl mx-auto">
            <h1 class="text-3xl font-bold text-center text-gray-800 mb-2">ESIM Triangulation Visualizer - Complete Highway Analysis</h1>
            <p class="text-center text-gray-600 mb-6">Delhi-Jaipur Highway Route Mapping</p>
            
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <!-- Input Panel -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">Highway Analysis Controls</h2>
                    
                    <div class="space-y-4 mb-4">
                        <div class="grid grid-cols-2 gap-2">
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Test Latitude</label>
                                <input type="number" id="testLat" value="28.613939" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border">
                            </div>
                            <div>
                                <label class="block text-sm font-medium text-gray-700">Test Longitude</label>
                                <input type="number" id="testLon" value="77.229508" step="any" class="mt-1 block w-full rounded-md border-gray-300 shadow-sm p-2 border">
                            </div>
                        </div>
                        
                        <div class="flex items-center">
                            <input type="checkbox" id="useGps" class="rounded border-gray-300 text-blue-600 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200">
                            <label for="useGps" class="ml-2 text-sm text-gray-700">Use as GPS reference</label>
                        </div>
                    </div>

                    <div class="space-y-3 mb-4">
                        <h3 class="font-semibold text-gray-700">Quick Test Locations</h3>
                        <div class="grid grid-cols-1 gap-2">
                            <button onclick="loadHighwayLocation(28.613939, 77.229508)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 1: Dhaula Kuan (Delhi)
                            </button>
                            <button onclick="loadHighwayLocation(28.556112, 77.250834)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 2: IGI Airport
                            </button>
                            <button onclick="loadHighwayLocation(28.459500, 77.026600)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 3: HUDA City Center
                            </button>
                            <button onclick="loadHighwayLocation(28.521456, 77.265789)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 4: Kherki Daula
                            </button>
                            <button onclick="loadHighwayLocation(28.489123, 77.278901)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 5: IMT Manesar
                            </button>
                            <button onclick="loadHighwayLocation(28.467890, 77.282345)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 6: Behror Border
                            </button>
                            <button onclick="loadHighwayLocation(28.445678, 77.293456)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 7: Kotputli
                            </button>
                            <button onclick="loadHighwayLocation(28.423456, 77.298765)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 8: Neemrana
                            </button>
                            <button onclick="loadHighwayLocation(28.401234, 77.306789)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 9: Shahpura
                            </button>
                            <button onclick="loadHighwayLocation(28.378901, 77.315678)" class="bg-blue-100 hover:bg-blue-200 text-blue-800 p-2 rounded text-sm text-left">
                                📍 Point 10: Jaipur Outskirts
                            </button>
                        </div>
                    </div>

                    <div class="space-y-2">
                        <button id="calculateHighwayBtn" class="w-full bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-4 rounded-lg transition-all">
                            Calculate Single Point
                        </button>
                        <button id="calculateCompleteHighwayBtn" class="w-full bg-purple-500 hover:bg-purple-600 text-white font-bold py-3 px-4 rounded-lg transition-all">
                            🛣️ Analyze Complete Highway
                        </button>
                        <button id="clearBtn" class="w-full bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg transition-all">
                            Clear All
                        </button>
                        <button id="fullscreenBtn" class="w-full bg-orange-500 hover:bg-orange-600 text-white font-bold py-2 px-4 rounded-lg transition-all">
                            Full Screen View
                        </button>
                    </div>
                </div>

                <!-- Results Panel -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">Analysis Results</h2>
                    
                    <div id="resultsContent" class="space-y-4">
                        <div class="text-center text-gray-500">
                            <p>Select a highway location or analyze complete highway</p>
                        </div>
                    </div>

                    <div class="mt-6" id="highwayResultsSection" style="display: none;">
                        <h3 class="font-semibold text-gray-700 mb-2">Highway Analysis</h3>
                        <div id="highwayResults" class="space-y-2">
                            <!-- Highway results will be populated here -->
                        </div>
                    </div>

                    <div class="mt-6" id="gpsComparisonSection">
                        <h3 class="font-semibold text-gray-700 mb-2">GPS Comparison</h3>
                        <div id="gpsComparison" class="space-y-2">
                            <!-- GPS comparison will be populated here -->
                        </div>
                    </div>

                    <div class="mt-6" id="methodsResultsSection">
                        <h3 class="font-semibold text-gray-700 mb-2">Algorithm Results</h3>
                        <div id="methodsResults" class="space-y-2">
                            <!-- Method results will be populated here -->
                        </div>
                    </div>

                    <div class="mt-6" id="towerDetailsSection">
                        <h3 class="font-semibold text-gray-700 mb-2">Tower Details</h3>
                        <div id="towerDetails" class="space-y-2">
                            <!-- Tower details will be populated here -->
                        </div>
                    </div>
                </div>

                <!-- Map Panel -->
                <div class="bg-white rounded-xl shadow-lg p-6">
                    <h2 class="text-xl font-bold text-gray-800 mb-4">Highway Visualization</h2>
                    <div id="map"></div>
                    <div class="mt-4 bg-blue-50 p-3 rounded-lg">
                        <p class="text-sm text-blue-700">
                            <strong>Map Legend:</strong><br>
                            <span style="color: #FF4444">📍 Input Location</span><br>
                            <span style="color: #44FF44">📍 Calculated Location</span><br>
                            <span style="color: #4444FF">📍 GPS Location</span><br>
                            <span style="color: #FF6B00">📍 Algorithm Results</span><br>
                            <span style="color: purple">📡 Tower Location</span><br>
                            <span style="color: blue">⭕ Tower Range</span><br>
                            <span style="color: #e53e3e">🛣️ Highway Route</span>
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
                <h2 class="text-2xl font-bold text-gray-800 mb-4">Highway Analysis - Full Screen</h2>
                <div id="fullscreenResultsContent" class="space-y-4">
                    <div class="text-center text-gray-500">
                        <p>Select a highway location or analyze complete highway</p>
                    </div>
                </div>
                <div class="mt-6" id="fullscreenHighwayResultsSection" style="display: none;">
                    <h3 class="font-semibold text-gray-700 mb-2">Highway Analysis</h3>
                    <div id="fullscreenHighwayResults" class="space-y-2">
                        <!-- Highway results will be populated here -->
                    </div>
                </div>
                <div class="mt-6" id="fullscreenGpsComparisonSection">
                    <h3 class="font-semibold text-gray-700 mb-2">GPS Comparison</h3>
                    <div id="fullscreenGpsComparison" class="space-y-2">
                        <!-- GPS comparison will be populated here -->
                    </div>
                </div>
                <div class="mt-6" id="fullscreenMethodsResultsSection">
                    <h3 class="font-semibold text-gray-700 mb-2">Algorithm Results</h3>
                    <div id="fullscreenMethodsResults" class="space-y-2">
                        <!-- Method results will be populated here -->
                    </div>
                </div>
                <div class="mt-6" id="fullscreenTowerDetailsSection">
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
            let currentData = null;
            let currentLocation = null;
            let highwayRoute = null;
            let fullscreenHighwayRoute = null;

            function initializeMap() {
                map = L.map('map').setView([28.613939, 77.229508], 10);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(map);
            }

            function initializeFullscreenMap() {
                const container = document.getElementById('fullscreenMap');
                fullscreenMap = L.map(container).setView([28.613939, 77.229508], 10);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                    attribution: '© OpenStreetMap contributors'
                }).addTo(fullscreenMap);
            }

            function clearMap() {
                markers.forEach(marker => map.removeLayer(marker));
                markers = [];
                circles.forEach(circle => map.removeLayer(circle));
                circles = [];
                if (highwayRoute) {
                    map.removeLayer(highwayRoute);
                    highwayRoute = null;
                }
            }

            function clearFullscreenMap() {
                if (fullscreenMap) {
                    fullscreenMarkers.forEach(marker => fullscreenMap.removeLayer(marker));
                    fullscreenCircles.forEach(circle => fullscreenMap.removeLayer(circle));
                    fullscreenMarkers = [];
                    fullscreenCircles = [];
                    if (fullscreenHighwayRoute) {
                        fullscreenMap.removeLayer(fullscreenHighwayRoute);
                        fullscreenHighwayRoute = null;
                    }
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

            function addHighwayRoute(points, color, targetMap = map, isFullscreen = false) {
                const polyline = L.polyline(points, {
                    color: color,
                    weight: 6,
                    opacity: 0.8,
                    dashArray: '10, 5'
                }).addTo(targetMap);
                
                if (isFullscreen) {
                    fullscreenHighwayRoute = polyline;
                } else {
                    highwayRoute = polyline;
                }
                return polyline;
            }

            function loadHighwayLocation(lat, lon) {
                document.getElementById('testLat').value = lat;
                document.getElementById('testLon').value = lon;
                currentLocation = {latitude: lat, longitude: lon};
                
                // Center map on the location
                map.setView([lat, lon], 13);
                if (fullscreenMap) {
                    fullscreenMap.setView([lat, lon], 13);
                }
            }

            async function calculateHighwayTriangulation() {
                const lat = parseFloat(document.getElementById('testLat').value);
                const lon = parseFloat(document.getElementById('testLon').value);
                const useGps = document.getElementById('useGps').checked;

                if (isNaN(lat) || isNaN(lon)) {
                    alert('Please enter valid coordinates');
                    return;
                }

                const requestData = {
                    latitude: lat,
                    longitude: lon,
                    use_gps: useGps
                };

                try {
                    const response = await fetch('/calculate-highway-triangulation', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(requestData)
                    });

                    if (!response.ok) {
                        throw new Error('Server error: ' + response.status);
                    }

                    const data = await response.json();
                    currentData = data;
                    displayResults(data);
                    displayFullscreenResults(data);
                    visualizeOnMap(data, lat, lon, useGps ? {lat, lon} : null);

                } catch (error) {
                    alert('Error calculating triangulation: ' + error.message);
                }
            }

            async function calculateCompleteHighway() {
                try {
                    const response = await fetch('/calculate-complete-highway', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    });

                    if (!response.ok) {
                        throw new Error('Server error: ' + response.status);
                    }

                    const data = await response.json();
                    currentData = data;
                    displayCompleteHighwayResults(data);
                    displayFullscreenCompleteHighwayResults(data);
                    visualizeCompleteHighwayOnMap(data);

                } catch (error) {
                    alert('Error analyzing complete highway: ' + error.message);
                }
            }

            function displayResults(data) {
                // Hide highway results section for single point analysis
                document.getElementById('highwayResultsSection').style.display = 'none';
                document.getElementById('fullscreenHighwayResultsSection').style.display = 'none';
                
                const resultsDiv = document.getElementById('resultsContent');
                const gpsDiv = document.getElementById('gpsComparison');
                const methodsDiv = document.getElementById('methodsResults');
                const towersDiv = document.getElementById('towerDetails');

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="bg-red-50 p-4 rounded-lg text-red-700">Error: ${data.error}</div>`;
                    return;
                }

                updateSinglePointResults(resultsDiv, gpsDiv, methodsDiv, towersDiv, data);
            }

            function displayCompleteHighwayResults(data) {
                // Show highway results section
                document.getElementById('highwayResultsSection').style.display = 'block';
                
                const resultsDiv = document.getElementById('resultsContent');
                const highwayDiv = document.getElementById('highwayResults');
                const gpsDiv = document.getElementById('gpsComparison');
                const methodsDiv = document.getElementById('methodsResults');
                const towersDiv = document.getElementById('towerDetails');

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="bg-red-50 p-4 rounded-lg text-red-700">Error: ${data.error}</div>`;
                    return;
                }

                updateCompleteHighwayResults(resultsDiv, highwayDiv, gpsDiv, methodsDiv, towersDiv, data);
            }

            function displayFullscreenResults(data) {
                document.getElementById('fullscreenHighwayResultsSection').style.display = 'none';
                
                const resultsDiv = document.getElementById('fullscreenResultsContent');
                const gpsDiv = document.getElementById('fullscreenGpsComparison');
                const methodsDiv = document.getElementById('fullscreenMethodsResults');
                const towersDiv = document.getElementById('fullscreenTowerDetails');

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="bg-red-50 p-4 rounded-lg text-red-700">Error: ${data.error}</div>`;
                    return;
                }

                updateSinglePointResults(resultsDiv, gpsDiv, methodsDiv, towersDiv, data);
            }

            function displayFullscreenCompleteHighwayResults(data) {
                document.getElementById('fullscreenHighwayResultsSection').style.display = 'block';
                
                const resultsDiv = document.getElementById('fullscreenResultsContent');
                const highwayDiv = document.getElementById('fullscreenHighwayResults');
                const gpsDiv = document.getElementById('fullscreenGpsComparison');
                const methodsDiv = document.getElementById('fullscreenMethodsResults');
                const towersDiv = document.getElementById('fullscreenTowerDetails');

                if (data.error) {
                    resultsDiv.innerHTML = `<div class="bg-red-50 p-4 rounded-lg text-red-700">Error: ${data.error}</div>`;
                    return;
                }

                updateCompleteHighwayResults(resultsDiv, highwayDiv, gpsDiv, methodsDiv, towersDiv, data);
            }

            function updateSinglePointResults(resultsDiv, gpsDiv, methodsDiv, towersDiv, data) {
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

            function updateCompleteHighwayResults(resultsDiv, highwayDiv, gpsDiv, methodsDiv, towersDiv, data) {
                // Main results
                resultsDiv.innerHTML = `
                    <div class="bg-purple-50 p-4 rounded-lg">
                        <h3 class="font-bold text-purple-800 text-lg">Complete Highway Analysis</h3>
                        <p class="text-sm">Delhi-Jaipur Highway Route Mapping</p>
                    </div>
                `;

                // Highway results
                highwayDiv.innerHTML = `
                    <div class="grid grid-cols-2 gap-4">
                        <div class="bg-green-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-green-800">Points Processed</h4>
                            <p class="text-2xl font-bold">${data.total_points_processed}/10</p>
                        </div>
                        <div class="bg-blue-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-blue-800">Average Accuracy</h4>
                            <p class="text-2xl font-bold">${data.average_accuracy_meters.toFixed(1)}m</p>
                        </div>
                        <div class="bg-orange-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-orange-800">Road Distance</h4>
                            <p class="text-2xl font-bold">${data.total_road_distance_km.toFixed(1)}km</p>
                        </div>
                        <div class="bg-red-50 p-3 rounded-lg">
                            <h4 class="font-semibold text-red-800">Towers Used</h4>
                            <p class="text-2xl font-bold">${data.all_towers.length}</p>
                        </div>
                    </div>
                `;

                // Point-by-point results
                if (data.road_path) {
                    const pointsDiv = document.createElement('div');
                    pointsDiv.className = 'mt-4 space-y-2';
                    pointsDiv.innerHTML = '<h4 class="font-semibold text-gray-700">Point Details:</h4>';
                    
                    data.road_path.forEach(point => {
                        const pointDiv = document.createElement('div');
                        pointDiv.className = 'bg-gray-50 p-2 rounded text-sm';
                        pointDiv.innerHTML = `
                            <strong>${point.name}</strong><br>
                            Original: ${point.original_location[0].toFixed(6)}, ${point.original_location[1].toFixed(6)}<br>
                            Calculated: ${point.calculated_location[0].toFixed(6)}, ${point.calculated_location[1].toFixed(6)}<br>
                            Accuracy: ${point.accuracy.toFixed(1)}m, Towers: ${point.towers_used}
                        `;
                        pointsDiv.appendChild(pointDiv);
                    });
                    highwayDiv.appendChild(pointsDiv);
                }

                // Clear other sections for highway analysis
                gpsDiv.innerHTML = '<div class="bg-gray-50 p-3 rounded-lg text-gray-500 text-center">GPS comparison available for single point analysis</div>';
                methodsDiv.innerHTML = '<div class="bg-gray-50 p-3 rounded-lg text-gray-500 text-center">Method details available for single point analysis</div>';

                // Tower details for highway
                if (data.all_towers) {
                    towersDiv.innerHTML = '';
                    data.all_towers.forEach(tower => {
                        const towerDiv = document.createElement('div');
                        towerDiv.className = 'bg-gray-50 p-2 rounded text-sm';
                        towerDiv.innerHTML = `
                            <strong>Tower ${tower.id}</strong> (${tower.operator})<br>
                            Location: ${tower.coordinates[0].toFixed(6)}, ${tower.coordinates[1].toFixed(6)}<br>
                            Serves: ${tower.serves_point}<br>
                            Distance: ${tower.distance.toFixed(1)}m, Signal: ${tower.signal_strength}dBm
                        `;
                        towersDiv.appendChild(towerDiv);
                    });
                }
            }

            function visualizeOnMap(data, inputLat, inputLon, gpsLocation) {
                clearMap();
                clearFullscreenMap();

                // Add input location
                addMarker(inputLat, inputLon, '#FF4444', 'I', 
                    `<strong>Input Location</strong><br>${inputLat.toFixed(6)}, ${inputLon.toFixed(6)}`, map);
                if (fullscreenMap) addMarker(inputLat, inputLon, '#FF4444', 'I', 
                    `<strong>Input Location</strong><br>${inputLat.toFixed(6)}, ${inputLon.toFixed(6)}`, fullscreenMap, true);

                // Add GPS location if available
                if (gpsLocation) {
                    addMarker(gpsLocation.lat, gpsLocation.lon, '#4444FF', 'G',
                        `<strong>GPS Location</strong><br>${gpsLocation.lat.toFixed(6)}, ${gpsLocation.lon.toFixed(6)}<br>Accuracy: ${data.gps_comparison?.gps_accuracy_meters.toFixed(1)}m`, map);
                    if (fullscreenMap) addMarker(gpsLocation.lat, gpsLocation.lon, '#4444FF', 'G',
                        `<strong>GPS Location</strong><br>${gpsLocation.lat.toFixed(6)}, ${gpsLocation.lon.toFixed(6)}<br>Accuracy: ${data.gps_comparison?.gps_accuracy_meters.toFixed(1)}m`, fullscreenMap, true);
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
                allPoints.push([inputLat, inputLon]);
                if (gpsLocation) allPoints.push([gpsLocation.lat, gpsLocation.lon]);
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

            function visualizeCompleteHighwayOnMap(data) {
                clearMap();
                clearFullscreenMap();

                if (!data.road_path || data.road_path.length === 0) {
                    alert('No highway data to visualize');
                    return;
                }

                // Create arrays for the highway route
                const originalRoute = [];
                const calculatedRoute = [];

                // Add markers and build routes
                data.road_path.forEach((point, index) => {
                    const number = index + 1;
                    
                    // Original location marker
                    addMarker(point.original_location[0], point.original_location[1], '#FF4444', number,
                        `<strong>${point.name}</strong><br>Original: ${point.original_location[0].toFixed(6)}, ${point.original_location[1].toFixed(6)}`, map);
                    
                    // Calculated location marker  
                    addMarker(point.calculated_location[0], point.calculated_location[1], '#44FF44', number,
                        `<strong>${point.name}</strong><br>Calculated: ${point.calculated_location[0].toFixed(6)}, ${point.calculated_location[1].toFixed(6)}<br>Accuracy: ${point.accuracy.toFixed(1)}m`, map);

                    // Add to routes
                    originalRoute.push(point.original_location);
                    calculatedRoute.push(point.calculated_location);

                    // For fullscreen map
                    if (fullscreenMap) {
                        addMarker(point.original_location[0], point.original_location[1], '#FF4444', number,
                            `<strong>${point.name}</strong><br>Original: ${point.original_location[0].toFixed(6)}, ${point.original_location[1].toFixed(6)}`, fullscreenMap, true);
                        addMarker(point.calculated_location[0], point.calculated_location[1], '#44FF44', number,
                            `<strong>${point.name}</strong><br>Calculated: ${point.calculated_location[0].toFixed(6)}, ${point.calculated_location[1].toFixed(6)}<br>Accuracy: ${point.accuracy.toFixed(1)}m`, fullscreenMap, true);
                    }
                });

                // Add highway routes
                addHighwayRoute(originalRoute, '#FF4444', map);
                addHighwayRoute(calculatedRoute, '#44FF44', map);

                if (fullscreenMap) {
                    addHighwayRoute(originalRoute, '#FF4444', fullscreenMap, true);
                    addHighwayRoute(calculatedRoute, '#44FF44', fullscreenMap, true);
                }

                // Add all towers
                if (data.all_towers) {
                    data.all_towers.forEach(tower => {
                        addMarker(tower.coordinates[0], tower.coordinates[1], 'purple', 'T',
                            `<strong>Tower ${tower.id}</strong> (${tower.operator})<br>${tower.coordinates[0].toFixed(6)}, ${tower.coordinates[1].toFixed(6)}<br>Serves: ${tower.serves_point}`, map);
                        addCircle(tower.coordinates[0], tower.coordinates[1], tower.distance, 'blue', map);

                        if (fullscreenMap) {
                            addMarker(tower.coordinates[0], tower.coordinates[1], 'purple', 'T',
                                `<strong>Tower ${tower.id}</strong> (${tower.operator})<br>${tower.coordinates[0].toFixed(6)}, ${tower.coordinates[1].toFixed(6)}<br>Serves: ${tower.serves_point}`, fullscreenMap, true);
                            addCircle(tower.coordinates[0], tower.coordinates[1], tower.distance, 'blue', fullscreenMap, true);
                        }
                    });
                }

                // Fit map to show entire highway
                const allPoints = [...originalRoute, ...calculatedRoute];
                if (data.all_towers) {
                    data.all_towers.forEach(tower => {
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
                    fullscreenElement.style.display = 'grid';
                    if (!fullscreenMap) {
                        initializeFullscreenMap();
                    }
                    if (currentData && currentLocation) {
                        if (currentData.complete_highway_analysis) {
                            displayFullscreenCompleteHighwayResults(currentData);
                            visualizeCompleteHighwayOnMap(currentData);
                        } else {
                            displayFullscreenResults(currentData);
                            const useGps = document.getElementById('useGps').checked;
                            visualizeOnMap(currentData, currentLocation.latitude, currentLocation.longitude, useGps ? currentLocation : null);
                        }
                    }
                    document.body.style.overflow = 'hidden';
                } else {
                    fullscreenElement.style.display = 'none';
                    document.body.style.overflow = 'auto';
                }
            }

            // Event listeners
            document.addEventListener('DOMContentLoaded', function() {
                initializeMap();
                
                document.getElementById('calculateHighwayBtn').addEventListener('click', calculateHighwayTriangulation);
                document.getElementById('calculateCompleteHighwayBtn').addEventListener('click', calculateCompleteHighway);
                document.getElementById('clearBtn').addEventListener('click', function() {
                    clearMap();
                    clearFullscreenMap();
                    document.getElementById('resultsContent').innerHTML = '<div class="text-center text-gray-500"><p>Select a highway location or analyze complete highway</p></div>';
                    document.getElementById('fullscreenResultsContent').innerHTML = '<div class="text-center text-gray-500"><p>Select a highway location or analyze complete highway</p></div>';
                    document.getElementById('highwayResultsSection').style.display = 'none';
                    document.getElementById('fullscreenHighwayResultsSection').style.display = 'none';
                    document.getElementById('gpsComparison').innerHTML = '';
                    document.getElementById('fullscreenGpsComparison').innerHTML = '';
                    document.getElementById('methodsResults').innerHTML = '';
                    document.getElementById('fullscreenMethodsResults').innerHTML = '';
                    document.getElementById('towerDetails').innerHTML = '';
                    document.getElementById('fullscreenTowerDetails').innerHTML = '';
                    currentData = null;
                    currentLocation = null;
                });
                document.getElementById('fullscreenBtn').addEventListener('click', toggleFullscreen);
                document.getElementById('exitFullscreenBtn').addEventListener('click', toggleFullscreen);

                // Load first location by default
                loadHighwayLocation(28.613939, 77.229508);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/calculate-complete-highway")
async def calculate_complete_highway():
    """Calculate triangulation for all highway points and return complete analysis"""
    try:
        tracker = HighPrecisionESIMTracker()
        result = tracker.calculate_complete_highway_triangulation()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Complete highway analysis error: {str(e)}")

@app.post("/calculate-highway-triangulation")
async def calculate_highway_triangulation(request: HighwayLocationRequest):
    try:
        tracker = HighPrecisionESIMTracker()
        
        # Get tower measurements for the specified location
        location = (request.latitude, request.longitude)
        tower_measurements = tracker.get_tower_measurements_for_location(location)
        
        if not tower_measurements:
            return {'error': f'No tower data found for location {location}'}
        
        # Set GPS location if requested
        gps_location = None
        if request.use_gps:
            gps_location = location
        
        # Perform triangulation
        result = tracker.advanced_triangulation(
            tower_measurements, 
            reference_location=location,
            gps_location=gps_location
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Highway triangulation error: {str(e)}")

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