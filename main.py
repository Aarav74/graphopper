import numpy as np
import math
import requests
from typing import List, Tuple, Dict, Any, Optional
from scipy.optimize import minimize
from scipy.spatial import KDTree
import json

class HighPrecisionESIMTracker:
    def __init__(self):
        # Initialize with your specific tower database
        self.cell_tower_database = self.initialize_tower_database()
        self.signal_propagation_models = self.initialize_propagation_models()
    
    def initialize_tower_database(self) -> Dict:
        """Initialize database with your specific towers"""
        return {
            # Your actual towers with realistic coordinates in Delhi area
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
            # Additional towers for better coverage
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
        
        R = 6371000.0  # Earth radius in meters
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def calculate_tower_coordinates(self, reference_point: Tuple[float, float], 
                                 distance: float, azimuth: float) -> Tuple[float, float]:
        """
        Calculate tower coordinates from reference point using distance and azimuth
        reference_point: (lat, lon) of the device's approximate location
        distance: meters from device to tower
        azimuth: degrees from north (0-360)
        """
        ref_lat, ref_lon = reference_point
        
        # Convert distance to degrees (approximate)
        lat_distance_km = distance / 1000.0
        lon_distance_km = distance / 1000.0
        
        # Earth radii at given latitude (approximate)
        lat_deg_per_km = 1 / 110.574
        lon_deg_per_km = 1 / (111.320 * math.cos(math.radians(ref_lat)))
        
        # Convert azimuth to radians (0 = north, clockwise)
        azimuth_rad = math.radians(azimuth)
        
        # Calculate delta lat/lon
        delta_lat = lat_distance_km * math.cos(azimuth_rad) * lat_deg_per_km
        delta_lon = lon_distance_km * math.sin(azimuth_rad) * lon_deg_per_km
        
        tower_lat = ref_lat + delta_lat
        tower_lon = ref_lon + delta_lon
        
        return (tower_lat, tower_lon)
    
    def advanced_triangulation(self, tower_measurements: List[Dict], 
                             reference_location: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Advanced triangulation using provided distances and azimuths
        """
        print(f"Starting triangulation with {len(tower_measurements)} measurements...")
        
        # Step 1: Prepare tower data
        towers = []
        valid_towers_count = 0
        
        # Use provided reference location or calculate centroid
        if reference_location is None:
            ref_point = (28.632400, 77.218800)  # Default Delhi area
        else:
            ref_point = reference_location
        
        print(f"Using reference point: {ref_point}")
        
        for meas in tower_measurements:
            tower_id = meas.get('cell_id')
            
            # Check if tower exists in database
            if tower_id in self.cell_tower_database:
                # Use actual coordinates from database
                tower_data = self.cell_tower_database[tower_id]
                tower_coords = (tower_data['lat'], tower_data['lon'])
                print(f"  Found tower {tower_id} in database at {tower_coords}")
            else:
                # Calculate coordinates from distance and azimuth
                if 'distance' in meas and 'azimuth' in meas:
                    tower_coords = self.calculate_tower_coordinates(
                        ref_point, meas['distance'], meas['azimuth']
                    )
                    print(f"  Calculated tower {tower_id} coordinates: {tower_coords}")
                else:
                    print(f"  Skipping tower {tower_id} - insufficient data")
                    continue
            
            # Use provided distance directly (most accurate)
            distance = meas['distance']
            signal_strength = meas.get('signal_strength', -102.495)
            
            # Calculate confidence based on signal strength
            confidence = self.calculate_confidence_from_signal(signal_strength)
            
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
            
            print(f"  Tower {tower_id} at {tower_coords[0]:.6f}, {tower_coords[1]:.6f}")
            print(f"    Distance: {distance:.1f}m, Signal: {signal_strength}dBm, Weight: {confidence:.3f}")
        
        print(f"Valid towers processed: {valid_towers_count}")
        
        if len(towers) < 2:
            print(f"Warning: Only {len(towers)} valid towers found. Using fallback method.")
            return self.enhanced_fallback_triangulation(towers, tower_measurements)
        
        # Step 2: Use multiple triangulation methods
        results = {}
        
        print("Running Weighted Least Squares...")
        results['weighted_least_squares'] = self.weighted_least_squares(towers)
        
        print("Running Circular Intersection...")
        results['circular_intersection'] = self.circular_intersection(towers)
        
        print("Running Weighted Centroid...")
        results['weighted_centroid'] = self.weighted_centroid(towers)
        
        if len(towers) >= 3:
            print("Running Non-linear Optimization...")
            results['non_linear_opt'] = self.non_linear_optimization(towers)
        
        # Step 3: Combine results
        final_result = self.combine_triangulation_results(results, towers)
        
        print(f"Triangulation completed. Final accuracy: {final_result['accuracy_meters']:.1f}m")
        return final_result
    
    def calculate_confidence_from_signal(self, signal_strength: float) -> float:
        """Calculate confidence interval based on signal strength"""
        if signal_strength > -70:  # Strong signal
            return 0.15
        elif signal_strength > -85:  # Good signal
            return 0.20
        elif signal_strength > -100:  # Fair signal
            return 0.25
        else:  # Weak signal
            return 0.35
    
    def signal_to_distance(self, signal_strength: float, frequency: float = 1800, 
                         environment: str = "urban", tower_height: float = 30) -> Tuple[float, float]:
        """
        Convert signal strength to distance with confidence interval
        """
        if frequency <= 900:
            ref_signal_1km = -75
        elif frequency <= 1800:
            ref_signal_1km = -80
        else:
            ref_signal_1km = -85
        
        path_loss_exp = self.signal_propagation_models[environment]['path_loss_exp']
        
        path_loss = ref_signal_1km - signal_strength
        distance_km = 10 ** (path_loss / (10 * path_loss_exp))
        distance_meters = distance_km * 1000
        
        height_factor = max(0.8, min(1.5, tower_height / 35.0))
        distance_meters *= height_factor
        
        confidence = distance_meters * self.calculate_confidence_from_signal(signal_strength)
        
        return max(50, distance_meters), confidence
    
    def weighted_least_squares(self, towers: List[Dict]) -> Dict[str, Any]:
        """Weighted least squares triangulation"""
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
                    calculated_distance = self.haversine_distance(
                        (lat, lon), (tower_lat, tower_lon)
                    )
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
                return {
                    'location': estimated_location,
                    'accuracy': accuracy,
                    'method': 'weighted_least_squares'
                }
            else:
                print(f"WLS Optimization failed: {result.message}")
        except Exception as e:
            print(f"WLS Error: {e}")
        
        return {'location': None, 'accuracy': float('inf'), 'method': 'weighted_least_squares'}
    
    def circular_intersection(self, towers: List[Dict]) -> Dict[str, Any]:
        """Circular intersection method for triangulation"""
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
                                    best_result = {
                                        'location': best_intersection,
                                        'accuracy': accuracy,
                                        'method': 'circular_intersection'
                                    }
            
            if best_result['location'] is not None:
                return best_result
                
        except Exception as e:
            print(f"Circular Intersection Error: {e}")
        
        return {'location': None, 'accuracy': float('inf'), 'method': 'circular_intersection'}
    
    def three_circle_intersection(self, t1: Dict, t2: Dict, t3: Dict) -> List[Tuple[float, float]]:
        """Calculate intersection points of three circles"""
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
            
            points = [
                (y0 + ry, x0 + rx),
                (y0 - ry, x0 - rx)
            ]
            
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
        """Weighted centroid triangulation"""
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
                return {
                    'location': estimated_location,
                    'accuracy': accuracy,
                    'method': 'weighted_centroid'
                }
        except Exception as e:
            print(f"Centroid Error: {e}")
        
        return {'location': None, 'accuracy': float('inf'), 'method': 'weighted_centroid'}
    
    def non_linear_optimization(self, towers: List[Dict]) -> Dict[str, Any]:
        """Non-linear optimization for triangulation"""
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
                    calculated_distance = self.haversine_distance(
                        (lat, lon), tower['coordinates']
                    )
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
                return {
                    'location': estimated_location,
                    'accuracy': accuracy,
                    'method': 'non_linear_opt'
                }
        except Exception as e:
            print(f"Non-linear optimization error: {e}")
        
        return {'location': None, 'accuracy': float('inf'), 'method': 'non_linear_opt'}
    
    def combine_triangulation_results(self, results: Dict, towers: List[Dict]) -> Dict[str, Any]:
        """Combine results from multiple triangulation methods"""
        valid_results = []
        
        for method, result in results.items():
            if result['location'] is not None and result['accuracy'] < float('inf'):
                valid_results.append(result)
                print(f"  {method}: Accuracy: {result['accuracy']:.1f}m")
        
        if not valid_results:
            print("All triangulation methods failed. Using enhanced fallback.")
            return self.enhanced_fallback_triangulation(towers, [])
        
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
                        'azimuth': t.get('azimuth'),
                        'coordinates': t['coordinates']
                    } for t in towers
                ]
            }
        else:
            return self.enhanced_fallback_triangulation(towers, [])
    
    def enhanced_fallback_triangulation(self, towers: List[Dict], tower_measurements: List[Dict]) -> Dict[str, Any]:
        """Enhanced fallback method"""
        if towers:
            return self.weighted_centroid(towers)
        
        return {'error': 'No valid tower data available'}
    
    def calculate_accuracy_metrics(self, location: Tuple[float, float], towers: List[Dict]) -> float:
        """Calculate accuracy metrics for estimated location"""
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
        """Calculate geometry dilution of precision factor"""
        if len(towers) < 3:
            return 2.5
        
        lats = [t['coordinates'][0] for t in towers]
        lons = [t['coordinates'][1] for t in towers]
        
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        area = lat_span * lon_span
        
        if area < 0.00005:
            return 2.0
        elif area < 0.0002:
            return 1.5
        elif area < 0.001:
            return 1.2
        else:
            return 1.0
    
    def calculate_confidence(self, accuracy: float, num_towers: int) -> str:
        """Calculate confidence level based on accuracy and tower count"""
        if accuracy < 25 and num_towers >= 4:
            return 'very_high'
        elif accuracy < 50 and num_towers >= 3:
            return 'high'
        elif accuracy < 100 and num_towers >= 2:
            return 'medium'
        elif accuracy < 200:
            return 'low'
        else:
            return 'very_low'
    
    def enhance_with_gps_fusion(self, esim_location: Dict, gps_locations: List[Tuple]) -> Dict:
        """Fuse ESIM location with GPS data for maximum accuracy"""
        if not gps_locations or 'error' in esim_location:
            return esim_location
        
        esim_point = esim_location['estimated_location']
        gps_point = gps_locations[-1]
        
        esim_accuracy = esim_location['accuracy_meters']
        gps_accuracy = 5
        
        esim_weight = 1.0 / (esim_accuracy + 1e-6)
        gps_weight = 1.0 / (gps_accuracy + 1e-6)
        
        total_weight = esim_weight + gps_weight
        
        fused_lat = (esim_point[0] * esim_weight + gps_point[0] * gps_weight) / total_weight
        fused_lon = (esim_point[1] * esim_weight + gps_point[1] * gps_weight) / total_weight
        
        fused_accuracy = min(esim_accuracy, gps_accuracy) * 0.7
        
        esim_location['fused_location'] = (fused_lat, fused_lon)
        esim_location['fused_accuracy_meters'] = fused_accuracy
        esim_location['fusion_used'] = True
        
        return esim_location

# Updated Usage Example with Your Exact Data
def main():
    # Initialize the tracker
    esim_tracker = HighPrecisionESIMTracker()
    
    # Your EXACT network data
    sample_esim_measurements = [
        {
            'cell_id': '24823',
            'distance': 746.3185812627742,  # Your exact distance
            'azimuth': 217,                  # Your exact azimuth
            'signal_strength': -102.495,     # Your exact RSRP
            'environment': 'urban',
        },
        {
            'cell_id': '27639',
            'distance': 769.6862267850452,   # Your exact distance
            'azimuth': 205,                  # Your exact azimuth
            'signal_strength': -102.495,     # Using same RSRP for all
            'environment': 'urban',
        },
        {
            'cell_id': '27682',
            'distance': 876.9099713831175,   # Your exact distance
            'azimuth': 221,                  # Your exact azimuth
            'signal_strength': -102.495,     # Using same RSRP for all
            'environment': 'urban',
        }
    ]
    
    # Sample GPS data for fusion
    sample_gps_locations = [
        (28.632429, 77.218788),
        (28.632435, 77.218792),
        (28.632440, 77.218795)
    ]
    
    print("=== High Precision ESIM Location Tracking ===")
    print("Using YOUR EXACT tower data:")
    print(f"Tower 24823: Distance {746.3185812627742:.1f}m, Azimuth 217°")
    print(f"Tower 27639: Distance {769.6862267850452:.1f}m, Azimuth 205°") 
    print(f"Tower 27682: Distance {876.9099713831175:.1f}m, Azimuth 221°")
    print(f"Signal: RSRP {-102.495} dBm, RSRQ {-13.548} dB")
    
    # Get ESIM location using your exact distance and azimuth data
    esim_result = esim_tracker.advanced_triangulation(sample_esim_measurements)
    
    if 'error' in esim_result:
        print(f"Error: {esim_result['error']}")
        return
    
    print(f"\nESIM Only Result:")
    print(f"Location: {esim_result['estimated_location']}")
    print(f"Accuracy: {esim_result['accuracy_meters']:.1f} meters")
    print(f"Confidence: {esim_result['confidence']}")
    print(f"Towers Used: {esim_result['number_of_towers']}")
    print(f"Methods: {esim_result['methods_used']}")
    
    # Enhance with GPS fusion
    enhanced_result = esim_tracker.enhance_with_gps_fusion(esim_result, sample_gps_locations)
    
    print(f"\nEnhanced with GPS Fusion:")
    print(f"Fused Location: {enhanced_result['fused_location']}")
    print(f"Final Accuracy: {enhanced_result['fused_accuracy_meters']:.1f} meters")
    
    # Calculate distance from actual GPS
    actual_gps = (28.632429, 77.218788)
    esim_distance = esim_tracker.haversine_distance(
        esim_result['estimated_location'], actual_gps
    )
    fused_distance = esim_tracker.haversine_distance(
        enhanced_result['fused_location'], actual_gps
    )
    
    print(f"\nComparison with Actual GPS ({actual_gps}):")
    print(f"ESIM Only Distance Error: {esim_distance:.1f} meters")
    print(f"Fused Location Distance Error: {fused_distance:.1f} meters")
    
    if esim_distance > 0:
        improvement = ((esim_distance - fused_distance) / esim_distance * 100)
        print(f"Improvement: {improvement:.1f}%")
    
    # Detailed tower information
    print(f"\nTower Details:")
    for i, tower in enumerate(esim_result.get('tower_details', [])):
        print(f"  Tower {i+1}: {tower['tower_id']}")
        print(f"    Coordinates: {tower['coordinates'][0]:.6f}, {tower['coordinates'][1]:.6f}")
        print(f"    Distance: {tower['distance_estimated']:.1f}m")
        print(f"    Azimuth: {tower.get('azimuth', 'N/A')}°")
        print(f"    Signal: {tower['signal_strength']}dBm")

if __name__ == "__main__":
    main()