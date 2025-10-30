import numpy as np
import math
import requests
from typing import List, Tuple, Dict, Any, Optional #to define the variable type
from scipy.optimize import minimize #for optional algorithms
from scipy.spatial import KDTree  #to handle saptial data
import json

class HighPrecisionESIMTracker:
    def __init__(self):
        #to initialize the cell tower database
        self.cell_tower_database = self.initialize_tower_database()
        #to initialize signal propagation model
        self.signal_propagation_models = self.initialize_propagation_models()
    
    #cell tower database setup
    def initialize_tower_database(self) -> Dict:
        """Initialize a comprehensive database of cell tower locations"""
        return {
            # Delhi area cell towers - expanded with more realistic coverage
            '404_84_12345678': {'lat': 28.632400, 'lon': 77.218800, 'type': '4G', 'height': 35},
            '404_84_12345679': {'lat': 28.631500, 'lon': 77.217500, 'type': '4G', 'height': 32},
            '404_84_12345680': {'lat': 28.633500, 'lon': 77.220200, 'type': '4G', 'height': 40},
            '404_84_12345681': {'lat': 28.630800, 'lon': 77.219500, 'type': '4G', 'height': 38},
            '404_84_12345682': {'lat': 28.634200, 'lon': 77.217000, 'type': '4G', 'height': 45},
            '404_07_12345683': {'lat': 28.631200, 'lon': 77.221000, 'type': '4G', 'height': 42},
            '404_07_12345684': {'lat': 28.633800, 'lon': 77.216500, 'type': '4G', 'height': 36},
            '404_07_12345685': {'lat': 28.630500, 'lon': 77.218000, 'type': '4G', 'height': 39},
            '404_07_12345686': {'lat': 28.632800, 'lon': 77.222000, 'type': '4G', 'height': 44},
        }
    
    def initialize_propagation_models(self) -> Dict:
        """Initialize signal propagation models for different environments"""
        return {
            'dense_urban': {'path_loss_exp': 3.5, 'shadow_std': 12},
            'urban': {'path_loss_exp': 3.2, 'shadow_std': 10},
            'suburban': {'path_loss_exp': 2.8, 'shadow_std': 8},
            'rural': {'path_loss_exponent': 2.3, 'shadow_std': 6},
            'free_space': {'path_loss_exp': 2.0, 'shadow_std': 3}
        }
    
    #to calculate the distance
    def haversine_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """High precision Haversine distance calculation"""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        R = 6371000.0  # Earth radius in meters
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        #haversine formula to account the earth's curvature
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def signal_to_distance(self, signal_strength: float, frequency: float = 1800, 
                         environment: str = "urban", tower_height: float = 30) -> Tuple[float, float]:
        """
        Convert signal strength to distance with confidence interval
        Returns: (distance_meters, confidence_interval)
        """
        # Reference signal strength at 1 km under ideal conditions
        if frequency <= 900:
            ref_signal_1km = -75  # dBm at 1 km for 900MHz
        elif frequency <= 1800:
            ref_signal_1km = -80  # dBm at 1 km for 1800MHz
        else:
            ref_signal_1km = -85  # dBm at 1 km for 2100MHz+
        
        # Path loss exponent from propagation model
        path_loss_exp = self.signal_propagation_models[environment]['path_loss_exp']
        shadow_std = self.signal_propagation_models[environment]['shadow_std']
        
        # Calculate distance using log-distance path loss model
        path_loss = ref_signal_1km - signal_strength
        distance_km = 10 ** (path_loss / (10 * path_loss_exp))
        distance_meters = distance_km * 1000
        
        # Adjust for tower height (higher towers have better coverage)
        height_factor = max(0.8, min(1.5, tower_height / 35.0))  # More realistic bounds
        distance_meters *= height_factor
        
        # Calculate confidence interval based on environment and signal quality
        base_confidence = 0.25  # 25% base uncertainty
        if signal_strength > -70:  # Strong signal
            base_confidence = 0.15
        elif signal_strength < -90:  # Weak signal
            base_confidence = 0.4
            
        confidence = distance_meters * base_confidence
        
        return max(50, distance_meters), confidence  # Minimum 50m distance
    
    #main triangulation method
    def advanced_triangulation(self, tower_measurements: List[Dict]) -> Dict[str, Any]:
        """
        Advanced triangulation using multiple algorithms for maximum accuracy
        """
        print(f"Starting triangulation with {len(tower_measurements)} measurements...")
        
        # Step 1: Prepare tower data with distances
        towers = []
        valid_towers_count = 0
        
        for meas in tower_measurements:
            #create tower ID
            tower_id = f"{meas['mcc']}_{meas['mnc']}_{meas['cell_id']}"
            #check if the tower is in database
            if tower_id in self.cell_tower_database:
                tower_data = self.cell_tower_database[tower_id]
                #calculate distance using signal strength
                distance, confidence = self.signal_to_distance(
                    meas['signal_strength'],
                    meas.get('frequency', 1800),
                    meas.get('environment', 'urban'),
                    tower_data['height']
                )
                
                #store tower information
                towers.append({
                    'id': tower_id,
                    'coordinates': (tower_data['lat'], tower_data['lon']),
                    'distance': distance,
                    'confidence': confidence,
                    'signal_strength': meas['signal_strength'],
                    'timing_advance': meas.get('timing_advance'),
                    'weight': 1.0 / (confidence + 1e-6)  # Higher confidence = higher weight
                })
                valid_towers_count += 1
                print(f"  Found tower {tower_id} at {tower_data['lat']:.6f}, {tower_data['lon']:.6f}")
                print(f"    Signal: {meas['signal_strength']}dBm -> Distance: {distance:.1f}m")
        
        print(f"Valid towers found: {valid_towers_count}")
        
        #check if there is enough towers
        if len(towers) < 3:
            print(f"Warning: Only {len(towers)} valid towers found. Using fallback method.")
            return self.enhanced_fallback_triangulation(towers, tower_measurements)
        
        # Step 2: Use multiple triangulation methods
        results = {}
        
        # Method 1: Weighted Least Squares
        print("Running Weighted Least Squares...")
        results['weighted_least_squares'] = self.weighted_least_squares(towers)
        
        # Method 2: Circular Intersection
        print("Running Circular Intersection...")
        results['circular_intersection'] = self.circular_intersection(towers)
        
        # Method 3: Centroid with weights
        print("Running Weighted Centroid...")
        results['weighted_centroid'] = self.weighted_centroid(towers)
        
        # Method 4: Non-linear optimization
        print("Running Non-linear Optimization...")
        results['non_linear_opt'] = self.non_linear_optimization(towers)
        
        # Step 3: Combine results using confidence-based weighting
        final_result = self.combine_triangulation_results(results, towers)
        
        print(f"Triangulation completed. Final accuracy: {final_result['accuracy_meters']:.1f}m")
        return final_result
    
    def weighted_least_squares(self, towers: List[Dict]) -> Dict[str, Any]:
        """Weighted least squares triangulation"""
        try:
            # Initial guess (weighted centroid)
            lat_sum, lon_sum, total_weight = 0, 0, 0
            for tower in towers:
                weight = tower['weight']
                lat_sum += tower['coordinates'][0] * weight
                lon_sum += tower['coordinates'][1] * weight
                total_weight += weight
            
            #for starting point optimization
            x0 = np.array([lat_sum / total_weight, lon_sum / total_weight])
            
            def objective_function(x):
                total_error = 0
                lat, lon = x #current position which is being tested
                
                for tower in towers:
                    #calculate distance from tower to destination
                    tower_lat, tower_lon = tower['coordinates']
                    calculated_distance = self.haversine_distance(
                        (lat, lon), (tower_lat, tower_lon)
                    )
                    #squared error between the calculated and estimated distance
                    error = (calculated_distance - tower['distance']) ** 2
                    #weight according to the weight
                    total_error += error * tower['weight']
                
                return total_error
            
            # Constrain to reasonable area around towers
            lat_coords = [t['coordinates'][0] for t in towers]
            lon_coords = [t['coordinates'][1] for t in towers]
            
            bounds = [
                (min(lat_coords) - 0.005, max(lat_coords) + 0.005),  # ~500m buffer
                (min(lon_coords) - 0.005, max(lon_coords) + 0.005)
            ]
            #run optimization for minimum error
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
            # Try multiple combinations of towers for better results
            best_result = {'location': None, 'accuracy': float('inf'), 'method': 'circular_intersection'}
            
            #try every possibel connection of three towers
            for i in range(len(towers)):
                for j in range(i+1, len(towers)):
                    for k in range(j+1, len(towers)):
                        t1, t2, t3 = towers[i], towers[j], towers[k]
                        
                        # Get circle intersections
                        intersections = self.three_circle_intersection(t1, t2, t3)
                        
                        if intersections:
                            # Choose the intersection that minimizes total error
                            best_intersection = None
                            min_error = float('inf')
                            
                            for point in intersections:
                                error = 0
                                for tower in towers:
                                    calculated_dist = self.haversine_distance(point, tower['coordinates'])
                                    error += abs(calculated_dist - tower['distance']) * tower['weight']
                                
                                if error < min_error:
                                    min_error = error
                                    best_intersection = point
                            
                            if best_intersection:
                                accuracy = self.calculate_accuracy_metrics(best_intersection, towers)
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
            #exrtract coordinates 
            x1, y1 = t1['coordinates'][1], t1['coordinates'][0]  # lon, lat
            x2, y2 = t2['coordinates'][1], t2['coordinates'][0]
            x3, y3 = t3['coordinates'][1], t3['coordinates'][0]
            
            # Convert distances to degrees (more accurate conversion)
            r1 = t1['distance'] / 111320  # 1 degree ≈ 111.32km at equator
            r2 = t2['distance'] / 111320
            r3 = t3['distance'] / 111320
            
            # Calculate intersection of first two circles
            dx = x2 - x1
            dy = y2 - y1
            d = math.sqrt(dx*dx + dy*dy) # Distance between centers
            
            #check if circles intersect or not
            if d > (r1 + r2) or d < abs(r1 - r2):
                return []  # No intersection
            
            # Find intersection points
            a = (r1*r1 - r2*r2 + d*d) / (2 * d)
            h = math.sqrt(r1*r1 - a*a)
            
            #center point between intersections
            x0 = x1 + a * dx / d
            y0 = y1 + a * dy / d
            
            #center point to offset
            rx = -dy * (h / d)
            ry = dx * (h / d)
            
            #possible intersection  points
            points = [
                (y0 + ry, x0 + rx),  # (lat, lon)
                (y0 - ry, x0 - rx)   # (lat, lon)
            ]
            
            # Filter points that are reasonable for all towers
            valid_points = []
            for point in points:
                total_error = 0
                valid = True
                
                for tower in [t1, t2, t3]:
                    dist_to_tower = self.haversine_distance(point, (tower['coordinates'][0], tower['coordinates'][1]))
                    error_ratio = abs(dist_to_tower - tower['distance']) / tower['distance']
                    if error_ratio > 0.8:  # Allow 80% error margin
                        valid = False
                        break
                    total_error += error_ratio
                
                if valid and total_error < 1.5:  # Reasonable total error
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
                # Weight based on signal strength and confidence
                weight = tower['weight']
                lat_sum += tower['coordinates'][0] * weight
                lon_sum += tower['coordinates'][1] * weight
                total_weight += weight
            
            if total_weight > 0:
                #calculate weighted average
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
            # Initial guess from weighted centroid
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
                    # use log error fro better numerical stability
                    error = math.log1p(abs(calculated_distance - tower['distance']))
                    total_error += error * tower['weight']
                
                return total_error
            
            # Constrain search area
            lat_coords = [t['coordinates'][0] for t in towers]
            lon_coords = [t['coordinates'][1] for t in towers]
            
            bounds = [
                (min(lat_coords) - 0.005, max(lat_coords) + 0.005),
                (min(lon_coords) - 0.005, max(lon_coords) + 0.005)
            ]
            
            #run optimization
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
        
        #do check valid results
        for method, result in results.items():
            if result['location'] is not None and result['accuracy'] < float('inf'):
                valid_results.append(result)
                print(f"  {method}: Location {result['location']}, Accuracy: {result['accuracy']:.1f}m")
        
        #use fallback if no valid results
        if not valid_results:
            print("All triangulation methods failed. Using enhanced fallback.")
            return self.enhanced_fallback_triangulation(towers, [])
        
        # Weight results by their accuracy (lower accuracy value = better)
        total_weight = 0
        weighted_lat, weighted_lon = 0, 0
        
        for result in valid_results:
            weight = 1.0 / (result['accuracy'] + 1e-6) 
            weighted_lat += result['location'][0] * weight
            weighted_lon += result['location'][1] * weight
            total_weight += weight
        
        if total_weight > 0:
            #weighted average of every successful result
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
                        'signal_strength': t['signal_strength']
                    } for t in towers
                ]
            }
        else:
            return self.enhanced_fallback_triangulation(towers, [])
    
    def enhanced_fallback_triangulation(self, towers: List[Dict], tower_measurements: List[Dict]) -> Dict[str, Any]:
        """Enhanced fallback method with distance calculation"""
        enhanced_towers = []
        
        for tower in towers:
            # Calculate distance for fallback method too
            distance, confidence = self.signal_to_distance(
                tower['signal_strength'], 1800, 'urban', 35
            )
            enhanced_towers.append({
                'coordinates': tower['coordinates'],
                'distance': distance,
                'weight': 1.0 / (confidence + 1e-6)
            })
        
        if not enhanced_towers:
            # If no towers with distance, use simple centroid
            all_towers = []
            for meas in tower_measurements:
                tower_id = f"{meas['mcc']}_{meas['mnc']}_{meas['cell_id']}"
                if tower_id in self.cell_tower_database:
                    tower_data = self.cell_tower_database[tower_id]
                    all_towers.append({
                        'coordinates': (tower_data['lat'], tower_data['lon']),
                        'signal_strength': meas['signal_strength']
                    })
            
            if not all_towers:
                return {'error': 'No valid tower data available'}
            
            # Simple centroid
            lat_sum = sum(t['coordinates'][0] for t in all_towers)
            lon_sum = sum(t['coordinates'][1] for t in all_towers)
            estimated_location = (lat_sum / len(all_towers), lon_sum / len(all_towers))
            
            return {
                'estimated_location': estimated_location,
                'accuracy_meters': 1500,  # High uncertainty for fallback
                'number_of_towers': len(all_towers),
                'methods_used': ['fallback_centroid'],
                'confidence': 'low',
                'tower_details': []
            }
        
        # Use weighted centroid with calculated distances
        return self.weighted_centroid(enhanced_towers)
    
    def calculate_accuracy_metrics(self, location: Tuple[float, float], towers: List[Dict]) -> float:
        """Calculate accuracy metrics for estimated location"""
        errors = []
        
        for tower in towers:
            #Estimated location se tower tak ki actual distance 
            calculated_distance = self.haversine_distance(location, tower['coordinates'])
            #error between the calculated and estimated distance
            error = abs(calculated_distance - tower['distance'])
            errors.append(error)
        
        # Use RMS(Root Mean Square) error as accuracy metric
        if errors:
            rms_error = math.sqrt(sum(e**2 for e in errors) / len(errors))
        else:
            rms_error = 1000  # Default high error
        
        # Adjust based on number of towers and geometry
        geometry_factor = self.calculate_geometry_factor(towers)
        
        return max(10, rms_error * geometry_factor)  # Minimum 10m accuracy
    
    def calculate_geometry_factor(self, towers: List[Dict]) -> float:
        """Calculate geometry dilution of precision factor"""
        if len(towers) < 3:
            return 2.5  # if tower is less than 3 Poor geometry
        
        # Calculate area covered by towers
        lats = [t['coordinates'][0] for t in towers]
        lons = [t['coordinates'][1] for t in towers]
        
        lat_span = max(lats) - min(lats)
        lon_span = max(lons) - min(lons)
        
        area = lat_span * lon_span
        
        if area < 0.00005:  # Very small area (towers too close)
            return 2.0
        elif area < 0.0002:  # Small area
            return 1.5
        elif area < 0.001:  # Medium area
            return 1.2
        else:  # Good spread
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
        if not gps_locations:
            return esim_location
        
        # Use weighted average to combine ESIM and GPS
        esim_point = esim_location['estimated_location']
        gps_point = gps_locations[-1]  # Use most recent GPS
        
        esim_accuracy = esim_location['accuracy_meters']
        gps_accuracy = 8  # Assume GPS accuracy of 8 meters (better than before)
        
        # Weighted average based on accuracy (more accurate = higher weight)
        esim_weight = 1.0 / (esim_accuracy + 1e-6)
        gps_weight = 1.0 / (gps_accuracy + 1e-6)
        
        total_weight = esim_weight + gps_weight
        
        #calculate weighted average position
        fused_lat = (esim_point[0] * esim_weight + gps_point[0] * gps_weight) / total_weight
        fused_lon = (esim_point[1] * esim_weight + gps_point[1] * gps_weight) / total_weight
        
        # Improved accuracy (weighted combination)
        fused_accuracy = (esim_accuracy * esim_weight + gps_accuracy * gps_weight) / total_weight
        
        #add fused results to original ESIM location
        esim_location['fused_location'] = (fused_lat, fused_lon)
        esim_location['fused_accuracy_meters'] = fused_accuracy
        esim_location['fusion_used'] = True
        
        return esim_location

# Usage Example and Test
def main():
    # Initialize the tracker
    esim_tracker = HighPrecisionESIMTracker()
    
    # Sample ESIM measurement data with realistic signal strengths
    sample_esim_measurements = [
        {
            'mcc': '404',  # Mobile Country Code (India)
            'mnc': '84',   # Mobile Network Code (Jio)
            'cell_id': '12345678',
            'signal_strength': -58,  # Strong signal (close to actual location)
            'timing_advance': 1,
            'frequency': 1800,
            'environment': 'urban'
        },
        {
            'mcc': '404',
            'mnc': '84', 
            'cell_id': '12345679',
            'signal_strength': -62,  # Medium-strong signal
            'timing_advance': 2,
            'frequency': 1800,
            'environment': 'urban'
        },
        {
            'mcc': '404',
            'mnc': '84',
            'cell_id': '12345680',
            'signal_strength': -65,  # Medium signal
            'timing_advance': 2, 
            'frequency': 1800,
            'environment': 'urban'
        },
        {
            'mcc': '404',
            'mnc': '07',
            'cell_id': '12345683',
            'signal_strength': -60,  # Strong signal
            'timing_advance': 1,
            'frequency': 2100,
            'environment': 'urban'
        }
    ]
    
    # Sample GPS data for fusion
    sample_gps_locations = [
        (28.632429, 77.218788),  # Actual GPS coordinates
        (28.632435, 77.218792),  # Slight movement
        (28.632440, 77.218795)   # More movement
    ]
    
    print("=== High Precision ESIM Location Tracking ===")
    print(f"Processing {len(sample_esim_measurements)} tower measurements...")
    
    # Get ESIM location
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
    
    # Additional diagnostics
    print(f"\nTower Details:")
    for i, tower in enumerate(esim_result.get('tower_details', [])):
        print(f"  Tower {i+1}: {tower['tower_id']}")
        print(f"    Signal: {tower['signal_strength']}dBm, Est. Distance: {tower['distance_estimated']:.1f}m")

if __name__ == "__main__":
    main()