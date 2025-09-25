import numpy as np

class KalmanFilter:
    """
    A simple Kalman filter for 2D position and velocity.
    The state is [x, y, vx, vy].
    """

    def __init__(self, process_variance, measurement_variance, initial_state):
        #                                Q                  R           state vector[x, y, vx, vy]
        """
        Initializes the Kalman filter.

        Args:
            process_variance (float): The uncertainty in the motion model (Q).
            measurement_variance (float): The uncertainty in the measurements (R).
            initial_state (np.array): The initial state vector [x, y, vx, vy].
        """
        # State vector [x, y, vx, vy]^T
        self.state_estimate = initial_state
        # State covariance matrix P
        self.P = np.eye(4) * 1000  # High initial uncertainty

        # Process noise covariance matrix Q
        self.Q = np.eye(4) * process_variance # Yeh predict step mein system mein aane wali uncertainty ko add karta hai.

        # Measurement noise covariance matrix R
        self.R = np.eye(2) * measurement_variance

        # Measurement matrix H
        self.H = np.array([[1, 0, 0, 0],  #state vector se kya measure kar rhe hai
                          [0, 1, 0, 0]])  #sirf position components ko select karta hai

    def predict(self, dt):
        """
        Predicts the next state of the system based on the motion model.

        Args:
            dt (float): The time step since the last update.
        """
        # State transition matrix F
        F = np.array([[1, 0, dt, 0],  # Yeh current state ko agle state mein project karta hai
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        # Predict the next state estimate
        self.state_estimate = F @ self.state_estimate

        # Predict the next state covariance
        self.P = F @ self.P @ F.T + self.Q  #Yeh line agle state estimate ko predict karti hai.

    def update(self, measurement):
        """
        Updates the state estimate with a new measurement.

        Args:
            measurement (np.array): The measurement vector [x, y]^T.
        """
        S = self.H @ self.P @ self.H.T + self.R #Yeh predicted measurement aur actual measurement ke beech ke difference (innovation) ki uncertainty ko calculate karta hai.
        K = self.P @ self.H.T @ np.linalg.inv(S) #Kalman Gain, "weight" jo btati hai ki prediction or measurement se kispe jyada trust kre

        # Update the state estimate
        self.state_estimate = self.state_estimate + K @ (measurement - self.H @ self.state_estimate)
 
        # Update the state covariance
        I = np.eye(4)  #4X4 identity matrix
        self.P = (I - K @ self.H) @ self.P  #update P

    def get_location(self):
        """
        Returns the current estimated location (x, y).
        """
        return self.state_estimate[0], self.state_estimate[1]

def get_user_input():
    """
    Prompts the user for a series of longitude, latitude, and time data.
    """
    data_points = []
    try:
        num_points = int(input("How many location-time pairs will you enter? "))
        if num_points <= 0:
            print("Please enter a positive number.")
            return None
        
        for i in range(num_points):
            print(f"\nEntering data for measurement {i + 1}:")
            latitude = float(input("Enter latitude: "))
            longitude = float(input("Enter longitude: "))
            
            # The timestamp is provided in milliseconds, so we convert it to seconds
            # for the Kalman filter's time step (dt).
            timestamp_ms = int(input("Enter timestamp (in milliseconds: "))
            timestamp_s = timestamp_ms / 1000.0 #converting the time into seconds
            data_points.append({ 'lat': latitude,'lon': longitude,'time': timestamp_s})
        
        return data_points
        
    except ValueError:
        print("\nInvalid input. Please enter a valid number.")
        return None

if __name__ == "__main__":
    
    measurements = get_user_input() #function ko call krke user se data le rhe hai
    
    if measurements and len(measurements) >= 2: #check krte hai user n valid data diya hai ya nhi
    
        initial_lat = measurements[0]['lat']
        initial_lon = measurements[0]['lon']
        
        initial_state = np.array([initial_lon, initial_lat, 0, 0])
        # Increased measurement variance to give the filter more "trust" in the prediction over the measurement
        kf = KalmanFilter(process_variance=0.1, measurement_variance=10.0, initial_state=initial_state) #hum filter ko bol rhe hai measurement m bhot noise hai

        print("\n--- Kalman Filter Location Estimation ---")
        
        # We process the first measurement separately
        last_timestamp = measurements[0]['time']
        
        # Process subsequent measurements
        for i in range(1, len(measurements)):  #dusri measurement k liye loop
            current_time = measurements[i]['time']
            dt = current_time - last_timestamp #dt = current aur previous measurement ke beech ka samay hai.
            
            if dt > 0: #if dt is not positive then skip the prediction step
                # Prediction step
                kf.predict(dt)
            
            # Update step with the new measurement
            measured_location = np.array([measurements[i]['lon'], measurements[i]['lat']]) #curr measurement to numpy array m convert kiya jaata hai
            kf.update(measured_location) #filter ko nayi measurement se correct kiya jaata hai
            
            last_timestamp = current_time
        
        # Only print the final result after the loop is complete
        final_estimated_location = kf.get_location()
        print(f"\nFinal Estimated Location: ({final_estimated_location[0]:.6f}, {final_estimated_location[1]:.6f})")

    elif measurements and len(measurements) < 2:
        print("Please provide at least two measurements to use the Kalman filter.")
