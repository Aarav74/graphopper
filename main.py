from graphh import GraphHopper
mapper = GraphHopper("add4c752-8787-4a6b-ae81-6c8a357504b4")
while True:
    origin = mapper.address_to_latlong(input("Enter origin address: "))
    destination = mapper.address_to_latlong(input("Enter destination address: "))
    distance = mapper.distance([origin, destination], unit="km")
    routing = mapper.route([origin, destination])
    print(f"Distance: {distance} km")
    print(f"Route: {routing}")
    
    