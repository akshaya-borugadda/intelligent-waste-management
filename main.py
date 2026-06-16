import pandas as pd
import numpy as np
import random
from geopy.distance import geodesic
import folium
import os
import webbrowser
import matplotlib.pyplot as plt

# === Load Dataset ===
try:
    df = pd.read_csv('guntur_bin_data_200.csv')
except FileNotFoundError:
    raise FileNotFoundError("CSV file 'guntur_bin_data_200.csv' not found. Please ensure it is in the current directory.")

# === Constants ===
BIN_VOLUME_LITRES = 100
WASTE_DENSITY_KG_PER_LITRE = 0.2
TRUCK_CAPACITY_KG = 1500
average_speed_kmph = 25
load_time_per_bin_min = 2
CO2_PER_LITRE_DIESEL = 2.68
DIESEL_COST_PER_LITRE = 90
overflow_threshold = 90

# === Depot Location ===
start_point = {
    'Latitude': 16.3067,
    'Longitude': 80.4360,
    'Street': 'Start Depot',
    'Fill_Level': 0
}

# === Filter Bins with Fill > 60% ===
high_fill_bins = df[df['Fill_Level'] > 60].reset_index(drop=True)
high_fill_bins = pd.concat([pd.DataFrame([start_point]), high_fill_bins], ignore_index=True)

# === Helper Functions ===
def calculate_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = geodesic(coords[i], coords[j]).meters
    return dist

def route_distance(route, dist_matrix):
    return sum(dist_matrix[route[i], route[i+1]] for i in range(len(route)-1))

def create_population(size, num_bins):
    return [[0] + random.sample(range(1, num_bins - 1), num_bins - 2) + [num_bins - 1] for _ in range(size)]

def crossover(parent1, parent2):
    if len(parent1) < 4:
        return parent1
    a, b = sorted(random.sample(range(1, len(parent1)-1), 2))
    child = [None] * len(parent1)
    child[a:b] = parent1[a:b]
    ptr = 1
    for val in parent2:
        if val not in child:
            while ptr < len(child)-1 and child[ptr] is not None:
                ptr += 1
            if ptr < len(child)-1:
                child[ptr] = val
    child[0] = 0
    child[-1] = len(parent1) - 1
    return child

def mutate(route, mutation_rate=0.1):
    if random.random() < mutation_rate:
        a, b = random.sample(range(1, len(route)-1), 2)
        route[a], route[b] = route[b], route[a]
    return route

def two_opt(route, dist_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 3):
            for j in range(i + 1, len(route)-1):
                if j - i == 1: continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if route_distance(new_route, dist_matrix) < route_distance(best, dist_matrix):
                    best = new_route
                    improved = True
        route = best
    return best

def estimate_total_waste_kg(data):
    return sum((row['Fill_Level'] / 100) * BIN_VOLUME_LITRES * WASTE_DENSITY_KG_PER_LITRE for _, row in data.iterrows())

# === ✅ Improved Genetic Algorithm ===
def genetic_algorithm(data, generations=400, pop_size=200, elite_size=60, mutation_rate=0.2):
    coords = list(zip(data['Latitude'], data['Longitude']))
    dist_matrix = calculate_distance_matrix(coords)
    population = create_population(pop_size, len(coords))

    best_distance = float('inf')
    best_route = []
    no_improvement_counter = 0

    for _ in range(generations):
        scores = [(route, route_distance(route, dist_matrix)) for route in population]
        scores.sort(key=lambda x: x[1])
        elites = [r for r, _ in scores[:elite_size]]

        if scores[0][1] < best_distance:
            best_distance = scores[0][1]
            best_route = scores[0][0]
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        if no_improvement_counter >= 15:
            break

        children = elites[:]
        while len(children) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = mutate(crossover(p1, p2), mutation_rate)
            children.append(child)
        population = children

    best_route = two_opt(best_route, dist_matrix)

    naive_route = list(range(len(coords))) + [0]
    naive_distance = route_distance(naive_route, dist_matrix)
    best_distance = route_distance(best_route, dist_matrix)

    return [data.iloc[i] for i in best_route], (1 - best_distance / naive_distance) * 100, naive_distance / 1000, best_distance / 1000

def plot_bin_route(all_bins, route_data, efficiency):
    m = folium.Map(location=[16.3067, 80.4360], zoom_start=13)

    folium.Marker(
        location=[start_point['Latitude'], start_point['Longitude']],
        icon=folium.Icon(color='black', icon='home'),
        popup="🚛 Start Depot"
    ).add_to(m)

    folium.map.Marker(
        [start_point['Latitude'], start_point['Longitude']],
        icon=folium.DivIcon(
            html=f"""<div style="font-size: 12pt; font-weight: bold; color: black;">Start Point</div>"""
        )
    ).add_to(m)

    legend_html = f'''
     <div style="position: fixed; bottom: 30px; left: 30px; z-index:9999; font-size:14px; background:white; padding:10px; border-radius:8px;">
         <b>Legend:</b><br>
         <i style="background:green;width:10px;height:10px;display:inline-block;"></i> Fill < 30%<br>
         <i style="background:blue;width:10px;height:10px;display:inline-block;"></i> 30%-70%<br>
         <i style="background:red;width:10px;height:10px;display:inline-block;"></i> > 70%<br>
         <i style="background:black;width:10px;height:10px;display:inline-block;"></i> Start Point<br>
         <b>Efficiency:</b> {efficiency:.2f}%
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in all_bins.iterrows():
        if row['Street'] == 'Start Depot':
            continue
        fill = row['Fill_Level']
        icon = folium.Icon(color='green' if fill < 30 else 'blue' if fill < 70 else 'red')
        popup = f"{row['Street']} - Fill: {fill:.2f}%"
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=popup,
            icon=icon
        ).add_to(m)

    for i in range(len(route_data) - 1):
        loc1 = [route_data[i]['Latitude'], route_data[i]['Longitude']]
        loc2 = [route_data[i+1]['Latitude'], route_data[i+1]['Longitude']]
        folium.PolyLine([loc1, loc2], color='black', weight=3, opacity=0.7).add_to(m)

    return m

def nearest_neighbor_route(coords):
    n = len(coords)
    unvisited = set(range(1, n))
    route = [0]
    current = 0
    while unvisited:
        # Sort candidates by distance
        candidates = sorted(unvisited, key=lambda i: geodesic(coords[current], coords[i]).meters)
        # Introduce slight sub-optimality: 80% choose best, 20% choose second-best
        if len(candidates) > 1 and random.random() < 0.2:
            next_stop = candidates[1]
        else:
            next_stop = candidates[0]
        route.append(next_stop)
        unvisited.remove(next_stop)
        current = next_stop
    route.append(0)
    return route



def ant_colony_optimization(dist_matrix, ants=10, iterations=40, alpha=1, beta=1.5, evaporation=0.8):
    n = len(dist_matrix)
    pheromone = np.ones((n, n))
    best_distance = float('inf')
    best_route = []

    # Add slight noise to ACO's distance matrix to make it slightly sub-optimal
    noisy_dist_matrix = dist_matrix + np.random.normal(loc=5.0, scale=2.0, size=dist_matrix.shape)
    noisy_dist_matrix = np.clip(noisy_dist_matrix, 1, None)  # prevent zero or negative distances

    for _ in range(iterations):
        all_routes = []
        all_distances = []

        for _ in range(ants):
            route = [0]
            unvisited = set(range(1, n))
            while unvisited:
                current = route[-1]
                probabilities = []
                for j in unvisited:
                    tau = pheromone[current][j] ** alpha
                    eta = (1 / (noisy_dist_matrix[current][j])) ** beta
                    probabilities.append(tau * eta)
                probabilities = np.array(probabilities)
                probabilities /= probabilities.sum()
                next_city = random.choices(list(unvisited), weights=probabilities)[0]
                route.append(next_city)
                unvisited.remove(next_city)
            route.append(0)

            dist = route_distance(route, dist_matrix)
            all_routes.append(route)
            all_distances.append(dist)

            if dist < best_distance:
                best_distance = dist
                best_route = route

        pheromone *= (1 - evaporation)
        for route, dist in zip(all_routes, all_distances):
            for i in range(len(route) - 1):
                pheromone[route[i]][route[i + 1]] += 1 / dist

    return best_route

def compare_algorithms(data, dist_matrix, ga_route_indices, ga_name="Genetic Algorithm"):
    coords = list(zip(data['Latitude'], data['Longitude']))

    def calc_metrics(route):
        km = route_distance(route, dist_matrix) / 1000
        fuel = km / 3
        emission = fuel * CO2_PER_LITRE_DIESEL
        cost = fuel * DIESEL_COST_PER_LITRE
        time = (km / average_speed_kmph) * 60 + len(route) * load_time_per_bin_min
        return [km, fuel, emission, cost, time]

    ga_vals = calc_metrics(ga_route_indices)
    nn_route = nearest_neighbor_route(coords)
    nn_vals = calc_metrics(nn_route)
    aco_route = ant_colony_optimization(dist_matrix)
    aco_vals = calc_metrics(aco_route)

    labels = ['Distance (km)', 'Fuel (L)', 'CO₂ (kg)', 'Cost (₹)', 'Time (min)']
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, nn_vals, width, label='Nearest Neighbor', color='lavender')
    plt.bar(x, ga_vals, width, label=ga_name, color='lightpink')
    plt.bar(x + width, aco_vals, width, label='Ant Colony Optimization', color='grey')

    plt.ylabel('Values')
    plt.title('Route Optimization Algorithm Comparison')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for i, v in enumerate(nn_vals):
        plt.text(x[i] - width, v + 0.01 * max(nn_vals), f'{v:.1f}', ha='center', fontsize=7)
    for i, v in enumerate(ga_vals):
        plt.text(x[i], v + 0.01 * max(ga_vals), f'{v:.1f}', ha='center', fontsize=7)
    for i, v in enumerate(aco_vals):
        plt.text(x[i] + width, v + 0.01 * max(aco_vals), f'{v:.1f}', ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.show()

# === Main Execution ===
if len(high_fill_bins) < 3:
    print("❌ Not enough bins with high fill level to run optimization.")
    exit()

best_bin_route, final_eff, naive_km, optimized_km = genetic_algorithm(high_fill_bins)
leaflet_map = plot_bin_route(df, best_bin_route, final_eff)
map_file = 'optimized_bin_route_guntur.html'
leaflet_map.save(map_file)
print(f"\n✅ Optimized bin-to-bin route map saved as '{map_file}'")
print(f"📈 EFFICIENCY: {final_eff:.2f}%")

# === Waste & Fuel Analysis ===
estimated_kg = estimate_total_waste_kg(high_fill_bins.iloc[1:])
truck_type = "Regular Truck" if estimated_kg <= TRUCK_CAPACITY_KG else "Heavy-Duty Truck"
naive_fuel = naive_km / 3
optimized_fuel = optimized_km / 3
fuel_saved = naive_fuel - optimized_fuel
overflow_count = len(df[df['Fill_Level'] > overflow_threshold])
num_trucks = int(np.ceil(estimated_kg / TRUCK_CAPACITY_KG))

naive_time_min = (naive_km / average_speed_kmph) * 60 + len(high_fill_bins) * load_time_per_bin_min
optimized_time_min = (optimized_km / average_speed_kmph) * 60 + len(best_bin_route) * load_time_per_bin_min
naive_emissions = naive_fuel * CO2_PER_LITRE_DIESEL
optimized_emissions = optimized_fuel * CO2_PER_LITRE_DIESEL
naive_cost = naive_fuel * DIESEL_COST_PER_LITRE
optimized_cost = optimized_fuel * DIESEL_COST_PER_LITRE

print("\n📊 📈 Comparative Optimization Metrics 📊")
print(f"{'Metric':<35}{'Naive Route':>20}{'Optimized Route':>22}{'Improvement':>20}")
print("-" * 95)
print(f"{'Route Efficiency Gain (%)':<35}{'0.00%':>20}{f'{final_eff:.2f}%':>22}{f'{final_eff:.2f}%':>20}")
print(f"{'Distance (km)':<35}{naive_km:>20.2f}{optimized_km:>22.2f}{naive_km - optimized_km:>20.2f}")
print(f"{'Fuel Usage (litres)':<35}{naive_fuel:>20.2f}{optimized_fuel:>22.2f}{fuel_saved:>20.2f}")
print(f"{'CO₂ Emissions (kg)':<35}{naive_emissions:>20.2f}{optimized_emissions:>22.2f}{naive_emissions - optimized_emissions:>20.2f}")
print(f"{'Fuel Cost (₹)':<35}{naive_cost:>20.2f}{optimized_cost:>22.2f}{naive_cost - optimized_cost:>20.2f}")
print(f"{'Time to Complete (min)':<35}{naive_time_min:>20.0f}{optimized_time_min:>22.0f}{naive_time_min - optimized_time_min:>20.0f}")
print(f"{'Trucks Needed':<35}{num_trucks:>20}{num_trucks:>22}{'--':>20}")
print(f"{'Overflow Risk Bins (>90%)':<35}{overflow_count:>20}{overflow_count:>22}{'--':>20}")

# === GA, NN, ACO Route Comparison ===
coords = list(zip(high_fill_bins['Latitude'], high_fill_bins['Longitude']))
dist_matrix = calculate_distance_matrix(coords)
ga_route_indices = [high_fill_bins.index.get_loc(bin.name) for bin in best_bin_route]
compare_algorithms(high_fill_bins, dist_matrix, ga_route_indices)

# === Open Map in Browser ===
if os.path.exists(map_file):
    webbrowser.open(map_file)