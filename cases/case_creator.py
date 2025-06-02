import yaml


# --- Custom list class to force flow-style YAML ---
class FlowList(list):
    pass


# --- Custom YAML representer for FlowList ---
def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


# Register the representer globally
yaml.add_representer(FlowList, flow_list_representer)


# --- FEM input generation logic ---
def generate_quadratic_connectivity(nel):
    conn = []
    for e in range(nel):
        left = 2 * e + 1
        mid = 2 * e + 2
        right = 2 * e + 3
        conn.append([left, mid, right])
    return FlowList(conn)

# --- FEM input generation logic ---
def generate_linear_connectivity(nel):
    conn = []
    for e in range(nel):
        left = e + 1
        right = e + 2
        conn.append([left, right])
    return FlowList(conn)


def generate_quadratic_config(nel, nqp):
    nodes = 2 * nel + 1
    return {
        "material": {
            "E": 2.1e11,
            "area": 2.697e-4,
            "density": 9577.30812,
            "gravity": 9.81
        },
        "geometry": {
            "length": 70,
            "nodes": nodes,
            "connectivity": generate_quadratic_connectivity(nel)
        },
        "boundary_conditions": {
            "dirichlet_nodes": [1],
            "dirichlet_values": [0.0],
            "surface_forces": FlowList([0.0] * (nodes - 1) + [9123.3])
        },
        "settings": {
            "nqp": nqp,
            "scalingfactor": 100
        }
    }

def generate_linear_config(nel, nqp):
    nodes = nel + 1

    return {
        "material": {
            "E": 2.1e11,
            "area": 2.697e-4,
            "density": 9577.30812,
            "gravity": 9.81
        },
        "geometry": {
            "length": 70,
            "nodes": nodes,
            "connectivity": generate_linear_connectivity(nel)
        },
        "boundary_conditions": {
            "dirichlet_nodes": [1],
            "dirichlet_values": [0.0],
            "surface_forces": FlowList([0.0] * (nodes - 1) + [9123.3])
        },
        "settings": {
            "nqp": nqp,
            "scalingfactor": 100
        }
    }

# --- Write YAML for various element counts ---
if __name__ == "__main__":
    for nel in [10, 20, 50, 100, 1000, 10000]:
        for nqp in [2, 3]:
            config = generate_quadratic_config(nel, nqp)
            filename = f"cases/input_nel{nel}_nqp{nqp}_quadratic.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)

            # Elongation due to self weight of the cable only
            config["boundary_conditions"]["surface_forces"][-1] = 0
            filename = f"cases/input_nel{nel}_nqp{nqp}_self_weight_quadratic.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)
            
            # Elongation due to elevator cabin only
            config["boundary_conditions"]["surface_forces"][-1] = 300 * 9.81
            filename = f"cases/input_nel{nel}_nqp{nqp}_elevator_cabin_quadratic.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)

            # Elongation due to single person
            config["boundary_conditions"]["surface_forces"][-1] = (300 + 7) * 9.81
            filename = f"cases/input_nel{nel}_nqp{nqp}_single_person_quadratic.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)

            # Linear
            
            config = generate_linear_config(nel, nqp)
            filename = f"cases/input_nel{nel}_nqp{nqp}_linear.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)

            # Elongation due to self weight of the cable only
            config["boundary_conditions"]["surface_forces"][-1] = 0
            filename = f"cases/input_nel{nel}_nqp{nqp}_self_weight_linear.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)
            
            # Elongation due to elevator cabin only
            config["boundary_conditions"]["surface_forces"][-1] = 300 * 9.81
            filename = f"cases/input_nel{nel}_nqp{nqp}_elevator_cabin_linear.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)

            # Elongation due to single person
            config["boundary_conditions"]["surface_forces"][-1] = (300 + 7) * 9.81
            filename = f"cases/input_nel{nel}_nqp{nqp}_single_person_linear.yml"
            with open(filename, "w") as f:
                yaml.dump(config, f, sort_keys=False)
            
            print(f"✅ Wrote {filename}")

