# cli.py
import argparse
import os
import yaml
import torch

from core import Fem1D, MaterialProperties, BoundaryConditions


def load_input_file(filepath: str) -> dict:
    """Load and parse YAML input file."""
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
    return data


def build_solver(config: dict) -> Fem1D:
    """Instantiate Fem1D solver from parsed YAML configuration."""
    torch.set_default_dtype(torch.float64)
    # Geometry and connectivity
    geo = config["geometry"]
    length = float(geo["length"])
    nodes = int(geo["nodes"])
    conn = torch.tensor(geo["connectivity"], dtype=torch.int64)
    conn = conn[:, [0, 2, 1]]  # Reorder to [left, mid, right]
    x = torch.linspace(0, length, nodes).view(-1, 1)

    # Material properties
    mat = config["material"]
    E = float(mat["E"])
    A = float(mat["area"])
    rho = float(mat["density"])
    g = float(mat["gravity"])
    nel = conn.shape[0]

    E_tensor = E * torch.ones(nel, 1)
    A_tensor = A * torch.ones(nel, 1)
    b = rho * g

    material = MaterialProperties(E=E_tensor, area=A_tensor, b=b)

    # Boundary conditions
    bc_data = config["boundary_conditions"]
    u_d = torch.tensor(bc_data["dirichlet_values"], dtype=torch.float64).view(-1, 1)
    drlt_dofs = torch.tensor(bc_data["dirichlet_nodes"], dtype=torch.int64)
    f_sur = torch.tensor(bc_data["surface_forces"], dtype=torch.float64).view(-1, 1)

    bc = BoundaryConditions(u_d=u_d, drlt_dofs=drlt_dofs, f_sur=f_sur)

    # Settings
    settings = config.get("settings", {})
    nqp = int(settings.get("nqp", 2))

    return Fem1D(x, conn, material=material, bc=bc, nqp=nqp)


def run_simulation(filepath: str, save_fig: bool, no_plot: bool) -> None:
    """Run the full FEM simulation pipeline."""
    config = load_input_file(filepath)
    solver = build_solver(config)

    solver.preprocess()
    solver.solve()
    solver.postprocess()
    solver.report()  # Print results to console

    if not no_plot:
        solver.plot()


def parse_cli() -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a 1D FEM simulation from YAML input")
    parser.add_argument("input_file", type=str, help="Path to the input YAML file")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")
    parser.add_argument("--save-fig", action="store_true", help="Save plot as PNG instead of showing it")
    return parser.parse_args()


def main():
    args = parse_cli()
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    run_simulation(args.input_file, save_fig=args.save_fig, no_plot=args.no_plot)