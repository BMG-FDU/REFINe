from __future__ import annotations
"""
Convert a graph-based scaffold description into an OBJ mesh representation.

The graph is interpreted as a set of centerline segments. Each segment is
thickened into a cylindrical fiber after global scaling, and the resulting
parts are concatenated into a single mesh for downstream CAD, visualization,
or finite-element (if you want) preprocessing workflows.
"""
import json
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import trimesh

@dataclass
# Graph2OBJ groups the complete graph-to-mesh conversion pipeline.
# The geometric scaling and fiber thickening parameters are kept explicit
# so that exported OBJ files remain traceable to physical dimensions.
class Graph2OBJ:
    json_path: Path
    out_path: Path = field(init=False)
    body_size_cm: float = 14.0        # cm
    fiber_diameter_cm: float = 0.4    # cm
    _verts: np.ndarray = field(init=False, repr=False)
    _edges: np.ndarray = field(init=False, repr=False)

    # Run the conversion pipeline in a fixed order: load graph data, normalize scale,
    # build the fiber mesh, and export the final OBJ file.
    def run(self) -> None:
        self._load_json()
        self._scale()
        mesh = self._build_mesh()
        mesh.export(self.out_path.as_posix())
    # Load node coordinates and graph connectivity from the JSON file.
    # The output path is derived from the input path to keep conversion results
    # paired with their source graph.
    def _load_json(self):
        data = json.loads(Path(self.json_path).read_text())
        nodes = sorted(data["nodes"], key=lambda n: n["id"])
        self._verts = np.array([[n["x"], n["y"], n["z"]] for n in nodes], dtype=np.float64)
        self._edges = np.array([[l["source"], l["target"]] for l in data["links"]], dtype=np.int64)
        self.out_path = self.json_path.with_suffix(".obj")
    # Normalize the graph so that its largest spatial span matches the prescribed
    # body size. This keeps different graph inputs comparable in physical units.
    def _scale(self):
        span = self._verts.max(axis=0) - self._verts.min(axis=0)
        scale = (self.body_size_cm) / span.max() if span.max() > 0 else 1.0
        self._verts *= scale
    # Build cylindrical mesh segments along graph edges.
    # Each edge is treated as a fiber centerline, while the radius is controlled by
    # the prescribed fiber diameter.
    def _build_mesh(self) -> trimesh.Trimesh:
        radius = self.fiber_diameter_cm / 2.0
        parts = []
        for i, j in self._edges:
            seg = (self._verts[i], self._verts[j])
            cyl = trimesh.creation.cylinder(
                radius=radius,
                segment=seg,
                sections=12,
            )
            parts.append(cyl)
        return trimesh.util.concatenate(parts)

# Command-line entry point for converting a selected graph JSON file.
# The hard-coded path is intended as a reproducible example and can be replaced
# by a project-specific input path or an argument parser if needed.
if __name__ == "__main__":
    src = Path(r"E:\3D_Outer_Contour\graph_json\SAVE\3_Lung_L1000.json")
    Graph2OBJ(json_path=src, body_size_cm=14.0, fiber_diameter_cm=0.4).run()
    print("✅")
