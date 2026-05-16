# Model Card for REFINe (REgular FIbrous Network Framework)

## Model Description

REFINe is a manufacturability-informed and physics-consistent AI framework for the design of fibrous network materials. It integrates three core components: (1) **TOPNet**, a topology-preserving network construction algorithm that formalizes Eulerian circuit continuity to ensure single-fiber fabrication compatibility, transforming digital topologies into continuous fiber laying– and 3D printing–compatible architectures; (2) a **physics-inspired graph neural network (GNN)** trained on automated finite element analysis (FEA) simulations to predict nonlinear mechanical responses (J-type and C-type load–displacement behaviors); and (3) a **reinforcement learning (RL) module** for inverse design optimization. The framework further supports projection of optimized 2D networks onto curved 3D surfaces via QuadriFlow-based surface mapping.

- **Developed by:** BMG-FDU
- **Model type:** Graph Neural Network (GNN) + Reinforcement Learning (RL)
- **Language(s):** Python
- **License:** MIT
- **Repository:** https://github.com/BMG-FDU/REFINe

---

## Intended Use

REFINe is intended for the computational design and optimization of fibrous network materials targeting specific mechanical properties. The framework is designed for researchers and engineers working in architected materials, soft robotics, textile engineering, and additive manufacturing. It supports both forward mechanical prediction and inverse design, with outputs directly compatible with continuous fiber laying and 3D printing (stereolithography and fused deposition modeling) fabrication pipelines.

---

## Quick Start

We recommend beginning with the provided `Quick_Start.ipynb`, which covers the principal components of the pipeline in a streamlined and accelerated fashion.

```bash
conda env create -f environment.yml
jupyter notebook Quick_Start.ipynb
```

The notebook runs end-to-end on a fresh installation in a few minutes and produces a representative inverse-design trajectory using the bundled simplified surrogate (`simplified config.json` + `model/`).

---

## Step-by-Step Reproduction

For users who wish to reproduce the full pipeline beyond the quick demo, the modules are organised in the following execution order:

**1. Dataset generation.** Generate the regular fibrous network dataset and extract external contours:
```bash
python P1_Gen_dataset_regular_net.py
python P2_External_contour.py
```

**2. Feature extraction.** Compute the 94-dimensional graph-level descriptors (output also provided as `features_94d.csv`):
```bash
python Features.py
python graph_feature_extractor.py
```

**3. FEA labelling** *(requires Abaqus 2024)*. Mesh the geometries and submit simulations under tension / compression / shear loading:
```bash
python geometry_mesh.py
python simulation_setup.py            # or simulation_setup_compress.py / simulation_setup_shear.py
python batch_submit.py                # or batch_submit_auto.py for automated dispatch
```

**4. Model training.** Train the surrogate predictor under cross-validation:
```bash
python "CV fold train.py"
```
Simplified weights are also provided as `model-1.rar` and `model-2.rar` (extract before use), allowing direct inference without retraining.

**5. Forward prediction.** Predict mechanical response from a topology:
```bash
python load_predictor.py
```

**6. Inverse design (RL).** Two-stage reinforcement-learning optimisation:
```bash
python RL-1.py        # stage 1
python RL-2.py        # stage 2
```
Typical runs achieve **~50% higher strength** and **~20% lower mass** relative to baseline designs, completing within minutes.

**7. Fabrication export.** Convert optimised topologies to printable / mappable formats:
```bash
python Graph2OBJ.py # also use `Quick_Start.ipynb`
python "geometry_mesh - 3D 0.5.py"    # optional: 3D meshing for surface mapping
```
For 2D image inputs, `2D-Img-STL.ipynb` provides an additional entry point for STL preparation. 3D surface mapping requires an external **QuadriFlow** configuration.

A detailed pseudocode description of the core algorithms is available as `Pseudocode.pdf`.

---

## Dependencies and Limitations

Certain components of the full pipeline depend on external software beyond the base conda environment:

- **FEM pipeline** (`geometry_mesh.py` → `simulation_setup.py` → `batch_submit.py`): Requires **Abaqus 2024**.
- **3D surface mapping** (QuadriFlow-based workflow): Requires **additional QuadriFlow configuration**.

---

## Additional Notes

The GNN is trained on a dataset of fibrous network topologies generated via `P1_Gen_dataset_regular_net.py`, with external contour processing handled by `P2_External_contour.py`. Structural and mechanical features are extracted through `Features.py` and `graph_feature_extractor.py`, encoded as 94-dimensional graph-level descriptors (provided as `features_94d.csv`). Ground-truth mechanical labels are obtained from an automated FEA pipeline supporting multiple loading conditions, including compression (`simulation_setup_compress.py`) and shear (`simulation_setup_shear.py`) in addition to the general setup (`simulation_setup.py`). Training follows a cross-validation procedure (`CV fold train.py`). Pretrained model weights are provided as `model-1.rar` and `model-2.rar`, corresponding to the two model components of the pipeline, allowing direct inference and inverse design without retraining from scratch. Forward mechanical response is handled by `load_predictor.py`. The RL-based inverse design is implemented across two stages (`RL-1.py`, `RL-2.py`), achieving approximately ~50% higher strength and ~20% lower mass relative to baseline designs, completing within minutes. The full pipeline is experimentally validated through stereolithography and fused deposition modeling. The framework supports 3D geometry meshing (`geometry_mesh - 3D 0.5.py`) and conversion of optimized graph topologies to OBJ format (`Graph2OBJ.py`) for downstream surface mapping and fabrication. A notebook for 2D image–to–STL conversion is also included (`2D-Img-STL.ipynb`), providing an additional entry point for geometry preparation. All major modules now ship with structured docstrings and inline comments to facilitate reading and reproduction; a detailed pseudocode description of the core algorithms is also available as `Pseudocode.pdf` in the repository.

---

## Citation
```
> BMG-FDU. *REFINe: REgular FIbrous Network Framework*. GitHub, 2026. https://github.com/BMG-FDU/REFINe
```
