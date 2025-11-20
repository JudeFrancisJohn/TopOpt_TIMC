Project: TopOpt (Julia)

Quick goal
- This workspace contains a small topology-optimization + stochastic material-parameter FE project written in Julia (Ferrite.jl + custom utilities). Primary scripts live in `src/` and helpers in `utils/`.

Big picture (what matters to an AI contributor)
- `src/COPY_stochastic_modified_v2_MC.jl` is the main driver for stochastic runs + topology optimization. It composes together input, FEA, KL-based random field generation, and OC-based design updates. Additionally, it includes an option for multiple runs.
- `utils/FE_updated_stoch.jl` contains the finite-element material routines, stress/tangent calculations for transverse isotropy, VTK export helpers, and a MaterialField type used by per-element material sampling.
- `utils/stochastic_utils.jl` provides KL expansion utilities: covariance builder, KL_realization (modes, eigen solvers), and helpers to build per-element material fields.
- `utils/opt.jl` implements the OC update (`OC`) and sensitivity filter (`check`) used by the topology loop.
- `utils/input.jl` holds default problem parameters (geometry, material means, optimization hyperparameters).

Key types & functions to know (names you will use frequently)
- MaterialParams (in `utils/FE_updated_stoch.jl`): holds scalar mean material properties (λ, μ_l, μ_t, alpha, beta, angle).
- MaterialField (mutable struct): holds per-element or per-node sampled fields (μ_l, μ_t, α, β, λ, angle).
- KL_realization(material_params, coords_elem; ...) (in `utils/stochastic_utils.jl`): generate a Dict of sampled fields per symbol (e.g. :μ_l, :μ_t, :α, :β) using KL expansion. Important kwargs: `use_centroids`, `make_sparse`, `Lc`, `N_modes`, `mode` (:additive or :lognormal).
- build_material_field(fields; use_centroids=false, eltype_out=Float32): converts KL output to MaterialField.
- build_KEStore!(dh, mf, nnodes_loc, avg_mp_store): computes per-element stiffness matrices (`KE_store`) using per-element averaged MaterialParams — used to accelerate FE solves inside topopt.
- FE_Run!(ℂ, x_fe) (in `src/COPY_stochastic_modified_v2 copy 2.jl`): run a staged load-stepping nonlinear solve (calls `NonlinearSolve` which uses `assemble_global!`).
- OC(x, volfrac, dc) and check(nelx,nely,rmin,x,dc) for design updates and filtering.

Project-specific conventions & gotchas
- SIMP is applied multiplicatively to stored element stiffness matrices (KE_store). The code precomputes KE_store using averaged material params and then scales by x^penal during assembly.
- Material sampling: the code supports either per-node/per-element fields or centroid-only values; `use_centroids=true` yields nelem×1 arrays while `use_centroids=false` yields nelem×nloc arrays. Many routines expect averaged MaterialParams for each element (see `build_KEStore!`).
- Kernel selection and eigen: KL_realization attempts dense eigen for small covariance matrices and Arpack for larger sparse cases. Expect fallback behavior if Arpack is not available or fails.
- FE solver uses Ferrite.jl-specific patterns: DofHandler, ConstraintHandler, start_assemble/assemble!, apply_zero!, create_sparsity_pattern. Be careful when modifying DOF ordering or boundary set names (e.g. "left_edge", "topmid_face").
- VTK exports: `WriteVTK` helpers are used in `exportresults` and custom `vtk_grid(...)` usage. File names are created under `./output/<script_name>/stochastic/run`.

Common workflows (how to run / debug locally)
- Run a quick FEA verification (no topopt) by launching `src/COPY_stochastic_modified_v2 copy 2.jl` in the Julia REPL or using `julia --project=. src/COPY_stochastic_modified_v2 copy 2.jl` from workspace root. The script prints an FEA verification message early on.
- Typical iterative TopOpt run: the main while-loop in `src/COPY_stochastic_modified_v2 copy 2.jl` performs FE_Run!, computes compliance/sensitivities, applies filter (`check`), updates with `OC`, and writes VTK files. Look at `save_path` near the top of that file for output location.
- If eigen/Arpack issues arise during KL sampling, reduce `N_modes`, set `make_sparse=true`, or force dense eigen by setting `make_sparse=false` and ensuring memory fits.

### Enhancements for AI Coding Agents

#### Debugging Workflows
- **VTK File Inspection**: Use ParaView to open `.vtu` files in the `output/` directory. Check displacement and stress fields for anomalies.
- **FEA Verification**: Run `src/COPY_stochastic_modified_v2 copy 2.jl` and confirm the "FEA verification" message appears early in the output.
- **KL Sampling Issues**: If eigen/Arpack errors occur, adjust `N_modes` or `make_sparse` in `KL_realization` calls.

#### Integration Points and External Dependencies
- **Ferrite.jl**: Central to the FE solver. Key patterns include `DofHandler`, `ConstraintHandler`, and `assemble!`.
- **VTK Export**: Relies on `WriteVTK` for output. File paths are dynamically generated under `./output/<script_name>/stochastic/run`.
- **Arpack.jl**: Used for sparse eigenvalue problems in KL sampling. Ensure it is installed and functional.

#### Additional Project-Specific Conventions
- **MaterialField Sampling**: Always verify whether `use_centroids` is set appropriately for the task. This affects array dimensions and downstream computations.
- **Boundary Conditions**: Naming conventions like `"left_edge"` and `"topmid_face"` are hardcoded. Maintain consistency when adding new conditions.

#### Example Updates
- **Adding a New Material Property**: Update `MaterialParams` in `utils/FE_updated_stoch.jl` and ensure `KL_realization` and `build_material_field` handle the new property.
- **Customizing Optimization**: Modify `OC` or `check` in `utils/opt.jl` to implement alternative update schemes.

#### Testing and Validation
- **Small-Scale Runs**: Reduce `nelx` and `nely` in `utils/input.jl` for faster iterations during debugging.
- **Environment Consistency**: Use Julia 1.11.5 as specified in `Project.toml`. Ensure dependencies match `Manifest.toml`.

#### Open Questions
- Should additional test cases be added for edge scenarios in KL sampling or OC updates?
- Are there preferred visualization tools/settings for `.vtu` files beyond ParaView?

#### Input Parameter Files
- **Geometrical Parameters**: Defined in `input/params_geom.jl`. Modify this file to adjust the geometry of the problem.
- **Material Parameters**: Defined in `input/params_mat.jl`. This file contains parameters for transversely isotropic materials, such as `λ`, `μ_l`, `μ_t`, `alpha`, `beta`, and `angle`.
- **Topology Optimization Parameters**: Defined in `input/params_topopt.jl`. Use this file to configure optimization-specific settings.

When changing input parameters, ensure consistency across these files to avoid mismatches during execution.

#### Main Driver Script
- Always confirm with the user which script is the main driver. While `src/COPY_stochastic_modified_v2 copy 2.jl` is currently the main driver for stochastic runs and topology optimization, there may be other versions or copies actively being used. Ensure any changes or debugging are applied to the correct version of the driver script.



