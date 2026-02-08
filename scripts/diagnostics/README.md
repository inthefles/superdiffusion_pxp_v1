# Diagnostic Scripts

This folder contains diagnostic and debugging scripts used during development to investigate:

## MPO and Gate Issues
- `debug_indices.jl` - Investigate index structure of gates and MPOs
- `debug_tebd.jl` - Comprehensive TEBD diagnostics
- `verify_gate_indices.jl` - Verify gate application index structure
- `correct_gate_application.jl` - Figure out correct gate application

## Inner Product and Norm Issues
- `diagnose_inner_product.jl` - Investigate inner product computation for merged MPO
- `test_inner_carefully.jl` - Careful testing of inner() function
- `understand_inner.jl` - Understand what inner(MPO, MPO) computes

## Merging Issues
- `diagnose_merge_mpo.jl` - Detailed diagnostics for merge_mpo_pairs function
- `check_energy_location.jl` - Locate energy density in merged representation

## Projector Tests
- `test_projector_correlation.jl` - Test correlation with explicit projector
- `test_projector_trace.jl` - Test trace of global projector

## Comparison Variants
- `compare_tebd_ed_fixed.jl` - TEBD vs ED with matching operators
- `compare_tebd_ed_original.jl` - TEBD vs ED using original representation
- `test_gate_application.jl` - Test gate application logic directly

These scripts were used to identify and fix issues but are not needed for normal use of the package.
