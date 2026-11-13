| checkpoint | backend/readout | seed | latent (free) | params | reported acc | reproduced acc | ok |
|---|---|---|---|---|---|---|---|
| ignore/results_experiments/cbm_results_seed0/cub_dataset/audit.pt | native/coordinates | 0 | 1248 (624) | 1529000 |  | evaluation cache unavailable: cache/cub_eval | no |
| ignore/results_experiments/cbm_results_seed0/cubs_dataset/audit.pt | native/coordinates | 0 | 1248 (624) | 1526502 | 0.206897 | 0.206897 | yes |
| ignore/results_experiments/cbm_results_seed0/shapes3d_dataset/audit.pt | native/coordinates | 0 | 16 (8) | 17080 | 0.984427 | 0.984427 | yes |
| ignore/results_experiments/cbm_results_seed12/cub_dataset/audit.pt | native/coordinates | 12 | 1248 (624) | 1529000 | 0.871501 | 0.871501 | yes |
| ignore/results_experiments/cbm_results_seed12/cubs_dataset/audit.pt | native/coordinates | 12 | 1248 (624) | 1526502 | 0.241379 | 0.241379 | yes |
| ignore/results_experiments/cbm_results_seed12/mpi3d_dataset/audit.pt | native/coordinates | 12 | 20 (10) | 20836 | 0.999978 | 0.999978 | yes |
| ignore/results_experiments/cbm_results_seed12/shapes3d_dataset/audit.pt | native/coordinates | 12 | 16 (8) | 17080 | 0.975250 | 0.975250 | yes |
| ignore/results_experiments/cem_results_seed0/cub_dataset/audit.pt | cem/block_projection | 0 | 2496 (1872) | 6255800 | 0.882952 | 0.882952 | yes |
| ignore/results_experiments/cem_results_seed0/cubs_dataset/audit.pt | cem/block_projection | 0 | 2496 (1872) | 6250806 | 0.241379 | 0.241379 | yes |
| ignore/results_experiments/cem_results_seed0/shapes3d_dataset/audit.pt | cem/block_projection | 0 | 32 (24) | 75120 | 0.993333 | 0.993333 | yes |
| ignore/results_experiments/cem_results_seed12/cub_dataset/audit.pt | cem/block_projection | 12 | 2496 (1872) | 6255800 | 0.874470 | 0.874470 | yes |
| ignore/results_experiments/cem_results_seed12/cubs_dataset/audit.pt | cem/block_projection | 12 | 2496 (1872) | 6250806 | 0.206897 | 0.206897 | yes |
| ignore/results_experiments/cem_results_seed12/mpi3d_dataset/audit.pt | cem/block_projection | 12 | 40 (30) | 92906 | 0.999967 | 0.999967 | yes |
| ignore/results_experiments/cem_results_seed12/shapes3d_dataset/audit.pt | cem/block_projection | 12 | 32 (24) | 75120 | 0.984573 | 0.984573 | yes |

| cache | rows | embedding | concepts | aligned | unique ids |
|---|---|---|---|---|---|
| cache_shapes3d_eval | 96000 | [96000, 1024] float32 | [96000, 8] | True | True |
| cache_shapes3d_full | 480000 | [480000, 1024] float32 | [480000, 8] | True | True |
| cache_shapes3d_train | 384000 | [384000, 1024] float32 | [384000, 8] | True | True |
