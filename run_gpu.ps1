# run_gpu.ps1
# Train PatchCore ensemble backbone on carpet + vial using GPU.
# Run from the workspace root: .\run_gpu.ps1
#
# Requirements:
#   - GPU with CUDA available
#   - .venv with torch (CUDA build) + faiss-gpu installed
#   - data/ folder created by setup_data.py

$env:PYTHONPATH = "./src"

.\.venv\Scripts\python.exe bin/run_patchcore.py `
    --gpu 0 `
    --seed 3 `
    --save_patchcore_model `
    --log_group "Carpet_Vial_Ensemble_WR101-ResNext101-Dense201_L2-3_P001_D1024-384_PS-3_AN-1_S3" `
    --log_project "AIL_Results" `
    results `
    patch_core `
        -b wideresnet101 `
        -b resnext101 `
        -b densenet201 `
        -le 0.layer2 -le 0.layer3 `
        -le 1.layer2 -le 1.layer3 `
        -le 2.features.denseblock2 -le 2.features.denseblock3 `
        --pretrain_embed_dimension 1024 `
        --target_embed_dimension 384 `
        --anomaly_scorer_num_nn 1 `
        --patchsize 3 `
    sampler -p 0.01 approx_greedy_coreset `
    dataset `
        --resize 256 `
        --imagesize 224 `
        --batch_size 2 `
        -d carpet `
        -d vial `
        mvtec data
