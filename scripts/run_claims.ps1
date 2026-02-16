param(
    [Parameter(Mandatory=$true)]
    [string]$InputJsonl
)

$ErrorActionPreference = "Stop"

$RUNID  = (Get-Date -Format "yyyyMMdd_HHmmss")
$OUTDIR = "artifacts\runs\claims\$RUNID"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

py -m certum.evaluation.runner `
    --input_jsonl $InputJsonl `
    --out_dir $OUTDIR `
    --dataset_name "casehold_claims" `
    --embedding_model "sentence-transformers/all-MiniLM-L6-v2" `
    --embedding_db "E:\data\global_embeddings.db" `
    --nli_model "MoritzLaurer/deberta-v3-base-mnli-fever-anli" `
    --top_k 3 `
    --limit 1000 `
    --geometry_top_k 1000 `
    --rank_r 32 `
    --seed 1337
