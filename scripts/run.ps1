param(
    [Parameter(Mandatory=$true)]
    [string]$Target
)

$ErrorActionPreference = "Stop"
$env:TRANSFORMERS_VERBOSITY = "error"

# ============================================================
# Global Defaults
# ============================================================
$gitHash = (git rev-parse HEAD)

$GLOBAL = @{
    MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDB   = "E:\data\certum_embeddings.db"
    REGIME    = "standard"
    FAR       = "0.02"
    CALFRAC   = "0.5"
    N         = "10000"
    SEED      = "1337"
}

# ============================================================
# Dataset Configurations
# ============================================================

$DATASETS = @{
    wiki = @{
        data_source = "feverous"
        kind     = "feverous"
        data     = "E:\data\feverous_dev_complete.jsonl"
        cache_db = "E:\data\certum_cache.db"
    }

    scifact = @{
        data_source = "scifact"
        kind     = "jsonl"
        data     = "E:\data\scifact_dev.jsonl"
        cache_db = "E:\data\certum_cache.db"
    }

    pubmed = @{
        data_source = "pubmed"
        kind     = "jsonl"
        data     = "E:\data\pubmedqa_train.jsonl"
        cache_db = "E:\data\certum_cache.db"
    }

    maud = @{
        data_source = "maud"
        kind     = "jsonl"
        data     = "E:\data\maud_dev.jsonl"
        cache_db = "E:\data\certum_cache.db"
    }

    hotpot = @{
        data_source = "hotpot"
        kind     = "jsonl"
        data     = "E:\data\hotpot_dev.jsonl"
        cache_db = "E:\data\certum_cache.db"
    }
}

# ============================================================
# Validate Target
# ============================================================

if (-not $DATASETS.ContainsKey($Target)) {
    Write-Host ""
    Write-Host "Invalid target: $Target"
    Write-Host "Available targets:"
    $DATASETS.Keys | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

$CFG = $DATASETS[$Target]

# ============================================================
# Build Run Folder
# ============================================================

$RUNID  = (Get-Date -Format "yyyyMMdd_HHmmss")
$OUTDIR = "artifacts\runs\$RUNID"
New-Item -ItemType Directory -Force -Path $OUTDIR | Out-Null

# ============================================================
# Adversarial Modes
# ============================================================

$NEG_MODES = @(
    "deranged",
    "hard_mined_v2"
)

# ============================================================
# Write Metadata File
# ============================================================

$MetaFile = "$OUTDIR\run_meta.json"

$meta = @{
    run_id = $RUNID
    timestamp = (Get-Date).ToUniversalTime().ToString("o")
    target = $Target
    git_commit = $gitHash

    dataset = @{
        kind = $CFG.kind
        path = $CFG.data
        n = [int]$GLOBAL.N
    }

    embedding = @{
        model = $GLOBAL.MODEL
        embedding_db = $GLOBAL.EMBEDDB
        cache_db = $CFG.cache_db
    }

    policy = @{
        regime = $GLOBAL.REGIME
        far = [double]$GLOBAL.FAR
        cal_frac = [double]$GLOBAL.CALFRAC
    }

    random = @{
        seed = [int]$GLOBAL.SEED
    }

    neg_modes_run = $NEG_MODES
}

$meta | ConvertTo-Json -Depth 6 | Out-File -Encoding UTF8 $MetaFile

# ============================================================
# Runner Function
# ============================================================

function Run-One($MODE) {

    $report = "$OUTDIR\negcal_$MODE.json"
    $pos    = "$OUTDIR\pos_$MODE.jsonl"
    $neg    = "$OUTDIR\neg_$MODE.jsonl"
    $plot   = "$OUTDIR\$MODE.png"

    py -m certum.orchestration.runner `
        --kind $CFG.kind `
        --in_path $CFG.data `
        --cache_db $CFG.cache_db `
        --embedding_db $GLOBAL.EMBEDDB `
        --model $GLOBAL.MODEL `
        --regime $GLOBAL.REGIME `
        --far $GLOBAL.FAR `
        --cal_frac $GLOBAL.CALFRAC `
        --n $GLOBAL.N `
        --seed $GLOBAL.SEED `
        --neg_mode $MODE `
        --out_report $report `
        --out_pos_scored $pos `
        --out_neg_scored $neg `
        --plot_png $plot
}

# ============================================================
# Execute Modes
# ============================================================

foreach ($mode in $NEG_MODES) {
    Run-One $mode
}

# ============================================================
# Validate Artifacts
# ============================================================

py -m certum.reporting.validate_gate_artifacts `
    --run $OUTDIR

Write-Host ""
Write-Host "DONE. Run folder: $OUTDIR"
