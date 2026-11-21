<#
  Project ARGUS - one wrapper around the docker commands in README.md. Every action is a plain
  `docker compose ...` call (printed before it runs), so nothing here is required: copy the command instead.

    .\argus.ps1 build                  build all images (training first: scoring and mlflow build on it)
    .\argus.ps1 up                     UI :3000, platform API :8090, MLflow :5000
    .\argus.ps1 serve [run]            + Fraud Model API :8080 and vLLM explainer :8000 for a finished run
    .\argus.ps1 unserve                stop scoring + explainer (frees the GPU for training)
    .\argus.ps1 train <profile> [run]  a pipeline run from the CLI (smoke, local_demo, demo_hybrid, ...)
    .\argus.ps1 report                 report.md per demo run + outputs/comparison.md (hybrid vs LLM scorer)
    .\argus.ps1 demo                   CRD / LLD scenarios against the running scoring API -> outputs/<run>/api/
    .\argus.ps1 logs [service]         follow logs (platform, frontend, mlflow, scoring, explainer, or a job id)
    .\argus.ps1 status                 containers, images and the latest platform jobs
    .\argus.ps1 test                   platform pytest + scoring pytest + frontend vitest, all in containers
    .\argus.ps1 down                   stop everything (outputs, data and models stay on disk)
#>
param(
  [Parameter(Position = 0)] [string] $Action = "help",
  [Parameter(Position = 1)] [string] $Arg1,
  [Parameter(Position = 2)] [string] $Arg2
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run([string] $cmd) {
  Write-Host "> $cmd" -ForegroundColor Cyan
  Invoke-Expression $cmd
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $cmd" }
}

switch ($Action) {
  "build" {
    Run "docker compose build trainer"
    Run "docker compose --profile serve build scoring"
    Run "docker compose build platform frontend"
  }
  "up" {
    Run "docker compose up -d"
    Write-Host "UI http://localhost:3000   API http://localhost:8090/docs   MLflow http://localhost:5000" -ForegroundColor Green
  }
  "serve" {
    $run = if ($Arg1) { $Arg1 } else { "argus_demo_llm_scorer" }
    $env:RUN_NAME = $run; $env:EXPLAINER_RUN = $run
    Run "docker compose --profile serve up -d scoring explainer"
    Write-Host "Fraud Model API http://localhost:8080/docs (serving $run)" -ForegroundColor Green
  }
  "unserve" { Run "docker compose --profile serve stop scoring explainer" }
  "train" {
    if (-not $Arg1) { throw "usage: .\argus.ps1 train <profile> [run_name]" }
    $set = if ($Arg2) { " --set run_name=$Arg2" } else { "" }
    Run "docker compose run --rm trainer python -m argus_ft.pipeline --config configs/$Arg1.yaml --no-sync$set"
  }
  "report" {
    Run "docker compose run --rm --no-deps trainer python -m argus_ft.compare argus_demo_hybrid argus_demo_llm_scorer"
  }
  "demo" {
    $run = (Invoke-RestMethod http://localhost:8080/v1/health).active_run
    Run "docker compose --profile serve exec scoring python -m argus_api.demo --run $run --url http://localhost:8080"
  }
  "logs" {
    if (-not $Arg1) { Run "docker compose logs -f --tail 200" }
    elseif ($Arg1 -match "^[0-9a-f]{10}$") { Run "docker logs -f argus-job-$Arg1" }
    else { Run "docker compose --profile serve logs -f --tail 200 $Arg1" }
  }
  "status" {
    Run "docker compose --profile serve ps -a"
    Run "docker ps -a --filter label=argus.job --format 'table {{.Names}}\t{{.Status}}\t{{.Label `"argus.run`"}}'"
    Run "docker images --filter reference='argus-*' --format 'table {{.Repository}}\t{{.Size}}'"
  }
  "test" {
    $index = if ($env:PIP_INDEX_URL) { $env:PIP_INDEX_URL } else { "https://pypi.org/simple" }
    Run "docker build -q --build-arg WITH_TESTS=true --build-arg PIP_INDEX_URL=$index -t argus-platform:test ./platform"
    Run "docker run --rm argus-platform:test python -m pytest -q"
    Run "docker run --rm -v `"${PWD}/scoring_api/tests:/workspace/tests`" -e CUDA_VISIBLE_DEVICES= argus-scoring:latest sh -c `"pip install -q pytest==9.2.0 && python -m pytest -q -p no:cacheprovider tests`""
    Run "docker build --target test ./frontend"
  }
  "down" { Run "docker compose --profile serve --profile job down" }
  default { Get-Content $PSCommandPath -TotalCount 15 | Select-Object -Skip 4 }
}
