<#
  Project ARGUS - one wrapper around the docker commands in README.md. Every action is a plain
  `docker compose ...` call (printed before it runs), so nothing here is required: copy the command instead.

    .\argus.ps1 build                  build all images (training first: scoring and mlflow build on it)
    .\argus.ps1 up                     UI :3000, platform API :8090, MLflow :5000
    .\argus.ps1 serve [run]            + Fraud Model API :8080 and vLLM explainer :8000 for a finished run (saved in .env)
    .\argus.ps1 unserve                stop scoring + explainer (frees the GPU for training)
    .\argus.ps1 apply <service>...     recreate services after a settings change (.env), like the UI's Settings page
    .\argus.ps1 agent [start|stop|status]  the operations agent: lets the UI's Operations / Settings pages act on this PC
    .\argus.ps1 train <profile> [run]  a pipeline run from the CLI (smoke, local_demo, demo_hybrid, ...)
    .\argus.ps1 report                 report.md per demo run + outputs/comparison.md (hybrid vs LLM scorer)
    .\argus.ps1 demo                   CRD / LLD scenarios against the running scoring API -> outputs/<run>/api/
    .\argus.ps1 loadtest [rps] [sec]   end-to-end API latency at a fixed rate (OBJ-10) -> outputs/<run>/api/
    .\argus.ps1 stream [stop]          Kinesis emulator + stream scorer :8003 (UI: Live stream); needs serve
    .\argus.ps1 prefetch <profile>     download every model a profile uses (checks gated access first)
    .\argus.ps1 aws <command>          AWS deployment: check | up | deploy | status | dataset | down | audit
    .\argus.ps1 logs [service]         follow logs (platform, frontend, mlflow, scoring, explainer, stream, or a job id)
    .\argus.ps1 status                 containers, images and the latest platform jobs
    .\argus.ps1 test                   platform, training, scoring, stream pytest + frontend vitest, all in containers
    .\argus.ps1 down                   stop everything (outputs, data and models stay on disk)
  GPU VM (Hyperstack) and AWS deployment: deploy/README.md.
#>
param(
  [Parameter(Position = 0)] [string] $Action = "help",
  [Parameter(Position = 1)] [string] $Arg1,
  [Parameter(Position = 2)] [string] $Arg2,
  [Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest   # e.g. aws down --all -y
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run([string] $cmd) {
  Write-Host "> $cmd" -ForegroundColor Cyan
  Invoke-Expression $cmd
  # Docker Desktop on Windows can hold a recreated container's old host port for a few seconds ("ports are not
  # available"): starting containers again is safe (compose only starts what is not running), so retry that briefly.
  for ($try = 1; $LASTEXITCODE -ne 0 -and $cmd -match "docker compose .*\bup\b" -and $try -le 3; $try++) {
    Write-Host "  start failed (often a host port Docker has not released yet): retry $try of 3 in 5 s" -ForegroundColor Yellow
    Start-Sleep -Seconds 5
    Invoke-Expression $cmd
  }
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $cmd" }
}

# Set KEY=VALUE in .env (replace the last line for KEY, or append). UTF-8 without BOM: Windows PowerShell's default
# encodings (UTF-16 / BOM) break docker compose's .env parsing.
function Set-EnvValue([string] $name, [string] $value) {
  $path = Join-Path $PSScriptRoot ".env"
  $lines = if (Test-Path $path) { @([IO.File]::ReadAllLines($path)) } else { @() }
  $idx = -1
  for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match "^\s*$name\s*=") { $idx = $i } }
  if ($idx -ge 0) { $lines[$idx] = "$name=$value" } else { $lines += "$name=$value" }
  [IO.File]::WriteAllText($path, (($lines -join "`n") + "`n"), (New-Object Text.UTF8Encoding($false)))
}

# A setting from the environment, else from .env, else the default (the same order docker compose uses).
function Setting([string] $name, [string] $default) {
  $value = [Environment]::GetEnvironmentVariable($name)
  if (-not $value -and (Test-Path .env)) {
    $line = Get-Content .env | Where-Object { $_ -match "^\s*$name\s*=" } | Select-Object -Last 1
    if ($line) { $value = ($line -split "=", 2)[1].Trim().Trim('"').Trim("'") }
  }
  if ($value) { $value } else { $default }
}
$UiPort = Setting "ARGUS_UI_PORT" "3000"; $ApiPort = Setting "ARGUS_API_PORT" "8090"
$MlflowPort = Setting "ARGUS_MLFLOW_PORT" "5000"; $ScoringPort = Setting "ARGUS_SCORING_PORT" "8080"
$StreamPort = Setting "ARGUS_STREAM_PORT" "8003"

# No NVIDIA GPU (a Mac, another GPU vendor): the platform would not start with its GPU reservation, so add the overlay
# that drops it. Training and serving still need an NVIDIA GPU.
if (-not (Setting "COMPOSE_FILE" "") -and -not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
  $env:COMPOSE_FILE = "docker-compose.yml;docker-compose.nogpu.yml"; $env:COMPOSE_PATH_SEPARATOR = ";"
  Write-Host "No NVIDIA GPU found: UI, platform and MLflow run without GPU telemetry (docker-compose.nogpu.yml)." -ForegroundColor Yellow
}

switch ($Action) {
  "build" {
    Run "docker compose build trainer"
    Run "docker compose --profile serve build scoring"
    Run "docker compose build platform frontend"
    Run "docker compose --profile stream build stream"
  }
  "up" {
    Run "docker compose up -d"
    Write-Host "UI http://localhost:$UiPort   API http://localhost:$ApiPort/docs   MLflow http://localhost:$MlflowPort" -ForegroundColor Green
  }
  "serve" {
    $run = if ($Arg1) { $Arg1 } else { Setting "RUN_NAME" "argus_demo_llm_scorer" }
    if ($run -notmatch "^[a-z0-9][a-z0-9_-]{1,62}$" -or -not (Test-Path "training/outputs/$run")) { throw "no trained run '$run' in training/outputs" }
    # saved in .env (like serve.sh on the VM): the UI shows it, and containers restarted by Docker keep serving it
    Set-EnvValue "RUN_NAME" $run; Set-EnvValue "EXPLAINER_RUN" $run
    $env:RUN_NAME = $run; $env:EXPLAINER_RUN = $run
    if (Test-Path "training/outputs/$run/explainer_student/awq") {
      # the scorer first and on its own: payments need it; the explainer is optional (templates explain without it),
      # so a vLLM image that cannot be pulled or a model that does not fit never blocks scoring
      Run "docker compose --profile serve up -d scoring"
      Write-Host "> docker compose --profile serve up -d explainer" -ForegroundColor Cyan
      docker compose --profile serve up -d explainer
      if ($LASTEXITCODE -ne 0) {
        Write-Host "the vLLM explainer did not start (image download / GPU memory?): explanations come from templates; run serve $run again to retry" -ForegroundColor Yellow
        $global:LASTEXITCODE = 0
      }
    } else {
      # vLLM cannot start without the model (it would restart forever holding GPU memory): templates explain instead
      Write-Host "run '$run' has no AWQ explainer: serving the scorer only, explanations come from templates" -ForegroundColor Yellow
      Invoke-Expression "docker compose --profile serve stop explainer" 2>$null
      Run "docker compose --profile serve up -d scoring"
    }
    Write-Host "Fraud Model API http://localhost:$ScoringPort/docs (serving $run)" -ForegroundColor Green
  }
  "apply" {
    # same rules as deploy/hyperstack/apply.sh: core services always, serving / stream services only if running now
    $svcs = @(@($Arg1, $Arg2) + $Rest | Where-Object { $_ })
    if (-not $svcs) { throw "usage: .\argus.ps1 apply <service>...  (frontend platform mlflow scoring explainer stream kinesis)" }
    $running = @(docker compose --profile serve --profile stream ps --services --status running)
    $todo = @()
    foreach ($s in $svcs) {
      if ($s -in "frontend", "platform", "mlflow") { $todo += $s }
      elseif ($s -in "scoring", "explainer", "stream", "kinesis") {
        if ($running -contains $s) { $todo += $s } else { Write-Host "$s is not running: it reads the new settings when it next starts" -ForegroundColor Yellow }
      } else { throw "unknown service '$s'" }
    }
    $others = @($todo | Where-Object { $_ -ne "platform" })
    if ($others) { Run ("docker compose --profile serve --profile stream up -d --no-deps " + ($others -join " ")) }
    if ($todo -contains "platform") { Run "docker compose up -d --no-deps platform" }   # last: the UI talks to it
    if (-not $todo) { Write-Host "nothing running needs a restart" -ForegroundColor Green }
  }
  "agent" {
    # the operations agent (deploy/agent.py): runs what the UI's Operations / Settings pages ask for on this PC
    switch ($Arg1) {
      "start" {
        Start-Process python -ArgumentList "deploy/agent.py" -WorkingDirectory $PSScriptRoot -WindowStyle Minimized
        Write-Host "agent started in a minimized window (stop: .\argus.ps1 agent stop)" -ForegroundColor Green
      }
      "stop" {
        $beat = "training/outputs/_platform/ops/agent.json"
        if (Test-Path $beat) { $agentPid = (Get-Content $beat -Raw | ConvertFrom-Json).pid; Stop-Process -Id $agentPid -Force -ErrorAction SilentlyContinue; Write-Host "agent stopped (pid $agentPid)" }
        else { Write-Host "no agent heartbeat found" }
      }
      "status" { python deploy/agent.py --status }
      default { Run "python deploy/agent.py" }   # foreground: Ctrl+C stops it
    }
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
    # 127.0.0.1, not localhost: Windows tries IPv6 ::1 first and stalls about 2 s on Docker Desktop's forwarding
    $run = (Invoke-RestMethod "http://127.0.0.1:$ScoringPort/v1/health").active_run
    # the demo client runs inside the scoring container: it uses the internal port 8080 whatever ARGUS_SCORING_PORT is
    Run "docker compose --profile serve exec scoring python -m argus_api.demo --run $run --url http://localhost:8080"
  }
  "loadtest" {
    $rps = if ($Arg1) { $Arg1 } else { "5" }; $sec = if ($Arg2) { $Arg2 } else { "60" }
    Run "docker compose --profile serve exec scoring python -m argus_api.loadtest --url http://localhost:8080 --rps $rps --duration $sec"
  }
  "stream" {
    if ($Arg1 -eq "stop") { Run "docker compose --profile stream stop stream kinesis" }
    else {
      Run "docker compose --profile stream up -d kinesis stream"
      Write-Host "Live stream: UI http://localhost:$UiPort/stream   API http://localhost:$StreamPort/live/docs" -ForegroundColor Green
    }
  }
  "prefetch" {
    if (-not $Arg1) { throw "usage: .\argus.ps1 prefetch <profile>" }
    Run "docker compose run --rm --no-deps trainer python -m argus_ft.prefetch --config configs/$Arg1.yaml"
  }
  "aws" {
    if (-not $Arg1) { throw "usage: .\argus.ps1 aws check|up|deploy|status|dataset|down|audit" }
    Run ("python deploy/ops.py " + ((@($Arg1, $Arg2) + $Rest | Where-Object { $_ }) -join " "))
  }
  "logs" {
    if (-not $Arg1) { Run "docker compose logs -f --tail 200" }
    elseif ($Arg1 -match "^[0-9a-f]{10}$") { Run "docker logs -f argus-job-$Arg1" }
    else { Run "docker compose --profile serve --profile stream logs -f --tail 200 $Arg1" }
  }
  "status" {
    Run "docker compose --profile serve --profile stream ps -a"
    # training jobs: read the labels from JSON (a quoted Go template does not survive Windows argument quoting)
    Write-Host "> docker ps -a --filter label=argus.job" -ForegroundColor Cyan
    docker ps -a --filter label=argus.job --format "{{json .}}" | ForEach-Object { $_ | ConvertFrom-Json } |
      Select-Object Names, Status, @{ n = "Run"; e = { ($_.Labels -split "," | Where-Object { $_ -like "argus.run=*" }) -replace "^argus.run=", "" } } |
      Format-Table -AutoSize
    Run "docker images --filter reference='argus-*' --format 'table {{.Repository}}\t{{.Size}}'"
  }
  "test" {
    $index = if ($env:PIP_INDEX_URL) { $env:PIP_INDEX_URL } else { "https://pypi.org/simple" }
    Run "docker build -q --build-arg WITH_TESTS=true --build-arg PIP_INDEX_URL=$index -t argus-platform:test ./platform"
    Run "docker run --rm -v `"${PWD}/deploy:/deploy:ro`" argus-platform:test python -m pytest -q"   # /deploy: the operations catalogue
    Run "docker compose run --rm --no-deps -T -e CUDA_VISIBLE_DEVICES= -v `"${PWD}/training/tests:/workspace/tests`" -v `"${PWD}/training/pytest.ini:/workspace/pytest.ini`" trainer sh -c `"pip install -q pytest==9.2.0 && python -m pytest -q -p no:cacheprovider tests`""
    Run "docker run --rm -v `"${PWD}/scoring_api/tests:/workspace/tests`" -v `"${PWD}/training/argus_ft:/workspace/argus_ft`" -e CUDA_VISIBLE_DEVICES= argus-scoring:latest sh -c `"pip install -q pytest==9.2.0 && python -m pytest -q -p no:cacheprovider tests`""
    Run "docker build -q --build-arg WITH_TESTS=true --build-arg PIP_INDEX_URL=$index -t argus-stream:test ./stream"
    Run "docker run --rm argus-stream:test python -m pytest -q"
    Run "docker run --rm -v `"${PWD}/deploy:/deploy`" -w /deploy python:3.12.16-slim-bookworm sh -c `"pip install -q pytest==9.2.0 && python -m pytest -q -p no:cacheprovider tests`""
    Run "docker build --target test ./frontend"
  }
  "down" { Run "docker compose --profile serve --profile stream --profile job down" }
  default { Get-Content $PSCommandPath -TotalCount 20 | Select-Object -Skip 4 }
}
