# ==============================================================================
# 대한민국 세법 RAG 환경 구성 스크립트 [Windows PowerShell]
#
# 컨테이너 3종[PostgreSQL+pgvector, Neo4j, API]을 빈 DB 상태로 기동하고 /health 를 기다린다.
# 데이터[코퍼스]는 포함되어 있지 않으며, 기동 후 db_build/ 스크립트로 직접 구축한다.
#
# 사용법:
#   .\deploy.ps1              [기동 + /health 대기]
#   .\deploy.ps1 -DryRun      [도커 기동 없이 배포 로직만 시뮬레이션]
#   .\deploy.ps1 -Help        [도움말 안내 출력]
#
# 환경변수[선택]: PG_PORT NEO4J_HTTP_PORT NEO4J_BOLT_PORT API_PORT API_BIND  [호스트 포트·바인딩]
#                 HEALTH_RETRIES  /health 대기 횟수 [5초 간격, 기본 60]
# 필수 설정: POSTGRES_PASSWORD NEO4J_PASSWORD[8자 이상]  [.env 또는 환경변수, 비어 있으면 안내 후 종료]
# ==============================================================================
[CmdletBinding()]
param (
    [switch]$DryRun,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'

if ($Help) {
    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host ' 대한민국 세법 RAG 환경 구성 스크립트 안내 [Windows]' -ForegroundColor Cyan
    Write-Host '============================================================' -ForegroundColor Cyan
    Write-Host ''
    Write-Host '사용법:'
    Write-Host '  .\deploy.ps1 [옵션]'
    Write-Host ''
    Write-Host '옵션:'
    Write-Host '  -DryRun      도커 기동 없이 전체 배포 로직 가상 시뮬레이션'
    Write-Host '  -Help        본 도움말 안내 출력'
    Write-Host ''
    Write-Host '환경변수[선택]:'
    Write-Host '  PG_PORT / NEO4J_HTTP_PORT / NEO4J_BOLT_PORT / API_PORT   호스트 포트 [기본 5432/7474/7687/8000]'
    Write-Host '  API_BIND        API 바인딩 주소 [기본 127.0.0.1. 클라우드 등 외부 접속이 필요하면 0.0.0.0,'
    Write-Host '                  DB 포트는 항상 127.0.0.1]'
    Write-Host ''
    Write-Host '안내:'
    Write-Host '  - 이 스크립트는 빈 DB로 기동합니다. 코퍼스는 포함되어 있지 않습니다.'
    Write-Host '    기동 후 db_build/README.md 를 보고 직접 구축하세요.'
    Write-Host '  - 빈 DB에서는 /health 가 "검색 테이블 없음" 경고를 내는 것이 정상입니다.'
    Write-Host '  - 질문 기능을 쓰려면 .env 에 OpenRouter API Key 를 넣어야 합니다.'
    Write-Host '  - DB 비밀번호는 저장소에 없습니다. 처음 실행하면 .env 가 만들어지고, 그 안의'
    Write-Host '    POSTGRES_PASSWORD, NEO4J_PASSWORD[8자 이상]를 직접 정해 넣어야 기동합니다.'
    Write-Host '    비어 있으면 안내를 출력하고 멈춥니다[-DryRun 도 같습니다].'
    Write-Host '    환경변수로 넘겨도 됩니다: $env:POSTGRES_PASSWORD="..."; $env:NEO4J_PASSWORD="..."; .\deploy.ps1'
    Write-Host '============================================================' -ForegroundColor Cyan
    exit 0
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# 설정값 조회: 셸 환경변수 > .env > 기본값 [docker compose 의 치환 규칙과 동일]
function Get-Cfg([string]$Key, [string]$Default) {
    $v = [Environment]::GetEnvironmentVariable($Key)
    if (-not $v -and (Test-Path '.env')) {
        $line = Get-Content '.env' | Where-Object { $_ -match ('^' + $Key + '=') } | Select-Object -Last 1
        if ($line) { $v = ($line -replace ('^' + $Key + '='), '').Trim().Trim('"').Trim("'") }
    }
    if ($v) { return $v } else { return $Default }
}

function Test-PortInUse([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect('127.0.0.1', $Port, $null, $null)
        if ($async.AsyncWaitHandle.WaitOne(500) -and $client.Connected) { return $true }
        return $false
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

Write-Host '============================================================' -ForegroundColor Cyan
Write-Host ' 🚀 대한민국 세법 RAG 환경 구성을 시작합니다 [빈 DB 기동, Windows]' -ForegroundColor Cyan
if ($DryRun) {
    Write-Host ' 🧪 [DRY-RUN 모드 활성화]: 도커 기동 없이 배포 로직만 시뮬레이션합니다.' -ForegroundColor Magenta
}
Write-Host '============================================================' -ForegroundColor Cyan

# 1. Docker 데몬 실행 점검
if (-not $DryRun) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host '❌ Docker가 설치되어 있지 않습니다. Docker Desktop을 설치해 주세요.' -ForegroundColor Red
        exit 1
    }
    cmd /c "docker info >nul 2>&1"
    if ($LASTEXITCODE -ne 0) {
        Write-Host '❌ Docker 데몬이 실행 중이지 않습니다. Docker Desktop을 실행해 주세요.' -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host '🔍 [DRY-RUN] Docker 데몬 연결 상태: 가상 통과 [OK]' -ForegroundColor Green
}

# 대화형 터미널 여부 [파이프·CI 에서는 질의 없이 진행]
$interactive = [Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)

# 2. .env 확인 및 생성
if (-not (Test-Path '.env')) {
    Write-Host '📄 .env 파일이 없습니다. .env.example을 기반으로 생성합니다...' -ForegroundColor Yellow
    if (Test-Path '.env.example') {
        Copy-Item '.env.example' '.env'
    } else {
        New-Item -ItemType File '.env' | Out-Null
    }

    if ((-not $DryRun) -and $interactive) {
        Write-Host ''
        $userKey = Read-Host '👉 OpenRouter API Key를 입력하세요 [건너뛰려면 Enter]'
        if ($userKey) {
            $envContent = Get-Content '.env' -Raw
            if ($envContent -match 'OPENROUTER_API_KEY=') {
                $envContent = $envContent -replace 'OPENROUTER_API_KEY=.*', "OPENROUTER_API_KEY=""$userKey"""
            } else {
                $envContent += "`nOPENROUTER_API_KEY=""$userKey"""
            }
            Set-Content -Path '.env' -Value $envContent -Encoding UTF8
            Write-Host '✅ API Key가 .env에 설정되었습니다.' -ForegroundColor Green
        } else {
            Write-Host '⚠️ API Key 입력을 건너뛰었습니다. 나중에 .env 파일에 직접 입력해 주세요.' -ForegroundColor Yellow
        }
    } elseif (-not $DryRun) {
        Write-Host '⚠️ 대화형 터미널이 아니어서 API Key 입력을 건너뜁니다. 나중에 .env 파일에 직접 입력해 주세요.' -ForegroundColor Yellow
    }
}

# 2-1. DB 비밀번호 확인 [compose 가 필수로 요구한다. 영문 오류 대신 여기서 먼저 안내하고 멈춘다. -DryRun 도 같다]
$pgPw = Get-Cfg 'POSTGRES_PASSWORD' ''
$neoPw = Get-Cfg 'NEO4J_PASSWORD' ''
$pwProblem = $false
if (-not $pgPw) {
    Write-Host '❌ POSTGRES_PASSWORD 가 비어 있습니다.' -ForegroundColor Red
    $pwProblem = $true
}
if (-not $neoPw) {
    Write-Host '❌ NEO4J_PASSWORD 가 비어 있습니다.' -ForegroundColor Red
    $pwProblem = $true
} elseif ($neoPw.Length -lt 8) {
    Write-Host ('❌ NEO4J_PASSWORD 는 8자 이상이어야 합니다 [현재 ' + $neoPw.Length + '자]. Neo4j 가 더 짧은 비밀번호를 거부합니다.') -ForegroundColor Red
    $pwProblem = $true
}
if ($pwProblem) {
    Write-Host ''
    Write-Host '👉 DB 비밀번호는 저장소에 들어 있지 않습니다. 직접 정해서 적어 주세요.' -ForegroundColor Yellow
    Write-Host ('   파일: ' + (Join-Path $scriptDir '.env') + '  [.env 가 없었다면 .env.example 을 복사해 만든 파일입니다]')
    Write-Host '     POSTGRES_PASSWORD=<영문·숫자 비밀번호>'
    Write-Host '     NEO4J_PASSWORD=<8자 이상 비밀번호>'
    Write-Host '   환경변수로 넘겨도 됩니다: $env:POSTGRES_PASSWORD="..."; $env:NEO4J_PASSWORD="..."; .\deploy.ps1'
    Write-Host '   적은 뒤 .\deploy.ps1 을 다시 실행하세요.'
    exit 1
}

# 3. 필수 디렉토리 생성
if (-not (Test-Path 'logs')) {
    New-Item -ItemType Directory 'logs' | Out-Null
}

# 4. 호스트 포트 충돌 점검 [이미 이 스택이 떠 있으면 건너뜀]
$apiPort = Get-Cfg 'API_PORT' '8000'
if ($DryRun) {
    Write-Host ("🔍 [DRY-RUN] 포트 충돌 점검 시뮬레이션: PG={0} NEO4J={1}/{2} API={3}" -f (Get-Cfg 'PG_PORT' '5432'), (Get-Cfg 'NEO4J_HTTP_PORT' '7474'), (Get-Cfg 'NEO4J_BOLT_PORT' '7687'), $apiPort) -ForegroundColor Magenta
} else {
    $running = cmd /c "docker compose ps --status running -q 2>nul"
    if (-not $running) {
        $conflict = $false
        foreach ($pair in @(@('PG_PORT', '5432'), @('NEO4J_HTTP_PORT', '7474'), @('NEO4J_BOLT_PORT', '7687'), @('API_PORT', '8000'))) {
            $port = [int](Get-Cfg $pair[0] $pair[1])
            if (Test-PortInUse $port) {
                Write-Host ("❌ 호스트 포트 {0} 가 이미 사용 중입니다. 다른 포트로 실행해 주세요. 예) `$env:{1}={2}; .\deploy.ps1" -f $port, $pair[0], ($port + 10000)) -ForegroundColor Red
                $conflict = $true
            }
        }
        if ($conflict) { exit 1 }
    }
}

# 5. 풀스택 서비스 기동 [빈 DB]
Write-Host ''
Write-Host '📦 Docker Compose로 서비스를 기동합니다...' -ForegroundColor Cyan
Write-Host '  - PostgreSQL 16 [pgvector 확장만, 빈 DB]'
Write-Host '  - Neo4j [빈 그래프]'
Write-Host "  - FastAPI 백엔드 웹 애플리케이션 [:$apiPort] — 최초 실행 시 이미지 빌드에 수 분 걸립니다"

if ($DryRun) {
    Write-Host '🔍 [DRY-RUN] docker compose up -d 명령 시뮬레이션: 완료 [성공]' -ForegroundColor Magenta
} else {
    docker compose up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host '❌ docker compose up 이 실패했습니다.' -ForegroundColor Red
        exit 1
    }
}

# 6. 자체 헬스체크 대기
Write-Host ''
$healthUrl = "http://localhost:$apiPort/health"
$maxRetries = if ($env:HEALTH_RETRIES) { [int]$env:HEALTH_RETRIES } else { 60 }
$healthy = $false
Write-Host ("⏳ 시스템 자체 건강검진 [/health] 응답 대기 중... [최대 {0}초]" -f ($maxRetries * 5)) -ForegroundColor Cyan

if ($DryRun) {
    Write-Host '🔍 [DRY-RUN] /health 엔드포인트 응답 200 OK 시뮬레이션: 통과' -ForegroundColor Magenta
    $healthy = $true
} else {
    for ($i = 1; $i -le $maxRetries; $i++) {
        try {
            $res = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
            if ($res.StatusCode -eq 200) {
                $healthy = $true
                break
            }
        } catch {}
        Write-Host "  ... 서비스 초기화 중입니다 ($i/$maxRetries - 5초 대기)"
        Start-Sleep -Seconds 5
    }
}

Write-Host '============================================================' -ForegroundColor Cyan
if ($healthy) {
    Write-Host '🎉 세무 RAG 서비스가 기동되었습니다!' -ForegroundColor Green
    Write-Host "👉 웹 브라우저 접속: http://localhost:$apiPort" -ForegroundColor Green
    Write-Host "👉 헬스체크 상태 확인: $healthUrl" -ForegroundColor Green
    Write-Host '💡 안내: 빈 DB 상태입니다. /health 의 "검색 테이블 없음" 경고는 정상입니다.' -ForegroundColor Yellow
    Write-Host '   데이터는 db_build/README.md 를 보고 직접 구축해 주세요.' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Cyan
} else {
    Write-Host '⚠️ 컨테이너는 기동했으나 헬스체크 응답이 없습니다.' -ForegroundColor Yellow
    Write-Host '  로그 확인 명령어: docker compose logs -f' -ForegroundColor Yellow
    Write-Host '============================================================' -ForegroundColor Cyan
    exit 1
}
