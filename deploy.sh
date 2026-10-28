#!/usr/bin/env bash
# ==============================================================================
# 대한민국 세법 RAG 환경 구성 스크립트 (Linux / macOS)
#
# 컨테이너 3종(PostgreSQL+pgvector, Neo4j, API)을 빈 DB 상태로 기동하고 /health 를 기다린다.
# 데이터(코퍼스)는 포함되어 있지 않으며, 기동 후 db_build/ 스크립트로 직접 구축한다.
#
# 사용법:
#   ./deploy.sh              (기동 + /health 대기)
#   ./deploy.sh --dry-run    (도커 기동 없이 배포 로직만 시뮬레이션)
#   ./deploy.sh --help       (도움말 안내 출력)
#
# 환경변수(선택): PG_PORT NEO4J_HTTP_PORT NEO4J_BOLT_PORT API_PORT API_BIND  (호스트 포트·바인딩)
#                 HEALTH_RETRIES  /health 대기 횟수 (5초 간격, 기본 60)
# 필수 설정: POSTGRES_PASSWORD NEO4J_PASSWORD(8자 이상)  (.env 또는 셸 환경변수, 비어 있으면 안내 후 종료)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_help() {
    cat << 'EOF'
============================================================
 대한민국 세법 RAG 환경 구성 스크립트 안내
============================================================

사용법:
  ./deploy.sh [옵션]

옵션:
  --dry-run          도커 기동 없이 전체 배포 로직 가상 시뮬레이션
  --help, -h         본 도움말 안내 출력

환경변수(선택):
  PG_PORT / NEO4J_HTTP_PORT / NEO4J_BOLT_PORT / API_PORT   호스트 포트 (기본 5432/7474/7687/8000)
  API_BIND           API 바인딩 주소 (기본 127.0.0.1. 클라우드 등 외부 접속이 필요하면 0.0.0.0,
                     DB 포트는 항상 127.0.0.1)

안내:
  - 이 스크립트는 빈 DB로 기동합니다. 코퍼스는 포함되어 있지 않습니다.
    기동 후 db_build/README.md 를 보고 직접 구축하세요.
  - 빈 DB에서는 /health 가 "검색 테이블 없음" 경고를 내는 것이 정상입니다.
  - 질문 기능을 쓰려면 .env 에 OpenRouter API Key 를 넣어야 합니다.
  - DB 비밀번호는 저장소에 없습니다. 처음 실행하면 .env 가 만들어지고, 그 안의
    POSTGRES_PASSWORD, NEO4J_PASSWORD(8자 이상)를 직접 정해 넣어야 기동합니다.
    비어 있으면 안내를 출력하고 멈춥니다(--dry-run 도 같습니다).
    셸 환경변수로 넘겨도 됩니다: POSTGRES_PASSWORD=... NEO4J_PASSWORD=... ./deploy.sh
============================================================
EOF
    exit 0
}

# 0. CLI 옵션 파싱 (--dry-run / --help)
DRY_RUN=false
COMPOSE_CMD=(docker compose)

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "❌ 알 수 없는 옵션: $arg  (./deploy.sh --help 참고)"
            exit 2
            ;;
    esac
done

# 설정값 조회: 셸 환경변수 > .env > 기본값 (docker compose 의 치환 규칙과 동일)
cfg() {
    local v="${!1:-}"
    if [ -z "$v" ] && [ -f .env ]; then
        v="$(grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d "\"'\r" || true)"
    fi
    echo "${v:-$2}"
}

# 헬스체크용 HTTP 200 확인: curl -> wget 순서
http_ok() {
    if command -v curl &> /dev/null; then
        curl -s -f "$1" > /dev/null 2>&1
    elif command -v wget &> /dev/null; then
        wget -q -O /dev/null "$1" > /dev/null 2>&1
    else
        return 1
    fi
}

port_in_use() {
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

echo "============================================================"
echo " 🚀 대한민국 세법 RAG 환경 구성을 시작합니다 (빈 DB 기동)"
if [ "$DRY_RUN" = true ]; then
    echo " 🧪 [DRY-RUN 모드 활성화]: 도커 기동 없이 배포 로직만 시뮬레이션합니다."
fi
echo "============================================================"

# 1. Docker 데몬 실행 여부 점검
if [ "$DRY_RUN" = true ]; then
    echo "🔍 [DRY-RUN] Docker 데몬 연결 상태: 가상 통과 (OK)"
else
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker가 설치되어 있지 않습니다. Docker를 먼저 설치해 주세요."
        exit 1
    fi

    if ! docker info &> /dev/null; then
        echo "❌ Docker 데몬이 실행 중이지 않습니다. Docker 서비스를 시작해 주세요."
        exit 1
    fi

    # compose v2 플러그인(docker compose) 우선, 없으면 v1(docker-compose)
    if ! docker compose version &> /dev/null; then
        if command -v docker-compose &> /dev/null; then
            COMPOSE_CMD=(docker-compose)
        else
            echo "❌ Docker Compose가 없습니다. Docker Compose 플러그인(docker compose) 또는 docker-compose를 설치해 주세요."
            exit 1
        fi
    fi

    if ! command -v curl &> /dev/null && ! command -v wget &> /dev/null; then
        echo "❌ curl 또는 wget이 필요합니다(헬스체크). 하나를 설치해 주세요."
        exit 1
    fi
fi

# 2. 환경변수(.env) 설정 확인 및 생성
if [ ! -f .env ]; then
    echo "📄 .env 파일이 없습니다. .env.example을 기반으로 생성합니다..."
    if [ -f .env.example ]; then
        cp .env.example .env
    else
        touch .env
    fi

    if [ "$DRY_RUN" = false ] && [ -t 0 ]; then
        # OpenRouter API Key 입력 안내
        echo ""
        echo "🔑 OpenRouter API Key가 필요합니다 (OpenRouter 콘솔에서 발급)."
        read -rp "👉 OpenRouter API Key를 입력하세요 (건너뛰려면 Enter): " user_api_key
        if [ -n "$user_api_key" ]; then
            if grep -q "OPENROUTER_API_KEY=" .env; then
                sed -i.bak "s|OPENROUTER_API_KEY=.*|OPENROUTER_API_KEY=\"$user_api_key\"|" .env && rm -f .env.bak
            else
                echo "OPENROUTER_API_KEY=\"$user_api_key\"" >> .env
            fi
            echo "✅ API Key가 .env에 설정되었습니다."
        else
            echo "⚠️ API Key 입력을 건너뛰었습니다. 나중에 .env 파일에 직접 입력해 주세요."
        fi
    elif [ "$DRY_RUN" = false ]; then
        echo "⚠️ 대화형 터미널이 아니어서 API Key 입력을 건너뜁니다. 나중에 .env 파일에 직접 입력해 주세요."
    fi
fi

# 2-1. DB 비밀번호 확인 (compose 가 필수로 요구한다. 영문 오류 대신 여기서 먼저 안내하고 멈춘다. --dry-run 도 같다)
PG_PW="$(cfg POSTGRES_PASSWORD '')"
NEO_PW="$(cfg NEO4J_PASSWORD '')"
PW_PROBLEM=false
if [ -z "$PG_PW" ]; then
    echo "❌ POSTGRES_PASSWORD 가 비어 있습니다."
    PW_PROBLEM=true
fi
if [ -z "$NEO_PW" ]; then
    echo "❌ NEO4J_PASSWORD 가 비어 있습니다."
    PW_PROBLEM=true
elif [ "${#NEO_PW}" -lt 8 ]; then
    echo "❌ NEO4J_PASSWORD 는 8자 이상이어야 합니다 (현재 ${#NEO_PW}자). Neo4j 가 더 짧은 비밀번호를 거부합니다."
    PW_PROBLEM=true
fi
if [ "$PW_PROBLEM" = true ]; then
    echo ""
    echo "👉 DB 비밀번호는 저장소에 들어 있지 않습니다. 직접 정해서 적어 주세요."
    echo "   파일: $SCRIPT_DIR/.env  (.env 가 없었다면 .env.example 을 복사해 만든 파일입니다)"
    echo "     POSTGRES_PASSWORD=<영문·숫자 비밀번호>"
    echo "     NEO4J_PASSWORD=<8자 이상 비밀번호>"
    echo "   셸 환경변수로 넘겨도 됩니다: POSTGRES_PASSWORD=... NEO4J_PASSWORD=... ./deploy.sh"
    echo "   적은 뒤 ./deploy.sh 를 다시 실행하세요."
    exit 1
fi

# 3. 필수 디렉토리 사전 생성
mkdir -p logs

# 4. 호스트 포트 충돌 점검 (이미 이 스택이 떠 있으면 건너뜀)
API_PORT_VALUE="$(cfg API_PORT 8000)"
if [ "$DRY_RUN" = true ]; then
    echo "🔍 [DRY-RUN] 포트 충돌 점검 시뮬레이션: PG=$(cfg PG_PORT 5432) NEO4J=$(cfg NEO4J_HTTP_PORT 7474)/$(cfg NEO4J_BOLT_PORT 7687) API=$API_PORT_VALUE"
elif [ -z "$("${COMPOSE_CMD[@]}" ps --status running -q 2>/dev/null)" ]; then
    conflict=false
    for pair in "PG_PORT:5432" "NEO4J_HTTP_PORT:7474" "NEO4J_BOLT_PORT:7687" "API_PORT:8000"; do
        key="${pair%%:*}"
        port="$(cfg "$key" "${pair##*:}")"
        if port_in_use "$port"; then
            echo "❌ 호스트 포트 $port 가 이미 사용 중입니다. 다른 포트로 실행해 주세요. 예) $key=$((port + 10000)) ./deploy.sh"
            conflict=true
        fi
    done
    if [ "$conflict" = true ]; then
        exit 1
    fi
fi

# 5. 풀스택 컨테이너 기동 (빈 DB)
echo ""
echo "📦 Docker Compose로 서비스를 기동합니다..."
echo "  - PostgreSQL 16 (pgvector 확장만, 빈 DB)"
echo "  - Neo4j (빈 그래프)"
echo "  - FastAPI 백엔드 웹 애플리케이션 (:$API_PORT_VALUE) — 최초 실행 시 이미지 빌드에 수 분 걸립니다"

if [ "$DRY_RUN" = true ]; then
    echo "🔍 [DRY-RUN] docker compose up -d 명령 시뮬레이션: 완료 (성공)"
else
    "${COMPOSE_CMD[@]}" up -d
fi

# 6. 백엔드 런타임 자체 헬스체크 대기
echo ""
HEALTH_URL="http://localhost:${API_PORT_VALUE}/health"
MAX_RETRIES="${HEALTH_RETRIES:-60}"
RETRY_COUNT=0
HEALTHY=false
echo "⏳ 시스템 자체 건강검진(/health) 응답 대기 중... (최대 $((MAX_RETRIES * 5))초)"

if [ "$DRY_RUN" = true ]; then
    echo "🔍 [DRY-RUN] /health 엔드포인트 응답 200 OK 시뮬레이션: 통과"
    HEALTHY=true
else
    while [ "$RETRY_COUNT" -lt "$MAX_RETRIES" ]; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if http_ok "$HEALTH_URL"; then
            HEALTHY=true
            break
        fi
        echo "  ... 서비스 초기화 중입니다 ($RETRY_COUNT/$MAX_RETRIES - 5초 대기)"
        sleep 5
    done
fi

echo "============================================================"
if [ "$HEALTHY" = true ]; then
    echo "🎉 세무 RAG 서비스가 기동되었습니다!"
    echo "👉 웹 브라우저 접속: http://localhost:${API_PORT_VALUE}"
    echo "👉 헬스체크 상태 확인: $HEALTH_URL"
    echo "💡 안내: 빈 DB 상태입니다. /health 의 \"검색 테이블 없음\" 경고는 정상입니다."
    echo "   데이터는 db_build/README.md 를 보고 직접 구축해 주세요."
    echo "============================================================"
else
    echo "⚠️ 컨테이너는 기동했으나 헬스체크 응답이 없습니다."
    echo "  로그 확인 명령어: ${COMPOSE_CMD[*]} logs -f"
    echo "============================================================"
    exit 1
fi
