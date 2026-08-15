# 공개 서비스 운영 가이드

운영 경로는 다음 구성으로 배포합니다.

```text
Browser → Nginx/TLS → oauth2-proxy/OIDC → Streamlit ASGI replica
                                      ├→ Process Pool workers
                                      ├→ Redis (전역 슬롯·사용자 rate limit)
                                      ├→ Prometheus /metrics
                                      └→ OpenTelemetry OTLP collector
```

`main.py`는 단일 사용자 로컬 실행용 진입점입니다. 공개 서비스는 health,
readiness, metrics와 프록시 인증 헤더 검증을 포함하는 `server.py`를 실행합니다.

## 1. 잠금 의존성 설치

```bash
uv sync --frozen --extra ops --python 3.11
uv pip check
```

Redis, Prometheus client와 OpenTelemetry OTLP exporter는 `ops` extra에 정확한
버전으로 고정되어 있습니다. 운영 호스트마다 같은 `uv.lock`을 사용하십시오.

## 2. Redis와 계산 Worker

`/etc/digital-beamforming.env`를 secret 관리 도구로 생성하고 서비스 계정만 읽게
설정합니다.

```dotenv
DBF_COMPUTE_BACKEND=process
DBF_PROCESS_WORKERS=2
DBF_PROCESS_MAX_TASKS=100
DBF_SERVER_ADDRESS=127.0.0.1
DBF_SERVER_PORT=8501

DBF_REDIS_URL=redis://127.0.0.1:6379/0
DBF_REDIS_PREFIX=dbf-prod
DBF_REDIS_FAIL_CLOSED=true
DBF_GLOBAL_MAX_CONCURRENT_CALCULATIONS=4

DBF_MAX_CONCURRENT_CALCULATIONS=2
DBF_COMPUTE_QUEUE_TIMEOUT_SECONDS=1
DBF_COMPUTE_TIMEOUT_SECONDS=10
DBF_SESSION_CALCULATIONS_PER_MINUTE=120
DBF_SESSION_BURST=8
DBF_HEALTH_LOG_INTERVAL_SECONDS=30

DBF_REQUIRE_PROXY_IDENTITY=true
OTEL_SERVICE_NAME=digital-beamforming
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://127.0.0.1:4318/v1/traces
OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=http://127.0.0.1:4318/v1/metrics
```

- Process Pool은 수치 계산을 Streamlit UI 프로세스와 분리합니다. 각 worker의 BLAS
  thread 수는 기본 1이고 `DBF_PROCESS_MAX_TASKS` 이후 worker를 교체해 장기 메모리
  파편화를 제한합니다.
- 프로세스 내부 `DBF_MAX_CONCURRENT_CALCULATIONS`와 모든 replica가 공유하는
  `DBF_GLOBAL_MAX_CONCURRENT_CALCULATIONS`를 모두 적용합니다.
- Redis Lua 연산은 인증된 OIDC 사용자 단위 token bucket과 만료 가능한 전역 계산
  lease를 원자적으로 처리합니다. Redis 장애 때 기본값은 fail-closed이며 `/readyz`도
  503을 반환합니다. 제한적인 내부 환경에서만 `DBF_REDIS_FAIL_CLOSED=false`를
  검토하십시오.
- Redis에는 rate-limit bucket과 계산 lease만 저장합니다. 시뮬레이션 설정·수치
  결과·사용자 데이터는 저장하지 않으며, 제한된 계산 결과 캐시는 각 Streamlit
  프로세스 내부에만 유지됩니다.

Redis는 비밀번호 또는 mTLS, 전용 네트워크, 지속성·백업, 메모리 상한과 eviction
정책을 환경 표준에 맞게 별도로 구성하십시오. oauth2-proxy 세션은 예제처럼 별도
Redis DB 또는 별도 prefix를 사용합니다.

## 3. OIDC/SSO

Basic Auth는 사용하지 않습니다. `oauth2-proxy.cfg.example`을 저장소 밖으로 복사한
후 issuer, redirect URL, client ID, 허용 도메인·그룹을 조직의 IdP에 맞춥니다.
client secret과 cookie secret은 파일에 직접 커밋하지 말고 secret manager 또는
권한이 제한된 environment file로 주입하십시오.

```bash
oauth2-proxy --config /etc/oauth2-proxy/dbf.cfg
```

`nginx.conf.example`은 Nginx `auth_request`로 OIDC 세션을 검증하고 신뢰된
`X-Auth-Request-Email`을 HTTP와 WebSocket 요청에 전달합니다. Streamlit 포트와
oauth2-proxy 포트는 loopback에만 bind해야 합니다. 사용자가 직접 해당 포트에
접속할 수 있으면 인증 헤더를 위조할 수 있으므로 방화벽에서도 차단하십시오.

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 4. 서비스와 replica

`digital-beamforming.service.example`의 경로, 서비스 계정과 quota를 호스트 사양에
맞게 수정해 설치합니다. 계산 프로세스까지 함께 정리되도록 `KillMode=mixed`를
유지합니다.

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now digital-beamforming
curl --fail http://127.0.0.1:8501/healthz
curl --fail http://127.0.0.1:8501/readyz
```

replica를 늘릴 때는 service unit을 복사하거나 template unit으로 바꾸고 인스턴스별
`DBF_SERVER_PORT`를 8501, 8502처럼 분리한 뒤 Nginx
`digital_beamforming` upstream에 모두 추가합니다. 각 replica는 동일한
`DBF_REDIS_URL`, `DBF_REDIS_PREFIX`, 전역 제한 및 Streamlit cookie secret을
사용해야 합니다. Streamlit 세션은 WebSocket 연결 동안 한 replica에 유지되며,
재연결 동작을 위해 load balancer의 session affinity도 검토하십시오.

`CPUQuota`, `MemoryMax`, `TasksMax`는 replica와 그 Process Pool 전체에 적용됩니다.
`DBF_PROCESS_WORKERS × replica 수`와 BLAS thread 수를 합산해 실제 CPU 코어와
메모리를 초과하지 않게 설정하십시오.

## 5. Prometheus와 OpenTelemetry

`server.py`가 제공하는 probe는 다음과 같습니다.

| 경로 | 의미 | 공개 여부 |
|---|---|---|
| `/healthz` | ASGI 프로세스 생존 | 내부 전용 |
| `/readyz` | Redis 전역 조정 사용 가능 | 내부 전용 |
| `/metrics` | Prometheus 지표 | collector 전용 |

Nginx 예제는 이 경로를 loopback에만 허용합니다. 원격 Prometheus를 사용하면
collector CIDR만 추가하십시오. `prometheus.yml.example`에는 scrape 설정이,
`prometheus-alerts.yml.example`에는 대표 경보 규칙이 포함됩니다.

주요 지표는 계산 결과·시간, 프로세스/전역 활성 계산, 대기 계산, Redis 가용성,
worker inflight, RSS, busy/rate/deadline/cancel 누계입니다. label은 view, 배열 형상,
backend와 결과처럼 제한된 집합만 사용하며 사용자 ID나 세션 ID를 label로 쓰지
않습니다.

OTLP endpoint가 설정되면 계산 구간 trace와 계산 요청·지연 metric을 batch
export합니다.
`otel-collector.yaml.example`은 OTLP/HTTP 수신 예제입니다. TLS, 인증, tail sampling,
보존 기간은 사용하는 관측 플랫폼에 맞게 추가하십시오.

권장 경보:

- `dbf_compute_coordinator_available == 0`
- global active가 전역 상한에 5분 이상 근접
- busy/rate/deadline 결과의 지속 증가
- RSS가 `MemoryMax`의 80% 초과 또는 재시작/OOM 증가
- p95 계산 시간이 서비스 목표 초과

## 6. 부하·Soak·성능 회귀

배포 전과 수치 계산 변경 후 다음 게이트를 실행합니다.

```bash
uv run --frozen --no-sync python benchmarks/check_performance_regression.py
uv run --frozen --no-sync python benchmarks/check_full_performance_regression.py
uv run --frozen --no-sync python benchmarks/soak_multi_session.py \
  --sessions 16 --iterations 50 --backend process --workers 4 \
  --memory-budget-mib 256 --json soak-result.json
```

전체 회귀 게이트는 Directivity, 적응형 3D Surface, 4개 다중 Null, 자동 스캔을
각각 시간·peak RSS·결과 유효성으로 검사합니다. soak 시험은 여러 세션이 pattern과
metrics 계산을 동시에 반복하는 동안 부모와 child worker 전체 RSS의 warm baseline
대비 증가량, 처리량과 실패 건수를 검사합니다.

Redis 통합 시험과 실제 spawn worker 시험은 다음과 같이 실행합니다.

```bash
RUN_REDIS_INTEGRATION=1 DBF_REDIS_URL=redis://127.0.0.1:6379/15 \
  uv run --frozen --no-sync python -m unittest \
  tests.integration.test_redis_coordination -v

RUN_PROCESS_POOL_INTEGRATION=1 \
  uv run --frozen --no-sync python -m unittest \
  tests.test_compute_tasks.ComputeTaskTests.test_process_executor_runs_in_spawned_worker -v
```

긴 soak는 릴리스 후보와 동일한 replica 수·worker 수·quota에서 최소 예상 세션
수명보다 오래 실행하십시오. CI는 빠른 smoke 구성을 실행하고, 실제 용량 판단은
운영과 같은 호스트에서 수집한 JSON·Prometheus·OTLP 결과를 기준으로 합니다.
