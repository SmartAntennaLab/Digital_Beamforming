# 성능·부하 벤치마크

성능 검증은 개별 Directivity 알고리즘과 사용자가 실제로 여는 전체 계산 화면을
분리해 측정합니다. 모든 명령은 저장소 루트에서 잠금 환경으로 실행하십시오.

## Directivity 기준

```bash
uv run --frozen --no-sync python benchmarks/benchmark_directivity.py
uv run --frozen --no-sync python benchmarks/benchmark_directivity.py \
  --sizes 64 128 --modes auto fast
uv run --frozen --no-sync python benchmarks/check_performance_regression.py
```

기본 CI 상한은 64×64 exact 5초, fast 6초입니다. 결과 유효성, 실제 적용 모드,
exact/fast 차이 0.5 dB 이하와 peak RSS 증가량을 함께 확인합니다. 환경별 명시적
조정은 `DBF_PERF_EXACT_64_SECONDS`, `DBF_PERF_FAST_64_SECONDS`,
`DBF_PERF_MAX_RSS_DELTA_MIB`로 할 수 있습니다.

## 전체 기능 성능 회귀

```bash
uv run --frozen --no-sync python benchmarks/check_full_performance_regression.py
```

다음 네 workload를 독립적으로 실행하고 wall time, peak RSS 증가량과 수치 결과
유효성을 검사합니다.

| workload | 대표 조건 | 기본 상한 |
|---|---|---:|
| `directivity` | 16×16 UPA, 전구 Directivity | 5초 |
| `surface` | 16×16 UPA, 적응형 3D Surface | 5초 |
| `multi_null` | 12×12 UPA, 4개 Null 및 전후 비교 | 8초 |
| `automatic_scan` | 8×8 UPA, preview 3D 6 frame | 12초 |
| `advanced_models` | 8×8 UPA, 배열/RF 오차·Wideband·Near-field·채널·LCMV·MUSIC | 10초 |

시간 상한은 `DBF_PERF_<WORKLOAD>_SECONDS`, 공통 메모리 상한은
`DBF_PERF_MAX_RSS_DELTA_MIB`로 조정합니다. CI runner 변동을 감안해 기본값은
회귀 차단용 완화 상한이며, 용량 산정에는 동일 호스트에서 반복 측정한 percentile을
사용하십시오.

## 다중 세션 Soak·메모리 누수

```bash
uv run --frozen --no-sync python benchmarks/soak_multi_session.py \
  --sessions 16 --iterations 50 --backend process --workers 4 \
  --memory-budget-mib 256 --json soak-result.json
```

시험은 session별 pattern과 metrics 요청을 병렬 반복하고 Process Pool의 parent와
child RSS를 합산합니다. worker warm-up 후 baseline, 시험 중 peak, GC 후 final을
측정하며 `final - baseline`이 메모리 예산을 초과하거나 계산 실패가 있으면 1로
종료합니다. `DBF_PROCESS_MAX_TASKS`와 같은 worker recycle 정책의 장기 효과를
확인하려면 iterations를 worker당 최대 작업 수보다 충분히 크게 잡으십시오.

권장 단계:

1. PR/CI: 4 sessions × 2 iterations smoke
2. 릴리스 후보: 예상 동시 세션 수 × 50회 이상
3. 운영 동등 환경: 최소 예상 세션 수명보다 긴 soak와 Prometheus/OTLP 동시 수집
4. replica 확장 시험: 동일 Redis prefix를 쓰는 여러 서비스 인스턴스에서 전역
   동시성·rate-limit rejection과 처리량 확인

Windows는 spawn 방식, Linux 운영도 명시적으로 spawn 기반 Process Pool을 사용하므로
두 플랫폼에서 적어도 한 번씩 실행하십시오. JSON 결과와 호스트 CPU/RAM, Python,
NumPy/BLAS 정보를 함께 보관해야 이전 기준과 재현성 있게 비교할 수 있습니다.
