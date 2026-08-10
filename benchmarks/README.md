# Directivity 성능 벤치마크

대형 UPA의 정확 pairwise 적분과 고속 비균일 전구 적분에 대해 벽시계 시간과
프로세스 peak RSS 증가량을 측정합니다. 기본 실행은 64×64와 128×128 배열에서
`exact`, `fast` 요청을 비교합니다. 128×128의 `exact` 요청은 제품 상한 정책에
따라 `fast`로 안전하게 전환되며 표의 Effective 열에 기록됩니다.

```bash
python benchmarks/benchmark_directivity.py
python benchmarks/benchmark_directivity.py --sizes 64 128 --modes auto fast
python benchmarks/benchmark_directivity.py --json /tmp/directivity-benchmark.json
```

RSS는 2 ms 간격으로 표본화하므로 실행 환경과 BLAS 구현에 따라 달라집니다.
비교 결과에는 Python/NumPy 프로세스의 기존 baseline 메모리를 제외한 증가량을
표시합니다.

## CI 성능 회귀 기준

`check_performance_regression.py`는 CI 공유 runner의 변동을 고려한 넉넉한 상한으로
64×64 exact 5초, fast 6초를 적용합니다. 두 계산이 요청한 실제 모드로 실행됐는지,
결과가 유효한지, exact/fast 차이가 0.5 dB 이하인지도 함께 검사합니다.

```bash
python benchmarks/check_performance_regression.py
DBF_PERF_EXACT_64_SECONDS=7 DBF_PERF_FAST_64_SECONDS=8 \
  python benchmarks/check_performance_regression.py
```

환경변수는 더 느린 자체 runner의 명시적 운영 조정용이며 GitHub Actions 기본값은
저장소에 기록된 5초/6초를 그대로 사용합니다.

## 기준 측정 결과

2026-08-10 Debian 12 개발 컨테이너의 동일 프로세스에서 측정한 결과입니다.

| Array | Elements | Requested | Effective | Time (s) | Peak RSS Δ (MiB) | Samples | Directivity (dBi) |
|---:|---:|---|---|---:|---:|---:|---:|
| 64x64 | 4,096 | exact | exact | 0.354 | 62.2 | pairwise | 38.041 |
| 64x64 | 4,096 | fast | fast | 0.707 | 21.7 | 80x41 | 38.212 |
| 128x128 | 16,384 | exact | fast | 2.909 | 28.1 | 80x41 | 43.480 |
| 128x128 | 16,384 | fast | fast | 2.721 | 45.6 | 80x41 | 43.480 |

64×64에서는 벡터화한 정확 계산이 더 빨랐지만 peak RSS 증가량은 고속 모드가
약 65% 작았습니다. 128×128 정확 요청은 268,435,456개 pairwise 항을 피하도록
제품 상한에서 고속 모드로 전환됐습니다. 시간은 절대 성능 보증이 아니라 같은
환경에서 모드 선택과 회귀를 비교하기 위한 기준값입니다.
