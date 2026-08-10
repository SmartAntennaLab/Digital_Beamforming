# Digital Beamforming Simulator

ULA·UPA·UCA와 MATLAB Phased Array Gallery 방식의 UHA 안테나 배열을 대상으로 정적 협대역 빔포밍을 분석하는 Streamlit 시뮬레이터입니다. 조향 방향, 배열 간격, 진폭 창, 위상 양자화, 소자 결함과 최대 8개의 간섭 방향 null 제약을 적용하고 2D/3D 패턴, 물리적 빔폭과 전구 적분 Directivity를 함께 확인할 수 있습니다.

이 프로그램은 RF/IQ 수신기나 실제 안테나 하드웨어를 제어하지 않습니다.

## 주요 기능

- ULA, UPA, UCA, Uniform Hexagonal Array(UHA)와 배포 정책 내 대규모 배열
- 형상과 실제 배열 차원을 반영한 Azimuth/Elevation 조향 범위
- Uniform, Hamming, Hanning, Blackman, Bartlett 진폭 창
- 이상적 위상 또는 2–6비트 위상 양자화
- Isotropic, Cosine, Cosine², Z축 반파장 Dipole 소자 패턴
- 고정 시드의 재현 가능한 소자 결함 시뮬레이션
- 최대 8개 간섭원, 방향별 요구 억압량, SVD/위상 전용 null 최적화와 진폭 상한
- 위상 전용 deterministic restart와 진폭 제한 bound-constrained 재최적화
- 요구 null 미달 방향, 양자화 전·후 제약 잔차와 진폭 포화 소자 진단
- 고정 좌표각 Azimuth/Elevation 컷, 목표 방향 Great-circle 주평면 컷, 3D 구면 패턴
- 소자 배치 탭에서 수평·수직 간격과 전체 배열 길이를 파장·cm로 표시하고 전체·활성·결함 소자 수 집계
- 좌표각·실제 각거리 HPBW, FNBW, SLL, 상대 배열 이득, 전구 적분 Directivity, 테이퍼·위상 효율
- Directivity 자동·정확·고속 근사 모드, 대형 배열 상한·경고와 동일 기하 kernel 캐시
- 형상별 격자 로브 또는 공간 앨리어싱 위험 진단
- CSV 패턴 데이터와 Markdown 설계 보고서 다운로드
- 동적 탭, form, fragment와 각도·소자 청크 기반 대규모 배열 계산
- 자동 스캔의 2D 전용·3D 미리보기·전체 품질 모드, 예상 시간 안내와 종료 후 전체 품질 최종 프레임
- Custom Component v2 기반 장치·브라우저별 설정 자동 복원과 선택적 공유 링크

## v1.5 개발 상태

2026-08-10 기준으로 다음 확장을 반영했습니다.

- 좌표각 HPBW와 목표 방향을 지나는 Great-circle 실제 각거리 HPBW 분리
- 최종 가중치의 전구 방사전력 적분 기반 directional directivity 추가
- Directivity O(N²) 정확 모드와 O(N×S) 고속 근사 모드, 자동 전환·상한·기하 캐시 추가
- 다중 null, 방향별 요구 억압량, 위상 전용 최적화, 소자 진폭 상한과 미달·포화 진단 추가
- 패턴·지표·소자 UI, 설정 저장, 스캔 제어, 패턴 표본화 모듈 분리
- 동시 계산·대기열·세션 호출률·deadline·취소 및 자원 상한 보호 추가

일반 환경의 전체 회귀 검사 결과는 **143개 통과, 브라우저 E2E 1개 skip**입니다. E2E는 `RUN_E2E=1`인 GitHub Actions 전용 단계에서 Chromium으로 별도 실행합니다.

## 지원 환경

프로젝트의 공식 지원 범위는 **64비트 CPython 3.11–3.14**입니다.

| 항목 | 지원 또는 기준 |
|---|---|
| 기준 개발 환경 | CPython 3.11, Debian 12 Dev Container |
| 지원 Python | CPython 3.11, 3.12, 3.13, 3.14 |
| 지원 운영체제 | Windows, macOS, Linux |
| Streamlit | 1.60.0 |
| 패키지 설치 기준 | `pyproject.toml` + 범용 `uv.lock` |

Streamlit 1.60 자체는 Python 3.10–3.14를 지원하지만, 이 프로젝트는 고정한 NumPy 버전과 Dev Container 기준을 고려해 Python 3.11부터 지원합니다. Python 3.10 이하와 3.15 이상, PyPy는 지원·회귀 검증 범위에 포함하지 않습니다.

직접 사용하는 패키지는 다음 다섯 개이며 재현 가능한 설치를 위해 버전을 고정했습니다.

| 패키지 | 버전 | 용도 |
|---|---:|---|
| Streamlit | 1.60.0 | 웹 UI, form, 동적 탭, fragment |
| NumPy | 2.3.5 | 배열 좌표와 빔포밍 수치 계산 |
| Plotly | 6.9.0 | 2D/3D 대화형 시각화 |
| pandas | 2.3.3 | CSV 내보내기 데이터 구성 |
| psutil | 7.2.2 | 프로세스 CPU·RSS와 시스템 자원 모니터링 |

`pyproject.toml`은 Python 범위를 `>=3.11,<3.15`로 제한하고 의존성을 네 그룹으로 분리합니다.

| 그룹 | 설치 대상 |
|---|---|
| 기본 런타임 | Streamlit, NumPy, Plotly, pandas, psutil |
| `dev` extra | pip, pip-audit, pip-licenses, setuptools |
| `e2e` extra | Playwright |
| `quality` extra | Coverage.py 7.15.4, Ruff 0.16.2, mypy 2.3.0 |

`uv.lock`은 위 직접 의존성과 모든 전이 의존성의 정확한 버전, 운영체제·Python 마커, 배포 파일 SHA-256을 함께 기록하는 단일 잠금 소스입니다. 수동으로 편집하지 말고 `uv lock`으로만 갱신합니다. 이 프로젝트가 검증한 uv 버전은 **0.11.29**입니다.

## 정확한 로컬 설치 절차

가상환경 활성화 여부에 따라 다른 Python이 실행되는 일을 피하기 위해 아래 명령은 가상환경의 Python을 직접 호출합니다. 모든 명령은 저장소 루트에서 실행하십시오.

### Windows PowerShell

Python 3.11이 설치된 새 환경에서:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.11.29/install.ps1 | iex"
# PowerShell을 다시 연 뒤 저장소 루트에서 실행
uv sync --frozen --python 3.11
uv run --frozen --no-sync streamlit run main.py
```

uv 0.11.29가 이미 설치되어 있다면 첫 명령은 생략합니다. uv는 지원 범위의 Python을 찾지 못하면 호환되는 CPython을 내려받을 수 있습니다.

### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/0.11.29/install.sh | sh
# 셸을 다시 연 뒤 저장소 루트에서 실행
uv sync --frozen --python 3.11
uv run --frozen --no-sync streamlit run main.py
```

정상 실행되면 <http://localhost:8501>에 접속합니다. 서버는 실행 터미널에서 `Ctrl+C`로 종료합니다.

설치 버전은 다음 명령으로 확인할 수 있습니다.

```powershell
uv --version
uv lock --check
uv pip check
uv tree --frozen
```

`uv sync`는 `.venv`를 생성하고 잠금 파일과 정확히 일치하도록 동기화합니다. 기존 가상환경에서 패키지 충돌이 계속되면 `.venv`를 삭제한 뒤 다시 동기화하십시오. 애플리케이션 패키지는 다른 프로젝트가 사용하는 전역 Python 환경에 설치하지 않습니다.

## VS Code Dev Container

필수 확장인 **Dev Containers**를 설치한 뒤 VS Code에서 `Dev Containers: Reopen in Container`를 선택합니다.

Dev Container는 다음 순서로 동작합니다.

1. Python 3.11 Debian 12 이미지를 생성합니다.
2. `postCreateCommand`가 uv 0.11.29를 설치하고 `uv.lock`의 런타임과 `dev` extra를 `--frozen`으로 동기화한 뒤 `pip check`를 실행합니다.
3. 컨테이너에 연결되면 `uv run --frozen --no-sync streamlit run main.py --server.headless=true`를 실행합니다.
4. 8501 포트를 전달하고 VS Code 미리 보기를 엽니다.

Streamlit을 별도로 설치하거나 잠금 파일 밖에서 패키지를 추가하지 않습니다. OS 패키지 목록이 없는 상태에서 매번 수행하던 `apt update`, `apt upgrade`도 사용하지 않습니다.

## 실행과 테스트

Windows:

```powershell
uv run --frozen --no-sync streamlit run main.py
uv run --frozen --no-sync python -m unittest discover -s tests -t . -v
```

macOS/Linux:

```bash
uv run --frozen --no-sync streamlit run main.py
uv run --frozen --no-sync python -m unittest discover -s tests -t . -v
```

테스트는 UHA 좌표·실제 소자 마스크, 수학 회귀, 형상 정합성, null 조향, 청크 계산, 대규모 배열, 동적 탭과 fragment 자동 스캔을 포함합니다.

정적 검사, 수치 코어 타입 검사, branch coverage와 대표 성능 기준을 로컬에서 CI와 동일하게 실행하려면:

```bash
uv sync --frozen --extra quality
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy
uv run --frozen --no-sync coverage run -m unittest tests.test_beamforming tests.test_directivity tests.test_simulation tests.test_resource_policy
uv run --frozen --no-sync coverage report
uv run --frozen --no-sync python benchmarks/check_performance_regression.py
```

Coverage.py는 UI 계층을 제외한 수치 코어의 statement와 branch를 함께 측정하며 최소 **80%**를 요구합니다. 2026-08-10 기준 측정값은 **87.3%**입니다. mypy는 동일 수치 코어 9개 모듈을 검사하고 Ruff는 전체 Python 저장소에서 import, 오류 가능 문법, pyupgrade와 bugbear 규칙을 검사합니다.

실제 JavaScript `localStorage` 저장·새 세션 복원·새로고침을 검증하려면 별도 브라우저 의존성을 설치합니다.

```powershell
uv sync --frozen --extra e2e
uv run --frozen --no-sync python -m playwright install chromium
$env:RUN_E2E = "1"
uv run --frozen --no-sync python -m unittest tests.e2e.test_local_storage -v
```

일반 단위 테스트에서는 이 E2E가 자동으로 skip됩니다. GitHub Actions는 Windows와 Linux에서 Python 3.11·3.14 잠금 설치 및 단위 테스트를 실행하고, Linux Python 3.11에서 Chromium E2E를 실행합니다. 별도 Linux Python 3.11 `quality-gates` job이 Ruff, mypy, branch coverage, 64×64 Directivity 성능 회귀를 순차적으로 차단 기준으로 적용합니다.

개발·E2E 도구를 포함한 전체 잠금 집합을 검사하려면 다음 명령을 사용합니다.

```powershell
uv sync --frozen --all-extras
uv run --frozen --no-sync python -m pip check
uv run --frozen --no-sync python -m pip_audit --local --strict --progress-spinner off
$env:PYTHONUTF8 = "1"
uv run --frozen --no-sync python -m piplicenses --format=markdown --with-authors --with-urls --fail-on "GPL;AGPL;UNKNOWN" --partial-match
```

CI는 모든 알려진 취약점에서 실패하므로 고위험 취약점 0개보다 엄격합니다. 라이선스 검사는 GPL·AGPL 또는 알 수 없는 라이선스가 검출되면 실패합니다. `dependency-health.yml`이 매주 화요일 02:17(KST)에 같은 검사를 다시 실행하고, Dependabot이 매주 uv 및 GitHub Actions 갱신 PR을 생성합니다.

`설정 적용 및 계산`을 누르면 주요 안테나·조향·시각화·스캔 입력이 브라우저 `localStorage`의 `digital_beamforming.settings.v1` 항목에 저장됩니다. 서버나 다른 장치로 전송하지 않으므로 같은 서버에 접속해도 PC와 휴대폰은 각자 마지막 값을 복원합니다. `공유 링크 생성`을 눌렀을 때만 검증된 설정이 URL의 단일 `settings` 파라미터로 추가되며, 공유 URL은 해당 브라우저의 저장값보다 우선합니다. 공유 링크로 연 설정에서 `설정 적용 및 계산`을 누르면 그 장치에 저장하고 주소의 공유 파라미터를 제거합니다. `저장 설정 초기화`는 현재 브라우저의 저장값과 공유 URL을 지웁니다. 자동 스캔의 실행 여부와 현재 프레임은 일시적인 상태이므로 저장하지 않습니다.

브라우저 저장값은 동일한 origin(프로토콜·호스트·포트 조합)에서만 공유됩니다. 예를 들어 `localhost:8501`과 `192.168.0.10:8501`, Chrome과 Safari, 일반 창과 시크릿 창은 각각 별도 설정입니다. 브라우저 사이트 데이터를 삭제하면 저장값도 삭제됩니다.

## 디렉터리 구조

```text
Digital_Beamforming/
├── .devcontainer/
│   └── devcontainer.json      # Python 3.11 컨테이너, frozen 잠금 동기화, 앱 실행
├── .github/
│   ├── dependabot.yml         # 주간 uv·GitHub Actions 갱신 PR
│   └── workflows/
│       ├── ci.yml             # Windows/Linux, Python 3.11/3.14, Chromium E2E
│       └── dependency-health.yml # 정기 취약점·라이선스 검사
├── .streamlit/
│   └── config.toml            # 서버 포트와 CORS/XSRF 보호 설정
├── deploy/
│   ├── nginx.conf.example     # TLS·Basic Auth·IP rate/connection limit 예제
│   ├── digital-beamforming.service.example # systemd CPU·메모리 quota 예제
│   └── README.md              # 공개 서버 보호·모니터링 운영 절차
├── benchmarks/
│   ├── benchmark_directivity.py # 64×64 이상 지향도 시간·peak RSS 측정
│   └── README.md              # 실행법과 기준 측정 결과
├── tests/
│   ├── e2e/
│   │   └── test_local_storage.py # 실제 localStorage 새로고침·복원 테스트
│   ├── __init__.py
│   ├── test_app.py            # 동적 탭과 fragment 통합 테스트
│   ├── test_beamforming.py    # 수학 로직 회귀 테스트
│   ├── test_compute_governor.py # semaphore·rate limit·deadline·취소 테스트
│   ├── test_directivity.py     # 전구 적분 지향도 수치 회귀 테스트
│   ├── test_device_settings.py # 브라우저 설정 검증·공유 토큰 테스트
│   ├── test_module_boundaries.py # 분리 모듈·호환 facade 경계 테스트
│   ├── test_resource_policy.py # 배열·스캔 계산 상한 테스트
│   └── test_simulation.py     # 프레임·청크·스캔 회귀 테스트
├── array_geometry.py          # 배열 좌표·창·결함·격자 로브
├── array_math.py              # 방향 코사인·조향 벡터·소자 패턴
├── beamforming.py             # 분리된 수치 API의 호환 import 계층
├── compute_governor.py        # 프로세스 동시성·세션 빈도·취소·자원 계측
├── device_settings.py         # 장치별 설정 스키마·검증·공유 토큰
├── device_storage.py          # Streamlit CCv2 localStorage 브리지
├── directivity.py             # 전구 방사전력 적분과 목표 방향 지향도
├── directivity_controls.py    # Directivity 계산 모드와 대형 배열 경고 UI
├── exporters.py               # CSV·Markdown 내보내기
├── interferer_sampling.py     # Null 적용 전후 응답·간섭원 great-circle 컷
├── model_options.py           # 안정적인 내부 ID와 한글 UI 라벨
├── null_controls.py           # 간섭원 수 draft/applied 상태와 제약 목록
├── null_solver.py             # SVD 제한 최소제곱과 진단
├── pattern_metrics.py         # 배열 인자·정규화·HPBW/FNBW/SLL·이득
├── pattern_sampling.py        # 좌표각·Great-circle 컷과 적응형 3D 표면 샘플링
├── resource_policy.py         # 요청·프로세스·세션별 계산 정책과 절대 상한
├── scan_controls.py           # 자동 스캔 폼·품질·시간 추정·시작/중지 상태
├── settings_panel.py          # 배열·조향·Null 입력 패널 조립
├── settings_storage.py        # 브라우저 저장·공유 링크·세션 설정 초기화
├── simulation.py              # UI 독립 상태 생성·directivity·스캔 orchestration
├── simulation_cache.py        # 제한된 Streamlit 계산 캐시
├── ui_elements.py             # 소자 배치·위상·진폭·포화 렌더링
├── ui_formatters.py           # N/A·단위·진단 표시 형식
├── ui_metrics.py              # 성능 지표·Null 진단·내보내기 UI
├── ui_pattern.py              # 2D/3D 방사 패턴 Plotly 렌더링
├── ui_renderers.py            # 공통 진단과 기존 렌더러 import 호환 facade
├── main.py                    # 앱 조립·동적 탭·fragment 진입점
├── pyproject.toml             # Python 범위와 런타임·dev·E2E 직접 의존성
├── uv.lock                    # 전이 의존성 버전·플랫폼 마커·SHA-256 잠금
├── README.md                  # 기준 설치·모델·배포 문서
└── ReadMe.txt                 # README.md로 안내하는 이전 파일명 호환 문서
```

`array_geometry.py`, `array_math.py`, `null_solver.py`, `pattern_metrics.py`, `pattern_sampling.py`, `directivity.py`, `beamforming.py`, `simulation.py`는 Streamlit에 의존하지 않으므로 별도 Python 코드에서도 가져와 사용할 수 있습니다. 코어 설정은 `UPA`, `uniform`, `isotropic`, `db` 같은 안정적인 ID를 사용하고 한글 문구는 UI에서만 변환합니다.

## 좌표계와 배열 모델

배열은 Y–Z 평면에 있고 +X가 배열 정면(broadside)입니다. Azimuth `φ`, Elevation `θ`에 대한 방향 코사인은 다음과 같습니다.

```text
u_x = cos(θ) cos(φ)
u_y = cos(θ) sin(φ)
u_z = sin(θ)
```

파장은 SI 정의값인 광속 `c=299,792,458 m/s`와 입력 주파수 `f`에 대해 `λ=c/f`로 계산합니다. 파수 `k=2π/λ`, 소자 위치 `(y_n,z_n)`에 대한 조향 벡터와 배열 인자는 다음 정의를 공유합니다.

```text
a_n(φ,θ) = exp(j k (y_n u_y + z_n u_z))
AF(φ,θ)  = Σ_n w_n a_n(φ,θ)
```

최종 패턴은 배열 인자에 선택한 소자 패턴의 진폭 인자를 곱합니다. dB 패턴은 최대 진폭으로 정규화한 `20 log10` 값이며 수치 하한은 -120 dB입니다. 2D 화면은 가독성을 위해 -40 dB 아래를 잘라 표시하고, 3D dB 표면 반경은 -30 dB 아래를 0으로 표시합니다.

### 배열 형상

- **ULA:** Y축의 수평 선형 배열입니다. 수평 간격 `dy`만 사용하고 유효 수직 소자 수는 1이며 Azimuth만 독립 조향합니다.
- **UPA:** Y–Z 직교 격자입니다. 수평 간격 `dy`와 수직 간격 `dz`를 독립적으로 적용하며, 실제 소자가 두 개 이상인 축만 독립 조향합니다.
- **UCA:** Y–Z 평면의 원형 배열입니다. 수평 간격 입력 `dy`는 인접 소자 중심의 chord 거리이고, `N≥2`일 때 반지름은 `R=dy/(2 sin(π/N))`입니다. UI 모델에서는 수직 소자 수를 1로 취급하고 `dz`를 사용하지 않으며, Elevation을 0°로 고정해 Azimuth만 조향합니다.
- **UHA:** MathWorks의 [Uniform Hexagonal Array 구성 예제](https://www.mathworks.com/help/phased/ug/phased-array-gallery.html)와 동일하게 최하단·최상단 행 길이 `Nmin`, 중앙 행 길이 `Nmax`를 사용합니다. 행 길이는 `[Nmin:Nmax, Nmax-1:-1:Nmin]`, 수직 행 간격은 `dz=dy sin(60°)`이며 각 행은 Y축 중심에 정렬됩니다. `Nmin=Nmax`이면 한 행의 선형 배열로 축퇴합니다.

`안테나 배치 및 위상` 탭의 전체 수평·수직 길이는 실제 소자 중심 좌표에 대해 각각 `max(y)-min(y)`, `max(z)-min(z)`로 계산한 투영 개구 길이입니다. 따라서 결함 소자도 물리적으로 존재하면 길이와 전체 소자 수에 포함하고, UHA의 빈 저장 슬롯은 제외합니다. 실제 소자의 직경이나 외형 치수는 포함하지 않습니다.

ULA와 UPA의 격자 로브는 축별 간격을 사용한 복제 방향 `u_y+p/(dy/λ)`, `u_z+q/(dz/λ)`와 가시원 `u_y²+u_z²≤1`로 판정합니다. UHA는 기저 `a₁=(dy,0)`, `a₂=(dy/2,dz)`의 삼각 격자 reciprocal-lattice 복제 방향을 검사합니다. UCA는 분리 가능한 직교 격자가 아니므로 복제 각도를 제시하지 않고 인접 chord 간격 `dy`가 0.5λ를 초과하는지를 보수적인 공간 앨리어싱 위험으로 표시합니다.

### 진폭, 결함과 위상

기준 진폭은 행·열 창 함수와 활성 소자 마스크를 결합합니다. UHA에서는 각 행의 길이에 맞춘 대칭 수평 창과 전체 행 방향 창을 곱합니다. 내부 직사각형 저장 공간의 빈 UHA 슬롯은 실제 소자 수, 결함률, 이득 및 렌더링에서 제외됩니다. 결함 마스크는 재현성을 위해 시드 42를 사용하며 결함 소자의 가중치는 0입니다. 결함 개수는 `N×요청률/100`을 **round-half-up**으로 정하므로 정확히 0.5개인 경우 1개로 올립니다. 화면에는 입력한 요청 결함률과 정수 결함 개수로부터 다시 계산한 실제 결함률을 구분해 표시합니다.

기본 조향 위상은 목표 방향 조향 벡터의 켤레입니다. `b`비트 위상 천이기의 간격은 `Δ=2π/2ᵇ`이고 최종 위상은 `round(φ/Δ)Δ`로 양자화합니다. 진폭 0인 소자는 양자화 후에도 정확히 0으로 유지합니다.

### Null 조향 현실성 확장 (2026-08-04)

Null 조향은 배열 인자와 동일한 조향 벡터로 목표 응답 보존과 최대 8개 간섭 방향 응답 0의 제약식을 구성합니다. UI에서 각 간섭원의 Azimuth/Elevation과 요구 억압량을 개별 입력합니다. `간섭원 수`를 바꾸면 입력 영역만 즉시 추가·제거되며 현재 계산 설정은 유지됩니다. 새 간섭원의 방향과 요구 억압량을 입력한 뒤 `설정 적용 및 계산`을 눌러야 간섭원 수와 모든 입력값이 함께 계산·저장됩니다. 진폭·위상 동시 제어는 진폭 가중 제어 공간의 제약 행렬을 `A=UΣVᴴ`로 직접 SVD 분해하고 `δ=VΣ⁻¹Uᴴ(d-Au₀)`인 최소노름 보정을 적용합니다. `AAᴴ` Gram matrix나 정규방정식을 만들지 않으므로 조건수를 제곱시키지 않습니다. 제약 행렬의 rank가 부족하거나 condition number가 허용 범위를 넘으면 기본 조향 가중치로 안전하게 되돌아갑니다.

위상 전용 모드는 기준 테이퍼 진폭을 고정하고 SVD 해, 기본 조향해와 고정 시드 위상 섭동을 사용하는 여러 초기점에서 projected-gradient 최적화를 수행합니다. 모든 restart는 같은 조건에서 재현 가능하며 최종 가중 목적함수가 가장 작은 해를 선택합니다. 고급 설정에서 restart 수, restart별 최대 반복 횟수와 수렴 허용오차를 조정할 수 있습니다.

최대 소자 진폭 제한을 사용하면 SVD 해를 한 번 clipping하고 끝내지 않습니다. clipping 결과를 초기점으로 사용해 각 활성 소자의 복소 가중치를 `|wₙ|≤w_max` 원판에 반복 투영하는 bound-constrained weighted least-squares를 수행합니다. 이 최적화는 진폭 상한을 유지하면서 목표 응답과 방향별 null 잔차를 다시 줄이며, 초기 clipping 목적함수보다 나빠지는 해는 채택하지 않습니다.

두 반복 최적화는 각 반복과 선탐색 경계에서 현재 계산 lease의 취소·deadline callback을 확인합니다. 결과에는 수렴 사유, 허용오차, 반복 상한, 선택 restart, 선택해·전체 반복 수, 최종 목적함수와 매 반복의 최악 null 상대 잔차 및 목표 방향 손실이 기록됩니다. 진폭 상한, 위상 전용 제약 또는 최종 위상 양자화 때문에 정확한 null이 불가능할 수 있으므로, 최종 null 깊이가 방향별 요구량을 충족하는지도 별도로 판정합니다.

성능 탭과 Markdown 설계 보고서는 연속 제약해와 최종 위상 양자화해를 각각 다시 평가합니다. `r=Cw-d`에 대해 목표 응답 절대·상대 오차, 각 null의 절대 잔차와 목표 응답으로 정규화한 dB 잔차, 전체 제약 잔차 노름의 양자화 열화, 최대 소자 진폭과 총 가중치 전력 `Σ|wₙ|²`를 표시합니다. 성능 탭의 반복 이력 표에서는 restart·반복별 목적함수, 최악 null 상대 잔차와 목표 방향 손실을 확인할 수 있습니다. 요구 억압 충족 수와 미달 방향, 진폭 상한에 도달한 포화 소자 수를 함께 표시하며 안테나 배치 도표에서는 포화 소자를 주황색 외곽선으로 구분합니다. 잔차 열화는 `20 log10(r_final/r_continuous)`로 정의하여 양수가 클수록 양자화로 제약이 더 나빠졌음을 뜻합니다.

간섭원별 적용 전·후 비교는 배열·목표 조향·테이퍼·결함·위상 양자화 조건을 동일하게 유지하고 Null 제약만 끈 기준해를 별도로 계산합니다. 성능 표에는 실제 간섭 방향에서 목표 응답으로 정규화한 적용 전 상대 응답, 적용 후 상대 응답과 두 값의 차이인 추가 억압량을 표시합니다. 패턴 탭에는 목표 방향과 각 간섭 방향을 정확히 잇는 전용 great-circle 컷을 추가하여 두 곡선을 중첩합니다. 전역 표본 외에 목표와 간섭 방향 주변을 모두 적응형 세분화하며 간섭 방향 자체를 표본 축에 강제로 포함하므로, 고정 Azimuth/Elevation 컷이 간섭 방향을 지나지 않거나 매우 좁은 Null이 전역 격자 사이에 놓이는 문제를 피합니다. 기존 좌표각 컷의 간섭원 표시는 해당 간섭원이 실제로 그 컷 위에 있을 때만 나타납니다.

## 지표 정의

### 정규화 패턴

`P(α)=|AF(α)E(α)| / max|AF·E|`이며 dB 값은 `20 log10(P)`입니다. 모든 응답이 0이면 0으로 나누지 않고 선형 패턴 0, dB 패턴 -120 dB를 반환합니다.

### HPBW

주엽 최대점 좌우에서 진폭이 `1/√2`로 내려가는 지점 사이의 각도 폭입니다. 이는 전력 기준 절반, 약 -3.0103 dB에 해당합니다. 교차 각도는 인접 각도 표본의 **선형 진폭** 사이를 선형 보간합니다. 한쪽 교차점만 있으면 최대점을 기준으로 대칭 폭을 사용하고, 교차점이 없으면 미검출 상태인 `None`으로 보관하고 화면과 보고서에 `N/A`로 표시합니다.

성능 탭은 두 종류의 HPBW를 구분합니다.

- **좌표각 HPBW:** Elevation을 고정한 Azimuth 좌표 컷과 Azimuth를 고정한 Elevation 좌표 컷에서 잰 좌표 차이입니다.
- **실제 각거리 HPBW:** 목표 단위벡터 `u₀`와 그 지점의 수평·수직 단위 접선 `t`에 대해 `u(δ)=u₀ cosδ+t sinδ`로 만든 Great-circle 주평면에서 잰 폭입니다. 가로축 `δ` 자체가 구면 각거리이므로 큰 조향각에서도 좌표 압축·확대의 영향을 받지 않습니다.

Broadside에서는 두 정의가 일치하지만 큰 Elevation에서 고정 Elevation Azimuth 폭은 실제 수평 각거리 폭과 달라질 수 있습니다. Great-circle 컷은 -180°부터 +180°까지 전 주평면을 그리고 목표 오프셋 0°를 반드시 포함합니다.

### FNBW

주엽 최대점 양쪽에서 가장 가까운 이산 국소 최소점 사이의 폭입니다. 한쪽 null만 검출되면 최대점에서 해당 null까지의 각도 차를 두 배로 계산합니다. 양쪽 모두 없으면 `None`/`N/A`입니다. 위치 자체는 2D 컷의 각도 표본 기반 근사값입니다.

### SLL

첫 null 구간 바깥에서 가장 큰 정규화 dB 값을 최대 부엽 레벨로 사용하고 해당 각도를 함께 표시합니다. 표본 범위 안에 부엽 구간이 없으면 레벨과 각도를 `None`으로 보관하고 화면과 보고서에 `N/A`로 표시합니다. 따라서 실제로 측정된 0°, -99 dB 등의 값과 미검출 상태가 구분됩니다.

### Null 깊이

목표 응답 대비 간섭 방향 억압량입니다.

```text
Null depth = -20 log10(|AF_null| / |AF_target|)
```

화면의 실제 null 깊이는 연속 제약해가 아니라 **최종 위상 양자화까지 적용한 가중치**로 다시 측정합니다. 수치 표시는 최대 300 dB로 제한합니다.

### 상대 배열 이득과 효율

`N_active`를 실제 활성 소자 수, `w_n`을 최종 복소 가중치, `A_t=Σw_n a_n(target)`을 목표 방향 응답이라고 정의합니다.

```text
η_taper = (Σ|w_n|)² / (N_active Σ|w_n|²)
η_phase = |A_t|² / (Σ|w_n|)²
N_eff   = |A_t|² / Σ|w_n|²
        = N_active η_taper η_phase
G_relative = 10 log10(N_eff) dB
```

`G_relative`는 단일 소자 대비 목표 방향의 상대적인 coherent combining 지표입니다. 실제 활성 소자, 진폭 테이퍼, 위상 양자화와 null 조향에 따른 손실은 반영하지만, 구면 전체의 방사 패턴을 적분하지 않으므로 directivity가 아니며 절대 이득 단위인 dBi로 표시하지 않습니다. 실제 소자 이득, 급전/RF 손실, 상호 결합 손실도 포함하지 않습니다. 모든 가중치가 0이면 이 값은 `N/A`입니다.

### 전구 적분 Directivity

동일한 최종 가중치와 소자 패턴으로 목표 방향의 물리적 지향도를 상대 배열 이득과 별도로 계산합니다.

```text
U(φ,θ) = |AF(φ,θ) E(φ,θ)|²
D_target = 4π U(target) / ∫sphere U(φ,θ) dΩ
Directivity = 10 log10(D_target) dBi
```

사이드바에서 다음 세 계산 모드를 선택할 수 있으며 실제 적용 모드와 전환 사유는 성능 지표와 설계 리포트에 기록됩니다.

| 모드 | 동작 |
|---|---|
| 자동 | 활성 소자 1,024개까지 정확 모드, 그보다 크면 고속 근사 |
| 정확 | Isotropic/Cosine 계열의 해석적 O(N²) pairwise 적분. 1,024개 초과 경고, 4,096개 초과 시 고속 근사로 안전 전환 |
| 고속 근사 | 목표·반대 방향 국부 표본을 보강한 비균일 O(N×S) 전구 수치 적분 |

정확 모드의 Isotropic, Cosine, Cosine² 소자 패턴은 제한된 3D 표시 메시와 무관한 소자 쌍별 전구 적분 커널을 사용합니다. 1,024소자 이하에서 동일한 `위치/파장` 정규화 기하와 소자 패턴의 kernel은 프로세스 내 32 MiB LRU 캐시로 재사용합니다. 정확 모드의 Z축 반파장 Dipole은 `sin(Elevation)` 축의 Gauss–Legendre 구적과 주기 Azimuth 구적을 사용합니다. 고속 모드는 모든 소자 패턴에 비균일 전구 적분을 적용하고 사용한 표본 수를 화면에 표시합니다. 모든 가중치가 0이거나 방사전력 분모가 수치적으로 유효하지 않으면 `N/A`입니다.

재현 가능한 64×64·128×128 시간·메모리 측정은 [Directivity 벤치마크](benchmarks/README.md)를 참고하십시오.

표시값은 조향 **목표 방향의 directional directivity**입니다. 실제 최대 방사 방향이 목표에서 벗어난 특이 설정에서는 패턴 전체의 최대 directivity와 다를 수 있습니다. 또한 급전·RF 손실과 상호 결합을 포함하지 않으므로 realized gain이 아닙니다. 3D 표면은 계속 시각화 전용이며 directivity 적분의 분모로 재사용하지 않습니다.

## 모델 가정과 한계

- 단일 주파수의 정적 협대역, 원거리장, 평면파 모델입니다.
- 모든 소자는 위치 외에는 동일하고 시간·주파수 동기와 보정이 완벽하다고 가정합니다.
- 소자 패턴은 배열 인자에 곱하는 실수 진폭 인자이며 편파는 모델링하지 않습니다.
- Isotropic은 전 방향 1, Cosine 계열은 +X broadside 기준, Dipole은 Z축 반파장 소자 모델입니다.
- 잡음, 채널, 다중경로, 도래 신호 표본, SINR, RF/IQ 체인과 ADC/DAC는 포함하지 않습니다.
- 상호 결합, 스캔 임피던스, 소자 위치 오차, 위상·진폭 보정 오차와 열 영향은 포함하지 않습니다.
- 적응형 MVDR/LCMV, DOA 추정과 실시간 데이터 처리는 구현하지 않습니다.
- 결함은 완전 비활성 소자로만 모델링하며 부분 성능 저하는 포함하지 않습니다.
- 고정 좌표각 2D 컷은 전역 360개 표본, Great-circle 컷은 전 주평면의 전역 720개 표본에 정확한 목표점과 투영 개구 기반 국부 표본을 합칩니다. 실제 표본 수와 세분화 반폭은 패턴 탭에 표시되며, FNBW와 SLL 위치는 최종 비균일 표본 해상도의 영향을 받습니다.
- 청크 처리는 배열 인자의 임시 위상 행렬을 기본 1,000,000개 항목 이하로 제한하지만 전체 좌표와 가중치는 메모리에 유지합니다.
- 3D 전역 해상도는 소자 수에 따라 50×50에서 20×20으로 낮아지지만, 실제 목표 조향각을 반드시 포함하고 배열 개구에 따라 목표 주변에 비균일 세부 표본을 추가합니다.

## Streamlit 서버 보안 설정

저장소의 `.streamlit/config.toml`은 다음 보호 기능을 명시적으로 유지합니다.

```toml
[server]
headless = true
port = 8501
enableCORS = true
enableXsrfProtection = true
```

저장소 기본값은 서버·Dev Container·Community Cloud에서 일관되게 동작하도록 `headless=true`입니다. 로컬에서 브라우저 자동 열기가 필요하면 실행할 때만 `--server.headless=false`를 지정하십시오. Dev Container 실행 명령은 CORS/XSRF를 끄는 플래그를 사용하지 않습니다.

별도 도메인이나 reverse proxy를 사용하는 경우 보호 기능을 끄지 말고 공개 URL에 맞게 `browser.serverAddress`, `browser.serverPort`, 필요한 `server.corsAllowedOrigins`를 배포 환경에서 설정하십시오. 다중 replica에서는 동일한 비밀 `server.cookieSecret`과 session stickiness가 필요할 수 있습니다. 비밀값은 `.streamlit/secrets.toml` 또는 배포 플랫폼의 secret 관리 기능에 저장하고 Git에 커밋하지 마십시오.

### 공개 배포 계산 상한과 인증

앱은 계산을 시작하기 전에 형상별 실제 소자 수, 스캔 프레임 수, `소자 수 × 프레임 수`를 검사합니다. 기본 세션 상한은 다음과 같습니다.

| 정책 | 기본값 | 코드상 절대 상한 |
|---|---:|---:|
| 실제 소자 수 | 4,096 | 16,384 |
| Directivity 정확 모드 경고 소자 수 | 1,024 | 4,096 |
| Directivity 정확 모드 소자 상한 | 4,096 | 4,096 |
| 스캔 프레임 | 400 | 1,000 |
| 누적 element-frames | 1,000,000 | 4,000,000 |
| 프로세스 동시 계산 | 2 | 32 |
| 계산 슬롯 대기 | 1초 | 30초 |
| 계산 제한 시간 | 10초 | 120초 |
| 세션 계산 빈도 | 분당 120회 | 분당 600회 |
| 세션 순간 burst | 8회 | 60회 |

운영 환경에서는 기존 `DBF_MAX_ELEMENTS`, `DBF_MAX_SCAN_FRAMES`, `DBF_MAX_SCAN_ELEMENT_FRAMES` 외에 `DBF_DIRECTIVITY_WARNING_ELEMENTS`, `DBF_DIRECTIVITY_EXACT_MAX_ELEMENTS`, `DBF_MAX_CONCURRENT_CALCULATIONS`, `DBF_COMPUTE_QUEUE_TIMEOUT_SECONDS`, `DBF_COMPUTE_TIMEOUT_SECONDS`, `DBF_SESSION_CALCULATIONS_PER_MINUTE`, `DBF_SESSION_BURST`, `DBF_HEALTH_LOG_INTERVAL_SECONDS`를 지정할 수 있습니다. 프로세스 전체 semaphore가 동시 계산을 제한하고, 세션별 token bucket이 반복 계산을 완화합니다. 계산 deadline과 사용자 취소는 배열 인자의 각도·소자 청크 경계에서 확인하므로 진행 중인 NumPy 호출 하나는 완료된 뒤 중단됩니다.

사이드바 `서버 계산 상태`에서 동시·대기 계산, CPU와 메모리, 거절·시간 초과·취소 누계를 확인할 수 있으며 동일 정보가 `compute_health` 구조화 로그로 주기적으로 출력됩니다. 이 제한은 애플리케이션 프로세스 내부 보호이며 사용자 인증, 여러 replica 전체의 전역 제한 또는 운영체제 quota를 대신하지 않습니다. 공개 배포에서는 [deploy/README.md](deploy/README.md)의 Nginx Basic Auth·IP rate limit 예제와 systemd CPU·메모리 제한을 함께 적용하십시오.

## Streamlit Community Cloud 배포

1. GitHub 저장소에 `main.py`, `pyproject.toml`, `uv.lock`, `.streamlit/config.toml`을 포함해 push합니다.
2. Streamlit Community Cloud에서 새 앱을 만들고 저장소와 `main` 브랜치를 선택합니다.
3. 엔트리포인트를 `main.py`로 지정합니다.
4. Advanced settings에서 Python **3.11**을 선택합니다.
5. 이 앱은 비밀값을 요구하지 않으므로 Secrets 입력은 비워 둡니다.
6. 배포 로그에서 `uv.lock`이 감지되고 잠긴 의존성이 설치됐는지 확인합니다.

Community Cloud는 `uv.lock`과 `pyproject.toml`을 사용해 기본 런타임 그룹만 설치합니다. 별도 `pip install streamlit` 명령은 필요하지 않습니다. Python 버전은 배포 후 제자리에서 변경할 수 없으므로 바꾸려면 앱을 다시 배포해야 합니다.

공개 데모: <https://digitalbeamforming-zcmwsx6pp54mpfpzcnoygw.streamlit.app/>

## 성능 구조

- form 제출 전에는 입력 변경으로 비싼 계산을 다시 실행하지 않습니다.
- Streamlit 1.60 동적 탭의 `.open` 상태를 확인해 현재 탭만 계산합니다.
- 자동 스캔은 `st.fragment(run_every=...)`로 활성 탭만 갱신합니다.
- 자동 스캔의 기본값인 `3D 미리보기`는 실행 중 전역 16×16과 목표 주변 17개 표본을 사용합니다. `2D 전용`은 실행 중 3D를 생략하고, `전체 품질 3D`는 모든 프레임을 정상 해상도로 계산합니다.
- `2D 전용`과 `3D 미리보기`는 중지 또는 완료 시 마지막 조향 방향을 유지한 채 전체 품질 3D를 한 번 다시 계산합니다. UI의 프레임·총 시간은 64×64 전체 품질 프레임 약 0.85초를 기준으로 한 경험적 추정치이며 실행 CPU와 활성 탭에 따라 달라집니다.
- 배열 인자는 각도 또는 소자 축을 자동 청크 처리하고 필요하면 두 축을 함께 나눕니다.
- 좌표각·Great-circle 2D 컷은 전역 격자에 최대 129개의 목표 주변 표본을 추가해 대규모 배열의 HPBW, FNBW와 초기 부엽을 세분화합니다.
- Directivity는 성능 지표 탭을 열 때만 계산하고 제한된 결과 캐시에 저장합니다. 1,024소자 이하의 해석 kernel은 정규화 기하별 32 MiB LRU 캐시에 저장하며, 대형 배열은 자동으로 비균일 전구 적분을 사용합니다. 모든 적분 경로가 계산 취소·시간 제한을 확인합니다.
- 3D 패턴은 배열 크기에 따라 전역 각도 해상도를 제한하고, 좁은 주엽을 놓치지 않도록 목표 방향 주변만 적응형으로 세분화합니다.
- Streamlit 데이터 캐시는 `max_entries`로 상한을 둡니다.

## 출력 데이터

- `beam_pattern_data.csv`: 서로 독립적인 Azimuth/Elevation 좌표 컷과 수평/수직 Great-circle 비균일 각거리 및 각 정규화 이득. 검출 기반 파생 지표의 sentinel 값은 포함하지 않습니다.
- `beamforming_design_report.md`: 배열 설정, 조향 조건, 좌표각·실제 각거리 HPBW/FNBW/SLL, 상대 배열 이득, Directivity와 null 지표. 미검출 지표는 `N/A`로 기록합니다.

## 참고 문서

- [Streamlit 명령줄 설치](https://docs.streamlit.io/get-started/installation/command-line)
- [Streamlit 서버 설정](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)
- [Streamlit Community Cloud 배포](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [uv 프로젝트 잠금과 동기화](https://docs.astral.sh/uv/concepts/projects/sync/)
- [pip-audit](https://pypi.org/project/pip-audit/)
- [Dependabot 지원 패키지 생태계](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories)
- [MathWorks Phased Array Gallery — Uniform Hexagonal Array](https://www.mathworks.com/help/phased/ug/phased-array-gallery.html)
