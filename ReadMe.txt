Digital Beamforming Simulator v1.7.0 안내

이 파일은 이전 파일명을 사용하던 환경과의 호환성을 위해 유지합니다.
설치, 실행, 테스트, 수치 모델, 운영 배포의 기준 문서는 README.md입니다.

v1.7.0 주요 기능

- 상호 결합과 소자 위치·진폭·위상 보정 오차
- 편파와 실측 Co/Cross-polar 소자 패턴 CSV
- Wideband Beam Squint와 Near-field Beam Focusing
- 채널·잡음·다중경로·SINR 분석
- MVDR·LCMV 적응 빔포밍과 MUSIC DOA 추정
- MATLAB·측정 Golden Dataset 교차 검증
- 설정 JSON과 Git Commit·난수 시드·계산 방식을 포함한 재현성 ZIP

Windows PowerShell 실행

cd D:\002_Source\001_AESA\002_Digital_BF
uv sync --frozen
uv run --frozen --no-sync streamlit run main.py

기존 .venv 접근 오류가 발생하면 다음 명령으로 직접 실행할 수 있습니다.

.\.venv\Scripts\python.exe -m streamlit run main.py

브라우저 접속 주소

http://localhost:8501

8501 포트를 사용 중이면 다음과 같이 다른 포트를 지정합니다.

uv run --frozen --no-sync streamlit run main.py --server.port 8502

품질 검증

.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe benchmarks\check_full_performance_regression.py

상세한 입력 파일 형식과 모델 가정·제한사항은 README.md를 확인하십시오.
