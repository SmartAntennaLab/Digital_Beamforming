## 공개 싸이트
https://digitalbeamforming-zcmwsx6pp54mpfpzcnoygw.streamlit.app/

#0. 실행 방법
python -m streamlit run main.py

# 1. 로컬 Git 저장소 초기화
git init

# 2. 업로드할 파일들 스테이징 영역에 추가 (생성한 .gitignore 덕분에 .venv 폴더는 제외됩니다)
git add .

# 3. 첫 커밋 생성
git commit -m "Initial commit"

# 4. 기본 브랜치 이름을 main으로 설정
git branch -M main

# 5. GitHub 원격 저장소 연결 (복사한 URL을 대입)
git remote add origin https://github.com/SmartAntennaLab/Digital_Beamforming.git

# 6. 원격 저장소로 코드 업로드
git push -u origin main


파이썬(특히 Streamlit)으로 개발 중인 프로그램을 타인과 공유하는 방법은 크게 3가지가 있습니다. 현재 프로젝트가 Streamlit 기반(streamlit run main.py)이므로, 일반적인 데스크톱 실행파일(.exe)로 빌드하는 것보다 웹을 통해 공유하거나 소스 코드를 공유하는 것이 훨씬 수월합니다.

1. Streamlit Community Cloud를 통한 웹 배포 (가장 추천)
프로그램을 클라우드 서버에 올리고, 상대방에게 **웹 주소(URL)**만 공유하는 방법입니다. 상대방은 파이썬 설치 없이 웹 브라우저에서 바로 실행할 수 있습니다.

준비물: GitHub 계정 및 코드 업로드 (이미 

ReadMe.txt
에 관련 Git 명령어가 작성되어 있습니다.)
방법:
Streamlit Community Cloud에 가입하고 GitHub 계정을 연동합니다.
"New app" 버튼을 클릭합니다.
GitHub 저장소(Digital_Beamforming), 브랜치(main), 그리고 메인 파일 경로(main.py)를 지정합니다.
**"Deploy!"**를 클릭하면 수 분 내에 누구나 접속할 수 있는 웹 링크가 생성됩니다.
2. 동일한 네트워크(Wi-Fi/LAN) 내에서 로컬 공유
같은 사무실이나 공유기를 사용하는 네트워크 환경이라면, 본인 PC에서 서버를 띄워 IP 주소로 바로 접속하게 할 수 있습니다.

방법:
터미널에서 python -m streamlit run main.py를 실행합니다.
실행 시 출력되는 터미널 메시지 중 Network URL: 주소(예: http://192.168.x.x:8501)를 복사합니다.
같은 Wi-Fi/네트워크에 연결된 동료에게 이 주소를 공유하면 상대방이 웹 브라우저로 접속할 수 있습니다. (주의: 본인 PC의 방화벽에서 해당 포트(기본 8501)가 열려 있어야 할 수 있습니다.)
3. 소스 코드 및 가상환경 설정 공유
상대방도 개발자이거나 파이썬 실행 환경을 갖추고 있다면, 코드를 직접 공유하여 실행하게 합니다.

방법:


requirements.txt
와 

main.py
를 GitHub 또는 압축파일(.zip)로 공유합니다.
상대방은 파일을 받아 아래 명령어로 패키지를 설치하고 실행합니다.
bash
pip install -r requirements.txt
python -m streamlit run main.py
4. 독립형 실행파일(.exe)로 만들고 싶은 경우 (난이도 높음)
Streamlit은 웹 서버 기반이므로 일반적인 PyInstaller 패키징 방식으로는 잘 작동하지 않거나 설정이 복잡합니다. 꼭 인터넷 연결 없이 단독 실행파일로 배포해야 한다면 다음과 같은 오픈소스 도구를 활용해야 합니다.

PyInstaller + PyPortable 등의 래퍼 사용
stlite: WebAssembly 기술을 이용해 브라우저 내부에서만 구동되도록 하거나 Electron과 결합하여 데스크톱 앱으로 빌드할 수 있는 프로젝트입니다.
상황에 맞는 가장 편리한 방식을 선택해 보시기 바랍니다. 추가로 도움이 필요하시거나 특정 방식의 구체적인 가이드가 필요하시면 편하게 말씀해 주세요!


--------------------------------
# 1. 변경된 파일 스테이징
git add .

# 2. 커밋 생성
git commit -m "feat: Release v1.1 - Add Amplitude Tapering, Element Factor, Grating Lobe Detection, 3D Optimization, and Data Export"

# 3. GitHub 원격 저장소로 업로드
git push origin main
--------------------------