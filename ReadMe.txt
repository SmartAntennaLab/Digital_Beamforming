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
git remote add origin https://github.com/사용자이름/저장소이름.git

# 6. 원격 저장소로 코드 업로드
git push -u origin main
