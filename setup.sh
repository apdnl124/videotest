#!/bin/bash

echo "🎬 AI 동영상 검색 시스템 설치 스크립트"
echo "=========================================="

# 시스템 업데이트
echo "📦 시스템 패키지 업데이트 중..."
sudo apt update && sudo apt upgrade -y

# 필수 패키지 설치
echo "🔧 필수 패키지 설치 중..."
sudo apt install -y python3 python3-pip python3-venv ffmpeg

# Python 가상환경 생성
echo "🐍 Python 가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# Python 패키지 설치
echo "📚 Python 패키지 설치 중..."
pip install --upgrade pip
pip install -r requirements.txt

# 디렉토리 구조 생성
echo "📁 디렉토리 구조 생성 중..."
mkdir -p uploads temp logs

echo "✅ 설치 완료!"
echo ""
echo "🚀 실행 방법:"
echo "1. AWS 자격 증명 설정: aws configure"
echo "2. S3 버킷 생성 및 IAM 권한 설정"
echo "3. 애플리케이션 실행: python ai_video_search_app.py"
echo "4. 브라우저에서 http://localhost:8082 접속"
echo ""
echo "📖 자세한 내용은 README.md를 참고하세요."
