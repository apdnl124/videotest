#!/usr/bin/env python3
"""
AI 동영상 검색 시스템
AWS Bedrock + Rekognition을 활용한 스마트 비디오 분석 및 검색

주요 기능:
- 비디오 업로드 및 자동 AI 분석
- 자연어 기반 장면 검색
- 스마트 프레임 추출 (장면 변화 감지)
- 실시간 처리 상태 모니터링
"""

import os
import subprocess
import tempfile
import time
import uuid
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import boto3
from botocore.exceptions import ClientError
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size

# AWS 설정
S3_BUCKET = 'ai-video-search-2025'
AWS_REGION = 'ap-northeast-2'

# 전역 변수로 작업 상태 추적
job_status = {}
completed_jobs = {}
video_metadata = {}  # 비디오별 메타데이터 저장

class AIVideoSearchProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=AWS_REGION)
        self.rekognition_client = boto3.client('rekognition', region_name=AWS_REGION)
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=AWS_REGION)
        self.mediaconvert_client = boto3.client("mediaconvert", region_name=AWS_REGION)
        
        # 한국어-영어 키워드 매핑
        self.keyword_mapping = {
            # 행동/동작
            "싸움": ["fight", "combat", "battle", "fighting"],
            "칼싸움": ["sword", "fight", "combat", "blade", "weapon"],
            "달리기": ["run", "running", "sprint", "jogging"],
            "춤": ["dance", "dancing", "performer"],
            "요리": ["cooking", "chef", "kitchen", "food"],
            "운전": ["driving", "car", "vehicle", "road"],
            "걷기": ["walking", "pedestrian", "person"],
            
            # 감정
            "웃음": ["smile", "laugh", "happy", "joy"],
            "울음": ["cry", "sad", "tears", "crying"],
            "화남": ["angry", "mad", "furious", "anger"],
            "놀람": ["surprised", "shock", "amazed"],
            
            # 객체
            "자동차": ["car", "vehicle", "automobile", "truck"],
            "집": ["house", "home", "building", "architecture"],
            "나무": ["tree", "forest", "nature", "plant"],
            "동물": ["animal", "pet", "dog", "cat"],
            "음식": ["food", "meal", "eating", "restaurant"],
            "사람": ["person", "people", "human", "man", "woman"],
            
            # 장소
            "바다": ["sea", "ocean", "water", "beach"],
            "산": ["mountain", "hill", "landscape"],
            "도시": ["city", "urban", "building", "street"],
            "공원": ["park", "garden", "outdoor"],
            # 사람 관련
            "여자": ["woman", "female", "girl", "lady"],
            "남자": ["man", "male", "boy", "gentleman"],
            "혼자": ["alone", "single", "solo", "one person"],
            "둘이": ["two people", "couple", "pair", "duo"],
            "여러명": ["multiple people", "group", "crowd", "many people"],
        }
    
    def extract_frames_smart(self, video_path, fps=0.5):
        """스마트 프레임 추출 - 장면 변화 감지"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"비디오 파일을 열 수 없습니다: {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps
        
        print(f"비디오 정보: {duration:.1f}초, {total_frames}프레임, {video_fps:.1f}fps")
        
        # 프레임 간격 계산
        frame_interval = int(video_fps / fps)
        
        prev_frame = None
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 장면 변화 감지 (선택적)
                if prev_frame is not None:
                    # 히스토그램 비교로 장면 변화 감지
                    hist1 = cv2.calcHist([prev_frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                    hist2 = cv2.calcHist([frame], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
                    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                    
                    # 장면 변화가 클 때만 추출 (임계값: 0.7)
                    if correlation < 0.7 or extracted_count == 0:
                        timestamp = frame_count / video_fps
                        frame_data = self.save_frame(frame, timestamp)
                        if frame_data:
                            frames.append(frame_data)
                            extracted_count += 1
                            print(f"프레임 추출: {timestamp:.1f}초 (변화도: {1-correlation:.3f})")
                else:
                    # 첫 번째 프레임은 항상 추출
                    timestamp = frame_count / video_fps
                    frame_data = self.save_frame(frame, timestamp)
                    if frame_data:
                        frames.append(frame_data)
                        extracted_count += 1
                        print(f"첫 프레임 추출: {timestamp:.1f}초")
                
                prev_frame = frame.copy()
            
            frame_count += 1
            
            # 최대 30개 프레임까지만 추출
            if extracted_count >= 30:
                break
        
        cap.release()
        print(f"총 {extracted_count}개 프레임 추출 완료")
        return frames
    
    def save_frame(self, frame, timestamp):
        """프레임을 임시 파일로 저장"""
        try:
            # OpenCV BGR을 RGB로 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 임시 파일로 저장
            temp_path = f"/tmp/frame_{uuid.uuid4().hex}_{timestamp:.1f}.jpg"
            pil_image.save(temp_path, 'JPEG', quality=85)
            
            return {
                'path': temp_path,
                'timestamp': timestamp,
                'size': os.path.getsize(temp_path)
            }
        except Exception as e:
            print(f"프레임 저장 실패: {e}")
            return None
    
    def analyze_frame_with_rekognition(self, frame_data):
        """Rekognition을 사용한 프레임 분석"""
        try:
            with open(frame_data['path'], 'rb') as image_file:
                image_bytes = image_file.read()
            
            results = {
                'timestamp': frame_data['timestamp'],
                'labels': [],
                'faces': [],
                'text': [],
                'moderation': []
            }
            
            # 라벨 감지
            try:
                response = self.rekognition_client.detect_labels(
                    Image={'Bytes': image_bytes},
                    MaxLabels=15,
                    MinConfidence=60
                )
                results['labels'] = [
                    {
                        'Name': label['Name'],
                        'Confidence': round(label['Confidence'], 2),
                        'Categories': [cat['Name'] for cat in label.get('Categories', [])]
                    }
                    for label in response['Labels']
                ]
            except Exception as e:
                print(f"라벨 감지 실패: {e}")
            
            # 얼굴 감지
            try:
                response = self.rekognition_client.detect_faces(
                    Image={'Bytes': image_bytes},
                    Attributes=['ALL']
                )
                results['faces'] = [
                    {
                        'Confidence': round(face['Confidence'], 2),
                        'AgeRange': face.get('AgeRange', {}),
                        'Gender': face.get('Gender', {}),
                        'Emotions': face.get('Emotions', [])[:3]  # 상위 3개 감정
                    }
                    for face in response['FaceDetails']
                ]
            except Exception as e:
                print(f"얼굴 감지 실패: {e}")
            
            # 텍스트 감지
            try:
                response = self.rekognition_client.detect_text(
                    Image={'Bytes': image_bytes}
                )
                results['text'] = [
                    {
                        'DetectedText': text['DetectedText'],
                        'Confidence': round(text['Confidence'], 2),
                        'Type': text['Type']
                    }
                    for text in response['TextDetections']
                    if text['Confidence'] > 70
                ]
            except Exception as e:
                print(f"텍스트 감지 실패: {e}")
            
            # 콘텐츠 조정 (선택적)
            try:
                response = self.rekognition_client.detect_moderation_labels(
                    Image={'Bytes': image_bytes},
                    MinConfidence=50
                )
                results['moderation'] = [
                    {
                        'Name': mod['Name'],
                        'Confidence': round(mod['Confidence'], 2),
                        'ParentName': mod.get('ParentName', '')
                    }
                    for mod in response['ModerationLabels']
                ]
            except Exception as e:
                print(f"콘텐츠 조정 실패: {e}")
            
            return results
            
        except Exception as e:
            print(f"Rekognition 분석 실패: {e}")
            return {
                'timestamp': frame_data['timestamp'],
                'labels': [],
                'faces': [],
                'text': [],
                'moderation': []
            }
    
    def analyze_query_with_bedrock(self, user_query):
        """Bedrock을 사용한 자연어 질의 분석"""
        try:
            prompt = f"""
사용자의 동영상 검색 질의를 분석해주세요:

질의: "{user_query}"

다음 정보를 JSON 형식으로 추출해주세요:
1. 핵심 키워드 (한국어)
2. 영어 키워드 변환
3. 검색 의도 (SCENE_SEARCH, OBJECT_DETECTION, EMOTION_SEARCH, TIME_BASED)
4. 신뢰도 임계값 (0-100)

예시:
{{
    "korean_keywords": ["칼싸움", "액션"],
    "english_keywords": ["sword", "fight", "combat", "action", "weapon"],
    "intent": "SCENE_SEARCH",
    "confidence_threshold": 70,
    "description": "칼을 사용한 전투 장면을 찾는 질의"
}}
"""
            
            response = self.bedrock_client.invoke_model(
                modelId="anthropic.claude-3-haiku-20240307-v1:0",
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.1
                })
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            # JSON 부분 추출
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis_result = json.loads(json_match.group())
                
                # 키워드 매핑 추가
                enhanced_keywords = set(analysis_result.get('english_keywords', []))
                for korean_word in analysis_result.get('korean_keywords', []):
                    if korean_word in self.keyword_mapping:
                        enhanced_keywords.update(self.keyword_mapping[korean_word])
                
                analysis_result['enhanced_keywords'] = list(enhanced_keywords)
                return analysis_result
            
        except Exception as e:
            print(f"Bedrock 분석 실패: {e}")
        
        # 폴백: 간단한 키워드 매핑
        enhanced_keywords = set()
        for korean_word, english_words in self.keyword_mapping.items():
            if korean_word in user_query:
                enhanced_keywords.update(english_words)
        
        return {
            "korean_keywords": [user_query],
            "english_keywords": list(enhanced_keywords) if enhanced_keywords else [user_query],
            "enhanced_keywords": list(enhanced_keywords) if enhanced_keywords else [user_query],
            "intent": "SCENE_SEARCH",
            "confidence_threshold": 70,
            "description": f"'{user_query}'에 대한 검색"
        }
    
    def search_scenes(self, query_analysis, video_id=None):
        """장면 검색 실행"""
        keywords = query_analysis.get('enhanced_keywords', [])
        confidence_threshold = query_analysis.get('confidence_threshold', 70)
        
        matching_scenes = []
        
        # 특정 비디오 검색 또는 전체 검색
        videos_to_search = [video_id] if video_id else list(video_metadata.keys())
        
        for vid_id in videos_to_search:
            if vid_id not in video_metadata:
                continue
                
            video_data = video_metadata[vid_id]
            
            for frame_analysis in video_data.get('frame_analyses', []):
                score = 0
                matched_labels = []
                
                # 라벨 매칭
                for label in frame_analysis.get('labels', []):
                    label_name = label['Name'].lower()
                    label_confidence = label['Confidence']
                    
                    for keyword in keywords:
                        if keyword.lower() in label_name or label_name in keyword.lower():
                            if label_confidence >= confidence_threshold:
                                score += label_confidence
                                matched_labels.append({
                                    'label': label['Name'],
                                    'confidence': label_confidence,
                                    'keyword': keyword
                                })
                
                # 텍스트 매칭
                for text in frame_analysis.get('text', []):
                    text_content = text['DetectedText'].lower()
                    for keyword in keywords:
                        if keyword.lower() in text_content:
                            score += text['Confidence'] * 0.5  # 텍스트는 가중치 0.5
                            matched_labels.append({
                                'label': f"Text: {text['DetectedText']}",
                                'confidence': text['Confidence'],
                                'keyword': keyword
                            })
                
                if score > 0:
                    matching_scenes.append({
                        'video_id': vid_id,
                        'video_name': video_data.get('filename', 'Unknown'),
                        'timestamp': frame_analysis['timestamp'],
                        'score': round(score, 2),
                        'matched_labels': matched_labels,
                        'frame_analysis': frame_analysis
                    })
        
        # 점수순으로 정렬
        matching_scenes.sort(key=lambda x: x['score'], reverse=True)
        
        return matching_scenes[:20]  # 상위 20개 결과만 반환
    
    def process_video_complete(self, video_path, job_id):
        """완전한 비디오 처리: 프레임 추출 + AI 분석 + 메타데이터 저장"""
        try:
            print(f"[{job_id}] 비디오 처리 시작")
            original_filename = os.path.basename(video_path)
            
            job_status[job_id] = {'status': 'processing', 'step': 'frame_extraction', 'progress': 10}
            
            # 1. 스마트 프레임 추출
            frames = self.extract_frames_smart(video_path, fps=0.5)
            if not frames:
                raise Exception("프레임 추출 실패")
            
            job_status[job_id] = {'status': 'processing', 'step': 'ai_analysis', 'progress': 30}
            
            # 2. 병렬 AI 분석
            frame_analyses = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_frame = {
                    executor.submit(self.analyze_frame_with_rekognition, frame): frame
                    for frame in frames
                }
                
                completed = 0
                for future in as_completed(future_to_frame):
                    try:
                        result = future.result(timeout=60)
                        frame_analyses.append(result)
                        completed += 1
                        
                        # 진행률 업데이트
                        progress = 30 + (completed / len(frames)) * 50
                        job_status[job_id]['progress'] = int(progress)
                        
                    except Exception as e:
                        print(f"프레임 분석 실패: {e}")
            
            # 3. 임시 파일 정리
            for frame in frames:
                try:
                    if os.path.exists(frame['path']):
                        os.remove(frame['path'])
                except Exception as e:
                    print(f"임시 파일 삭제 실패: {e}")
            
            job_status[job_id] = {'status': 'processing', 'step': 'saving_metadata', 'progress': 90}
            
            # 4. 메타데이터 생성 및 저장
            metadata = {
                'job_id': job_id,
                'filename': original_filename,
                'processed_at': datetime.now().isoformat(),
                'total_frames': len(frame_analyses),
                'frame_analyses': frame_analyses,
                'summary': self.generate_summary(frame_analyses)
            }
            
            # 메모리에 저장
            video_metadata[job_id] = metadata
            
            # S3에도 저장
            s3_key = f"metadata/{job_id}_metadata.json"
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=json.dumps(metadata, ensure_ascii=False, indent=2),
                ContentType='application/json'
            )
            
            job_status[job_id] = {'status': 'completed', 'step': 'finished', 'progress': 100}
            completed_jobs[job_id] = {
                'metadata': metadata,
                'completed_at': datetime.now().isoformat()
            }
            
            print(f"[{job_id}] 비디오 처리 완료")
            return metadata
            
        except Exception as e:
            print(f"[{job_id}] 비디오 처리 실패: {e}")
            job_status[job_id] = {'status': 'failed', 'error': str(e), 'progress': 0}
            return None
    
    def generate_summary(self, frame_analyses):
        """분석 결과 요약 생성"""
        all_labels = []
        all_faces = []
        all_texts = []
        
        for analysis in frame_analyses:
            all_labels.extend(analysis.get('labels', []))
            all_faces.extend(analysis.get('faces', []))
            all_texts.extend(analysis.get('text', []))
        
        # 라벨 빈도 계산
        label_counts = {}
        for label in all_labels:
            name = label['Name']
            label_counts[name] = label_counts.get(name, 0) + 1
        
        top_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_frames_analyzed': len(frame_analyses),
            'total_labels_detected': len(all_labels),
            'total_faces_detected': len(all_faces),
            'total_texts_detected': len(all_texts),
            'top_labels': [{'label': label, 'count': count} for label, count in top_labels],
            'has_faces': len(all_faces) > 0,
            'has_text': len(all_texts) > 0,
            'unique_labels': len(set(label['Name'] for label in all_labels))
        }

# 전역 프로세서 인스턴스
processor = AIVideoSearchProcessor()

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('ai_video_search.html')

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """비디오 업로드 및 처리 시작"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': '비디오 파일이 없습니다'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다'}), 400
        
        # 파일 저장
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        
        # 임시 파일로 저장
        temp_path = f"/tmp/{job_id}_{filename}"
        file.save(temp_path)
        
        # S3에 원본 비디오 업로드
        s3_key = f"videos/{job_id}_{filename}"
        processor.s3_client.upload_file(temp_path, S3_BUCKET, s3_key)
        
        # 백그라운드에서 처리 시작
        job_status[job_id] = {'status': 'queued', 'step': 'initializing', 'progress': 0}
        
        def process_video():
            processor.process_video_complete(temp_path, job_id)
            # 임시 파일 정리
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                print(f"임시 파일 삭제 실패: {e}")
        
        thread = threading.Thread(target=process_video)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': '비디오 업로드 및 처리가 시작되었습니다',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({'error': f'업로드 실패: {str(e)}'}), 500

@app.route('/api/status/<job_id>')
def get_job_status(job_id):
    """작업 상태 조회"""
    if job_id in job_status:
        return jsonify(job_status[job_id])
    else:
        return jsonify({'error': '작업을 찾을 수 없습니다'}), 404

@app.route('/api/search', methods=['POST'])
def search_videos():
    """AI 기반 비디오 검색"""
    try:
        data = request.get_json()
        user_query = data.get('query', '').strip()
        video_id = data.get('video_id')  # 특정 비디오 검색 (선택적)
        
        if not user_query:
            return jsonify({'error': '검색 질의가 없습니다'}), 400
        
        # 1. Bedrock으로 질의 분석
        query_analysis = processor.analyze_query_with_bedrock(user_query)
        
        # 2. 장면 검색 실행
        matching_scenes = processor.search_scenes(query_analysis, video_id)
        
        return jsonify({
            'query': user_query,
            'query_analysis': query_analysis,
            'results': matching_scenes,
            'total_results': len(matching_scenes)
        })
        
    except Exception as e:
        return jsonify({'error': f'검색 실패: {str(e)}'}), 500

@app.route('/api/videos/list')
def list_videos():
    """처리된 비디오 목록 조회"""
    try:
        videos = []
        for job_id, metadata in video_metadata.items():
            videos.append({
                'job_id': job_id,
                'filename': metadata.get('filename'),
                'processed_at': metadata.get('processed_at'),
                'summary': metadata.get('summary'),
                'total_frames': metadata.get('total_frames', 0)
            })
        
        # 최신순으로 정렬
        videos.sort(key=lambda x: x['processed_at'], reverse=True)
        
        return jsonify({
            'videos': videos,
            'total_count': len(videos)
        })
        
    except Exception as e:
        return jsonify({'error': f'비디오 목록 조회 실패: {str(e)}'}), 500

@app.route('/api/video/<job_id>/details')
def get_video_details(job_id):
    """비디오 상세 정보 조회"""
    try:
        if job_id not in video_metadata:
            return jsonify({'error': '비디오를 찾을 수 없습니다'}), 404
        
        return jsonify(video_metadata[job_id])
        
    except Exception as e:
        return jsonify({'error': f'비디오 정보 조회 실패: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=False)

@app.route('/api/convert_quality', methods=['POST'])
def convert_quality():
    """화질 변환 API"""
    try:
        data = request.get_json()
        video_key = data.get('video_key')
        target_quality = data.get('target_quality', 'SD')
        
        if not video_key:
            return jsonify({"error": "비디오 키가 필요합니다"}), 400
        
        # S3 경로 설정
        input_s3_path = f"s3://{S3_BUCKET}/{video_key}"
        output_s3_path = f"s3://{S3_BUCKET}/converted/"
        
        # MediaConvert 작업 생성 (간단한 버전)
        try:
            endpoints = processor.mediaconvert_client.describe_endpoints()
            endpoint = endpoints['Endpoints'][0]['Url']
            mediaconvert_client = boto3.client('mediaconvert', 
                                             region_name=AWS_REGION,
                                             endpoint_url=endpoint)
            
            job_settings = {
                "Role": "arn:aws:iam::052402487676:role/MediaConvertServiceRole",
                "Settings": {
                    "Inputs": [{
                        "FileInput": input_s3_path
                    }],
                    "OutputGroups": [{
                        "Name": "File Group",
                        "OutputGroupSettings": {
                            "Type": "FILE_GROUP_SETTINGS",
                            "FileGroupSettings": {
                                "Destination": output_s3_path
                            }
                        },
                        "Outputs": [{
                            "NameModifier": "_SD",
                            "VideoDescription": {
                                "Width": 720,
                                "Height": 480
                            },
                            "ContainerSettings": {
                                "Container": "MP4"
                            }
                        }]
                    }]
                }
            }
            
            response = mediaconvert_client.create_job(**job_settings)
            job_id = response['Job']['Id']
            
            return jsonify({
                "success": True,
                "job_id": job_id,
                "message": "화질 변환 작업이 시작되었습니다"
            })
            
        except Exception as e:
            return jsonify({"error": f"MediaConvert 작업 생성 실패: {str(e)}"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

