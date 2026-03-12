# Hailo-8 YOLOv8 Live Detection Server 사용 가이드

## 개요

Raspberry Pi 5 + Hailo-8 AI 가속기 + USB 카메라(Logitech C920)를 이용한
실시간 객체 감지(Object Detection) 웹 서버입니다.

- **모델**: YOLOv8s (COCO 80 클래스)
- **추론 가속**: Hailo-8 NPU (30 FPS)
- **스트리밍**: Flask MJPEG 스트림
- **UI**: 웹 브라우저에서 실시간 확인

## 시스템 요구사항

| 항목 | 내용 |
|------|------|
| 보드 | Raspberry Pi 5 |
| AI 가속기 | Hailo-8 (PCIe) |
| 카메라 | USB 웹캠 (Logitech C920 등) |
| OS | Raspberry Pi OS (Debian trixie) |
| 패키지 | `hailo-all`, `python3-opencv`, `python3-flask` |

## 파일 구성

```
/home/pw/ws/
├── flask_yolo_app.py      # Flask 웹 서버 (메인)
├── hailo_yolo_cam.py      # CLI 버전 (5초 녹화용)
├── hailo_yolo_output.mp4  # 녹화 샘플 영상
└── hailo_yolo_snapshot.jpg # 스냅샷 샘플
```

## 사용법

### 1. 서버 시작

```bash
# 포그라운드 실행
python3 flask_yolo_app.py

# 백그라운드 실행 (SSH 세션 종료 후에도 유지)
nohup python3 /home/pw/ws/flask_yolo_app.py > /tmp/flask_yolo.log 2>&1 &
```

서버가 시작되면 다음 메시지가 출력됩니다:
```
Starting Flask server on http://0.0.0.0:5000
```

### 2. 웹 브라우저 접속

같은 네트워크의 PC/모바일에서:
```
http://<라즈베리파이IP>:5000
```

예시: `http://192.168.100.46:5000`

### 3. 웹 UI 기능

| 영역 | 설명 |
|------|------|
| **영상 패널** (좌측) | MJPEG 실시간 스트리밍, bbox + 클래스명 + 신뢰도 오버레이 |
| **Video Source** (우측 상단) | USB 카메라(Live) 또는 MP4 파일 선택 |
| **Detections** (우측 하단) | 감지된 객체 목록, 신뢰도(%), 클래스별 카운트 |
| **FPS 배지** (헤더 우측) | 실시간 초당 프레임 수 |

### 4. 비디오 소스 변경

- 드롭다운에서 `USB Camera (Live)` 선택 → 실시간 카메라
- 드롭다운에서 MP4 파일 선택 → 해당 영상에 대해 YOLO 추론 (반복 재생)
- `/home/pw/ws/` 폴더에 MP4/AVI 파일을 넣으면 자동으로 목록에 표시

### 5. 서버 중지

```bash
# PID로 종료
kill <PID>

# 또는 프로세스 이름으로 종료
pkill -f flask_yolo_app.py
```

## API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/` | GET | 웹 UI 페이지 |
| `/video_feed` | GET | MJPEG 스트림 |
| `/detections` | GET | 현재 감지 결과 JSON |
| `/sources` | GET | 사용 가능한 비디오 소스 목록 |
| `/set_source` | POST | 비디오 소스 변경 (`{"source": "camera"}`) |

### 감지 결과 JSON 예시

```json
{
  "fps": 30.0,
  "source": "camera",
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.912,
      "bbox": [478, 52, 639, 478]
    }
  ]
}
```

## CLI 버전 (녹화용)

```bash
# 5초 영상 녹화 + 스냅샷 저장
python3 hailo_yolo_cam.py
```

출력 파일:
- `hailo_yolo_output.mp4` — 5초 녹화 영상
- `hailo_yolo_snapshot.jpg` — 첫 감지 시 스냅샷

## 설정 변경

`flask_yolo_app.py` 상단 상수 수정:

```python
CONF_THRESHOLD = 0.4    # 감지 신뢰도 임계값 (낮추면 더 많이 감지)
VIDEO_DIR = "/home/pw/ws"  # MP4 파일 검색 폴더
```

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| 카메라 인식 안 됨 | `v4l2-ctl --list-devices`로 확인, `/dev/video0` 존재 여부 체크 |
| Hailo 디바이스 오류 | `hailortcli fw-control identify`로 보드 상태 확인 |
| 첫 프레임 핑크색 | 카메라 웜업 문제, 자동으로 처리됨 (10프레임 스킵) |
| 포트 충돌 | `lsof -i :5000`으로 확인 후 해당 프로세스 종료 |
