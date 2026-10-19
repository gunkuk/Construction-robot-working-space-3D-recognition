# 실행 및 재현 범위

## 1. 저장된 성능 수치 확인(CPU)

NumPy 설치 후 `python run.py verify-results`. 저장된 전체Test96 혼동행렬에서5-group-B의 mIoU와그룹별IoU를 다시 계산한다. GPU 추론 검증과는 다르다.

## 2. 실행환경

관측환경은 Python3.11,PyTorch2.5.1+cu124,Windows,RTX3070Ti8GB다. 정확한 패키지는 environment_observed.json에 보존했다. FlashAttention은 비활성화했다. requirements.txt는 관측한 핵심 버전이며 새 PC에서 일괄 설치 성공을 검증한 lockfile이 아니다. 특히 Windows spconv-cu120 2.3.6과 torch-scatter wheel의 플랫폼/CUDA 호환을 확인해야 한다. 기존 PC의 실행환경 또는 공식 Pointcept 설치 경로를 사용한다. 설치 파일·가상환경을 Git에 넣지 않는다.

## 3. 공식 데이터 검사

`python run.py check-data --data-root D:/datasets/rohbau3d`

원본배열은 데이터 제공처에서 별도로 받는다. data/splits.json은 공식354/54/96장면 목록이다. check-data는 파일 존재를 검사하며 데이터 전체 checksum 인증은 아니다.

## 4. 기존 캐시로 실제 평가

`python run.py evaluate --data-root D:/datasets/rohbau3d --cache-root D:/rohbau3d_utonia_work/cache_spill/94739f87a44ad936 --split test --output runs/test30`

백본이동결돼있으므로기존특징을재사용하고공개epoch30 head를GPU에서실제추론한다. `--scene-limit 1`은동작확인용부분평가이며최종성능이아니다. 전체Test는96장면·2,294,339,698원본점을평가해야한다. 개인경로는명령인자로제공하며소스에저장하지않는다.

## 5. 특징캐시 새로 생성

`python run.py build-cache --data-root D:/datasets/rohbau3d --cache-root D:/derived/rohbau3d/fresh_cache --backbone weights/utonia_tiny.pth --split test`

backbone은공식배포처에서받고configs/recipe.json의SHA256과일치해야한다. 원본폴더와캐시폴더는분리한다. 학습용은train/val도각각실행한다. 전체기존캐시는약112GB이며공간여유를별도로확보한다. 새캐시생성은기존캐시와다른namespace를사용한다. encoder의random serialization과GPU수치차이가있어원본재생성결과가발표수치와완전히같다고보장하지않는다. 이패키지에서원본으로부터전체캐시재생성/전체재학습은새로검증하지않았다.

## 6. 재학습(명시적으로 실행할 때만)

먼저 `python run.py train --cache-root D:/derived/rohbau3d/fresh_cache --dry-run`으로계획을확인한다.

`python run.py train --data-root D:/datasets/rohbau3d --cache-root D:/derived/rohbau3d/fresh_cache --epochs 30 --output runs/retrain --start-training`

고정백본특징의head만18종CE/AdamW로학습하고매epoch저장·5epoch전체Val평가한다. 이것은공개recipe의독립재학습adapter이며과거캐시생성/RNG/중간재개연대기를복원하는프로그램이아니다. 과거원실행소스와감독기록은로컬프로젝트에별도보존한다. 이번패키지검증에서학습을재개하지않았다. Test를재학습선택에사용하지않는다.
