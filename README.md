# 건설로봇 작업면 인식 — Rohbau3D · epoch30 · 5-group-B

미장·도장 로봇의 작업면과 작업 제외 영역을 구분하기 위한 **3D 점군 의미 분할 연구**입니다. 공개 기준은 **epoch30 / 5-group-B**이며, 최종 Test mIoU는 **0.711466**입니다. 현재는 오프라인 점군 평가이며 로봇 제어·현장 작업 가능성을 검증한 결과는 아닙니다.

![방법 개요](results/figures/pipeline.png)

## 데이터와 공식 분할

[Rohbau3D](https://open-data.unibw.de/dataset.xhtml?persistentId=doi:10.60776/ZWJFI4)의 Dataverse 4.0 metadata snapshot을 사용했습니다. 좌표·RGB·법선을 입력으로 사용하고, 모델의 출력과 정답 라벨은 원본 18종을 유지했습니다. 데이터는 저장소에 포함하지 않습니다.

| 분할 | 장면 수 | 역할 | 현장 관계 |
|---|---:|---|---|
| Train | 354 | head 학습 | 12개 현장 |
| Validation | 54 | 개발 성능 관측 | Train과 같은 12개 현장, 장면 중복 없음 |
| Test | 96 | 성능 평가 | 별도 현장 site_00(36장면)·site_08(60장면) |

공식 benchmark 분할을 변경하지 않았습니다. **Test는 현장 단위로 분리되지만 Train/Validation은 현장을 공유합니다.** [정확한 장면 목록](data/splits.json)

## 모델과 학습

- Utonia tiny 백본을 고정하고 다중 해상도 특징을 융합하는 18종 head만 학습했습니다.
- 원본 좌표 단위 voxel 0.02, 블록 16,384점, FP16 특징 캐시와 원본점 inverse 복원을 사용했습니다.
- 실제 학습 손실은 **18종 CrossEntropy**, 최적화는 AdamW(lr 0.001, weight decay 0.01)입니다.
- **5-group-B로 새로 학습한 모델은 아닙니다.** 18종 argmax 예측과 정답을 아래 5개 그룹으로 병합해 평가했습니다.
- 사용자가 공개 checkpoint를 epoch30으로 고정했습니다. Test 최고점을 자동 선택한 결과가 아닙니다. [학습 설정](configs/recipe.json)

## 최종 성능: epoch30 · 5-group-B

Test 96장면의 **원본점 2,294,339,698개를 모두 평가**했습니다. 아래 IoU는 전체 Test 혼동행렬에서 TP/(TP+FP+FN)로 계산했으며, mIoU는 다섯 값의 산술평균입니다.

| 그룹 | 원본 라벨 번호 | Test IoU |
|---|---|---:|
| Wall+Drywall | 5 + 6 | 0.855781 |
| Openings | 12 + 13 + 3 + 7 + 14 | 0.154761 |
| Floor | 4 | 0.934807 |
| Ceiling+Slanted Ceiling | 1 + 2 | 0.955494 |
| Other | 0 + 8 + 9 + 10 + 11 + 15 + 16 + 17 | 0.656487 |
| **평균 mIoU** | **5개 그룹의 산술평균** | **0.711466** |

Openings는 **Window + Window Cutout + Ceiling Cutout + Wall Cutout + Door Rough Opening**, 천장 그룹은 **Ceiling + Slanted Ceiling**입니다. None은 Other에 포함합니다. [번호 매핑](configs/groups.json) · [표 CSV](results/group_scores.csv)

![Test 그룹별 IoU](results/figures/grouped_test.png)

개구부 그룹 IoU는 **0.154761**입니다. 전체 mIoU가 개구부 인식 성능을 뜻하지 않으며, 작업 제외 영역의 안정적인 인식이 해결됐다고 주장하지 않습니다.

![Test 오분류 분포](results/figures/confusion5.png)

## 30epoch까지의 Validation 성능 변화

![5-group-B 학습곡선](results/figures/learning_curve.png)

점으로 표시한 epoch만 실측했습니다. 저장된 각 epoch의 Validation 예측을 동일한 5-group-B 정의로 다시 집계한 곡선이며, 5-group-B 손실로 학습했다는 뜻은 아닙니다. Test 학습곡선으로 해석하지 않습니다. Test는 이미 노출된 상태에서 사용자 요청으로 추가 평가했으므로 새로운 미관측 독립 최종검증으로 표현하지 않습니다.

## 실행과 재현

```powershell
python run.py verify-results
```

공개된 전체 Test 혼동행렬에서 최종 mIoU와 그룹별 IoU를 재계산합니다. 실제 GPU 평가·특징 캐시 생성·명시적 재학습 명령은 [재현 안내](docs/REPRODUCIBILITY.md)를 따릅니다. 외부 데이터와 백본 가중치는 별도로 준비합니다. 현재 head는 semantic-only이며 instance Mask AP를 산출하지 않습니다.

[출처와 라이선스](THIRD_PARTY_NOTICES.md) · [가중치·원근거 해시](PROVENANCE.json)
