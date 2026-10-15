# KOSPI × 개인 투자자 수급 대시보드

KOSPI 지수와 **개인 순매수 · 개인 거래량 · 투자자예탁금 · 신용융자 잔고**를 같은 날짜 축에 놓고 보는 Streamlit 앱입니다.

## 화면 구성

| 탭 | 내용 |
|---|---|
| 📈 KOSPI + 지표 | KOSPI(라인/캔들, 로그 축) 아래에 지표 패널을 쌓아 같은 날짜로 비교. 마우스를 올리면 그날의 모든 값이 함께 표시 |
| 🔀 한 차트로 비교 | 단위가 다른 지표를 z-score 또는 "기간 시작 = 100"으로 맞춰 한 차트에 겹쳐 보기 |
| 🔗 상관관계 | 각 지표와 KOSPI 같은 기간 / 향후 N일 수익률의 상관계수 |
| 🗂 데이터 | 전체 표 + CSV 다운로드 |
| ℹ️ 지표 설명 | 지표 정의와 출처 |

상단 요약: KOSPI, 기간 개인 순매수 합계, 예탁금, 신용융자, 신용융자/예탁금 비율, **개인 투자심리 지수**

**개인 투자심리 지수** = 아래 요소의 z-score 평균 (사용 가능한 요소만 반영)
- 개인 순매수 N일 이동합
- 개인 거래대금 비중 (개인 매수+매도 ÷ 시장 전체)
- 투자자예탁금 N일 증가율
- 신용융자 N일 증가율

## 데이터 출처

| 데이터 | 1순위 | 대체 |
|---|---|---|
| KOSPI 지수 | KRX (pykrx) | 네이버 차트 |
| 개인 순매수 | KRX (pykrx) | 네이버 투자자별 매매동향 |
| 개인 거래량 · 거래대금 비중 | KRX (pykrx) | 없음 |
| 투자자예탁금 · 신용융자 | 금융투자협회 FreeSIS | 네이버 증시자금동향 |

> pykrx 는 KRX 정보데이터시스템(data.krx.co.kr) **로그인이 필요**합니다. 계정이 없으면 개인 순매수는 네이버에서 가져오고,
> 개인 거래량·거래대금 비중 패널은 표시되지 않습니다.
> 예탁금·신용융자는 보통 1~2 영업일 늦게 공표됩니다.

## 실행

```bash
pip install -r kospi_retail_dashboard/requirements.txt
streamlit run kospi_retail_dashboard/app.py
```

KRX 계정은 사이드바에 입력하거나, `kospi_retail_dashboard/.streamlit/secrets.toml.example` 을
`secrets.toml` 로 복사해 넣어 두면 자동으로 사용합니다 (환경변수 `KRX_ID`, `KRX_PW` 도 가능).

인터넷 연결 없이 화면만 확인하려면 사이드바에서 **데모 (가상 데이터)** 를 선택하세요.
직접 모은 데이터는 **CSV 업로드**로 불러올 수 있습니다 (양식 다운로드 버튼 제공).

## 테스트

```bash
cd kospi_retail_dashboard && python -m pytest -q tests
```

## 파일

- `app.py` — Streamlit 화면
- `data_sources.py` — KRX / 금융투자협회 / 네이버 수집, 단위 통일(원·주), 날짜 정렬
- `indicators.py` — 파생 지표, 투자심리 지수, 상관관계, 데모 데이터
- `charts.py` — Plotly 차트

본 앱의 지표는 참고용이며 투자 판단의 근거가 아닙니다.
