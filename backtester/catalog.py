"""자주 쓰는 종목/ETF 목록과 예시 포트폴리오."""

from __future__ import annotations

from dataclasses import dataclass

# 코드 -> (이름, 한 줄 설명, 분류)
TICKERS: dict[str, tuple[str, str, str]] = {
    # 미국 상장 ETF (달러)
    "SPY": ("SPDR S&P 500", "미국 대표 500개 기업 (S&P500)", "미국 ETF"),
    "VOO": ("Vanguard S&P 500", "S&P500, SPY보다 보수가 낮음", "미국 ETF"),
    "QQQ": ("Invesco QQQ", "나스닥 상위 100개 (기술주 중심)", "미국 ETF"),
    "VTI": ("Vanguard Total Stock Market", "미국 주식 시장 전체", "미국 ETF"),
    "SCHD": ("Schwab US Dividend Equity", "미국 배당 성장주", "미국 ETF"),
    "VEA": ("Vanguard FTSE Developed", "미국 외 선진국 주식", "미국 ETF"),
    "VWO": ("Vanguard FTSE Emerging", "신흥국 주식", "미국 ETF"),
    "TLT": ("iShares 20+Y Treasury", "미국 장기 국채 (20년 이상)", "미국 ETF"),
    "IEF": ("iShares 7-10Y Treasury", "미국 중기 국채 (7~10년)", "미국 ETF"),
    "SHY": ("iShares 1-3Y Treasury", "미국 단기 국채 (1~3년), 현금 대용", "미국 ETF"),
    "BND": ("Vanguard Total Bond", "미국 채권 시장 전체", "미국 ETF"),
    "GLD": ("SPDR Gold Shares", "금", "미국 ETF"),
    "DBC": ("Invesco DB Commodity", "원자재 (원유, 금속, 곡물 등)", "미국 ETF"),
    "DIA": ("SPDR Dow Jones", "미국 다우존스 30개 대형주", "미국 ETF"),
    "IWM": ("iShares Russell 2000", "미국 소형주 2000개", "미국 ETF"),
    "VNQ": ("Vanguard Real Estate", "미국 부동산 (리츠)", "미국 ETF"),
    "SOXX": ("iShares Semiconductor", "미국 반도체 기업", "미국 ETF"),
    "TQQQ": ("ProShares UltraPro QQQ", "나스닥100 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    # 한국 상장 ETF (원화)
    "069500": ("KODEX 200", "코스피 대표 200개 기업", "한국 ETF"),
    "229200": ("KODEX 코스닥150", "코스닥 대표 150개 기업", "한국 ETF"),
    "360750": ("TIGER 미국S&P500", "S&P500을 원화로 (환율 효과 포함)", "한국 ETF"),
    "133690": ("TIGER 미국나스닥100", "나스닥100을 원화로 (환율 효과 포함)", "한국 ETF"),
    "379800": ("KODEX 미국S&P500", "S&P500을 원화로 (환율 효과 포함)", "한국 ETF"),
    "381170": ("TIGER 미국테크TOP10 INDXX", "미국 대형 기술주 10개를 원화로", "한국 ETF"),
    "458730": ("TIGER 미국배당다우존스", "미국 배당 성장주를 원화로", "한국 ETF"),
    "091160": ("KODEX 반도체", "한국 반도체 기업", "한국 ETF"),
    "122630": ("KODEX 레버리지", "코스피200 하루 수익률의 2배 (레버리지, 위험 큼)", "한국 ETF"),
    "148070": ("KIWOOM 국고채10년", "한국 10년 국채", "한국 ETF"),
    "114260": ("KODEX 국고채3년", "한국 3년 국채", "한국 ETF"),
    "153130": ("KODEX 단기채권", "단기 채권, 현금 대용", "한국 ETF"),
    "305080": ("TIGER 미국채10년선물", "미국 10년 국채를 원화로", "한국 ETF"),
    "132030": ("KODEX 골드선물(H)", "금 (환율 영향 제거)", "한국 ETF"),
    # 개별 주식
    "005930": ("삼성전자", "한국 개별 주식", "한국 주식"),
    "000660": ("SK하이닉스", "한국 개별 주식", "한국 주식"),
    "035420": ("NAVER", "한국 개별 주식", "한국 주식"),
    "035720": ("카카오", "한국 개별 주식", "한국 주식"),
    "005380": ("현대차", "한국 개별 주식", "한국 주식"),
    "051910": ("LG화학", "한국 개별 주식", "한국 주식"),
    "207940": ("삼성바이오로직스", "한국 개별 주식", "한국 주식"),
    "373220": ("LG에너지솔루션", "한국 개별 주식", "한국 주식"),
    "AAPL": ("Apple", "미국 개별 주식", "미국 주식"),
    "MSFT": ("Microsoft", "미국 개별 주식", "미국 주식"),
    "NVDA": ("NVIDIA", "미국 개별 주식", "미국 주식"),
    "GOOGL": ("Alphabet (구글)", "미국 개별 주식", "미국 주식"),
    "AMZN": ("Amazon", "미국 개별 주식", "미국 주식"),
    "META": ("Meta (페이스북)", "미국 개별 주식", "미국 주식"),
    "TSLA": ("Tesla", "미국 개별 주식", "미국 주식"),
}


def display_name(ticker: str) -> str:
    info = TICKERS.get(ticker)
    return f"{info[0]} ({ticker})" if info else ticker


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    weights: dict[str, float]  # 코드 -> 비중(%)
    benchmark: str


PRESETS: list[Preset] = [
    Preset(
        "주식 60 : 채권 40 (미국)",
        "가장 기본적인 자산배분이에요. 주식으로 성장을, 채권으로 안정을 챙겨요. "
        "주식이 크게 떨어질 때 채권이 충격을 줄여주는 경우가 많아요.",
        {"SPY": 60, "IEF": 40},
        "SPY",
    ),
    Preset(
        "S&P500 하나만",
        "미국 대표 500개 기업에 한 번에 투자해요. 많은 사람들이 '이것보다 잘하기 어렵다'고 "
        "말하는 기준이에요.",
        {"SPY": 100},
        "SPY",
    ),
    Preset(
        "올웨더 (간단 버전)",
        "레이 달리오의 '어떤 경제 상황에서도 버티는' 포트폴리오를 단순화한 버전이에요. "
        "주식 비중이 낮고 채권·금·원자재로 나눠서 출렁임이 작아요.",
        {"VTI": 30, "TLT": 40, "IEF": 15, "GLD": 7.5, "DBC": 7.5},
        "SPY",
    ),
    Preset(
        "영구 포트폴리오",
        "주식·장기채·금·현금(단기채)에 25%씩. 호황·불황·인플레이션·디플레이션 중 "
        "무엇이 와도 하나는 버텨준다는 아이디어예요.",
        {"SPY": 25, "TLT": 25, "GLD": 25, "SHY": 25},
        "SPY",
    ),
    Preset(
        "한국 주식 60 : 채권 40",
        "코스피200 ETF와 한국 국채 10년 ETF로 만든 60:40이에요. 모두 원화라 환율 걱정이 없어요.",
        {"069500": 60, "148070": 40},
        "069500",
    ),
    Preset(
        "국내 상장 ETF로 만든 글로벌 분산",
        "한국 증권 계좌에서 바로 살 수 있는 ETF만으로 미국 주식·미국 채권·금에 나눠 담았어요. "
        "2020년 이후 상장된 ETF가 있어 기간이 짧아요.",
        {"360750": 40, "133690": 20, "305080": 25, "132030": 15},
        "069500",
    ),
    Preset(
        "직접 만들기",
        "아래 표에 종목코드와 비중을 자유롭게 입력하세요. 한국 종목은 6자리 코드, "
        "미국 종목은 티커(영문)를 쓰면 돼요.",
        {"SPY": 50, "069500": 30, "005930": 20},
        "SPY",
    ),
]


# ---------------------------------------------------------------- 웹 버전용 추가 종목
# 웹 버전(web/)은 시세를 미리 받아 싣는다. 아래 목록 + S&P500 전 종목 + 코스피·코스닥 시가총액 상위 종목
# + 한국 ETF 시가총액 상위 종목을 scripts/fetch_universe.py가 받는다.
WEB_EXTRA: dict[str, tuple[str, str, str]] = {
    # 미국 ETF: 지수
    "IVV": ("iShares Core S&P 500", "S&P500", "미국 ETF"),
    "SPLG": ("SPDR Portfolio S&P 500", "S&P500, 보수가 아주 낮음", "미국 ETF"),
    "QQQM": ("Invesco NASDAQ 100", "나스닥100, QQQ보다 보수가 낮음", "미국 ETF"),
    "RSP": ("Invesco S&P 500 Equal Weight", "S&P500 동일 비중", "미국 ETF"),
    "VUG": ("Vanguard Growth", "미국 대형 성장주", "미국 ETF"),
    "VTV": ("Vanguard Value", "미국 대형 가치주", "미국 ETF"),
    "SCHG": ("Schwab US Large-Cap Growth", "미국 대형 성장주", "미국 ETF"),
    "VT": ("Vanguard Total World Stock", "전 세계 주식", "미국 ETF"),
    "ACWI": ("iShares MSCI ACWI", "전 세계 주식", "미국 ETF"),
    "VXUS": ("Vanguard Total International", "미국 밖 전 세계 주식", "미국 ETF"),
    "VGK": ("Vanguard FTSE Europe", "유럽 주식", "미국 ETF"),
    "EWJ": ("iShares MSCI Japan", "일본 주식", "미국 ETF"),
    "EWY": ("iShares MSCI South Korea", "한국 주식 (달러로)", "미국 ETF"),
    "INDA": ("iShares MSCI India", "인도 주식", "미국 ETF"),
    "FXI": ("iShares China Large-Cap", "중국 대형주", "미국 ETF"),
    "KWEB": ("KraneShares China Internet", "중국 인터넷 기업", "미국 ETF"),
    "EWZ": ("iShares MSCI Brazil", "브라질 주식", "미국 ETF"),
    # 미국 ETF: 배당
    "JEPI": ("JPMorgan Equity Premium Income", "월배당 커버드콜 (S&P500)", "미국 ETF"),
    "JEPQ": ("JPMorgan Nasdaq Equity Premium Income", "월배당 커버드콜 (나스닥)", "미국 ETF"),
    "QYLD": ("Global X NASDAQ 100 Covered Call", "나스닥100 커버드콜, 월배당", "미국 ETF"),
    "VIG": ("Vanguard Dividend Appreciation", "배당을 꾸준히 늘려 온 기업", "미국 ETF"),
    "DGRO": ("iShares Core Dividend Growth", "배당 성장주", "미국 ETF"),
    "VYM": ("Vanguard High Dividend Yield", "고배당주", "미국 ETF"),
    "HDV": ("iShares Core High Dividend", "고배당주", "미국 ETF"),
    "SPYD": ("SPDR S&P 500 High Dividend", "S&P500 고배당 80종목", "미국 ETF"),
    # 미국 ETF: 업종·테마
    "XLK": ("Technology Select Sector SPDR", "미국 기술주", "미국 ETF"),
    "XLF": ("Financial Select Sector SPDR", "미국 금융주", "미국 ETF"),
    "XLE": ("Energy Select Sector SPDR", "미국 에너지주", "미국 ETF"),
    "XLV": ("Health Care Select Sector SPDR", "미국 헬스케어주", "미국 ETF"),
    "XLY": ("Consumer Discretionary Select Sector SPDR", "미국 경기소비재", "미국 ETF"),
    "XLP": ("Consumer Staples Select Sector SPDR", "미국 필수소비재", "미국 ETF"),
    "XLI": ("Industrial Select Sector SPDR", "미국 산업재", "미국 ETF"),
    "XLU": ("Utilities Select Sector SPDR", "미국 유틸리티", "미국 ETF"),
    "XLB": ("Materials Select Sector SPDR", "미국 소재", "미국 ETF"),
    "XLRE": ("Real Estate Select Sector SPDR", "미국 부동산", "미국 ETF"),
    "XLC": ("Communication Services Select Sector SPDR", "미국 커뮤니케이션", "미국 ETF"),
    "SMH": ("VanEck Semiconductor", "반도체 기업", "미국 ETF"),
    "ARKK": ("ARK Innovation", "혁신 성장 기업 (캐시 우드)", "미국 ETF"),
    "ICLN": ("iShares Global Clean Energy", "친환경 에너지", "미국 ETF"),
    "TAN": ("Invesco Solar", "태양광", "미국 ETF"),
    "LIT": ("Global X Lithium & Battery", "리튬·2차전지", "미국 ETF"),
    "URA": ("Global X Uranium", "우라늄·원자력", "미국 ETF"),
    "ITA": ("iShares US Aerospace & Defense", "항공우주·방산", "미국 ETF"),
    "IBIT": ("iShares Bitcoin Trust", "비트코인 현물", "미국 ETF"),
    "BITO": ("ProShares Bitcoin Strategy", "비트코인 선물", "미국 ETF"),
    # 미국 ETF: 채권·현금
    "AGG": ("iShares Core US Aggregate Bond", "미국 채권 시장 전체", "미국 ETF"),
    "BIL": ("SPDR 1-3 Month T-Bill", "초단기 국채, 현금 대용", "미국 ETF"),
    "SGOV": ("iShares 0-3 Month Treasury", "초단기 국채, 현금 대용", "미국 ETF"),
    "TIP": ("iShares TIPS Bond", "물가연동 국채", "미국 ETF"),
    "LQD": ("iShares Investment Grade Corporate", "우량 회사채", "미국 ETF"),
    "HYG": ("iShares High Yield Corporate", "하이일드(고위험) 회사채", "미국 ETF"),
    "EDV": ("Vanguard Extended Duration Treasury", "초장기 국채 (20~30년)", "미국 ETF"),
    "VGLT": ("Vanguard Long-Term Treasury", "장기 국채", "미국 ETF"),
    "BNDX": ("Vanguard Total International Bond", "미국 밖 채권 (환헤지)", "미국 ETF"),
    "EMB": ("iShares JP Morgan EM Bond", "신흥국 달러 채권", "미국 ETF"),
    # 미국 ETF: 원자재
    "IAU": ("iShares Gold Trust", "금, GLD보다 보수가 낮음", "미국 ETF"),
    "GLDM": ("SPDR Gold MiniShares", "금, 보수가 아주 낮음", "미국 ETF"),
    "SLV": ("iShares Silver Trust", "은", "미국 ETF"),
    "USO": ("United States Oil Fund", "원유 (선물)", "미국 ETF"),
    "PDBC": ("Invesco Optimum Yield Commodity", "원자재", "미국 ETF"),
    # 미국 ETF: 레버리지·인버스 (위험 큼)
    "QLD": ("ProShares Ultra QQQ", "나스닥100 하루 수익률의 2배 (레버리지)", "미국 ETF"),
    "SSO": ("ProShares Ultra S&P500", "S&P500 하루 수익률의 2배 (레버리지)", "미국 ETF"),
    "UPRO": ("ProShares UltraPro S&P500", "S&P500 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "SPXL": ("Direxion Daily S&P 500 Bull 3X", "S&P500 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "SOXL": ("Direxion Daily Semiconductor Bull 3X", "반도체 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "TECL": ("Direxion Daily Technology Bull 3X", "기술주 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "TNA": ("Direxion Daily Small Cap Bull 3X", "소형주 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "TMF": ("Direxion Daily 20+ Year Treasury Bull 3X", "장기 국채 하루 수익률의 3배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "TSLL": ("Direxion Daily TSLA Bull 2X", "테슬라 하루 수익률의 2배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "NVDL": ("GraniteShares 2x Long NVDA", "엔비디아 하루 수익률의 2배 (레버리지, 위험 매우 큼)", "미국 ETF"),
    "SQQQ": ("ProShares UltraPro Short QQQ", "나스닥100 하루 수익률의 −3배 (인버스, 위험 매우 큼)", "미국 ETF"),
    "SOXS": ("Direxion Daily Semiconductor Bear 3X", "반도체 하루 수익률의 −3배 (인버스, 위험 매우 큼)", "미국 ETF"),
    "SH": ("ProShares Short S&P500", "S&P500 하루 수익률의 −1배 (인버스)", "미국 ETF"),
    "PSQ": ("ProShares Short QQQ", "나스닥100 하루 수익률의 −1배 (인버스)", "미국 ETF"),
    # S&P500 밖의 인기 미국 상장 주식
    "TSM": ("TSMC", "대만 반도체 위탁생산 (미국 상장)", "미국 주식"),
    "ASML": ("ASML", "네덜란드 반도체 장비 (미국 상장)", "미국 주식"),
    "ARM": ("Arm Holdings", "반도체 설계", "미국 주식"),
    "BABA": ("Alibaba", "중국 전자상거래 (미국 상장)", "미국 주식"),
    "PDD": ("PDD Holdings", "테무·핀둬둬 (미국 상장)", "미국 주식"),
    "MELI": ("MercadoLibre", "중남미 전자상거래", "미국 주식"),
    "SHOP": ("Shopify", "온라인 쇼핑몰 플랫폼", "미국 주식"),
    "NVO": ("Novo Nordisk", "덴마크 제약 (미국 상장)", "미국 주식"),
    "SONY": ("Sony Group", "일본 전자·게임 (미국 상장)", "미국 주식"),
    "TM": ("Toyota Motor", "일본 자동차 (미국 상장)", "미국 주식"),
    "CPNG": ("Coupang", "쿠팡 (미국 상장)", "미국 주식"),
    "MSTR": ("Strategy (MicroStrategy)", "비트코인 보유 기업", "미국 주식"),
    "IONQ": ("IonQ", "양자컴퓨터", "미국 주식"),
    "RGTI": ("Rigetti Computing", "양자컴퓨터", "미국 주식"),
    "RKLB": ("Rocket Lab", "우주 발사체", "미국 주식"),
    "OKLO": ("Oklo", "소형 원자로", "미국 주식"),
    "SMR": ("NuScale Power", "소형 원자로", "미국 주식"),
    "SOFI": ("SoFi Technologies", "핀테크", "미국 주식"),
    "RIVN": ("Rivian", "전기차", "미국 주식"),
    "NIO": ("NIO", "중국 전기차 (미국 상장)", "미국 주식"),
    "SNOW": ("Snowflake", "클라우드 데이터", "미국 주식"),
    "SPOT": ("Spotify", "음악 스트리밍", "미국 주식"),
}

# 검색용 한국어 이름 (미국 종목)
KO_ALIASES: dict[str, str] = {
    "AAPL": "애플", "MSFT": "마이크로소프트 마소", "NVDA": "엔비디아", "GOOGL": "구글 알파벳", "GOOG": "구글 알파벳",
    "AMZN": "아마존", "META": "메타 페이스북", "TSLA": "테슬라", "AVGO": "브로드컴", "BRK-B": "버크셔 해서웨이 워런 버핏",
    "JPM": "JP모건", "V": "비자", "MA": "마스터카드", "LLY": "일라이 릴리", "UNH": "유나이티드헬스", "XOM": "엑슨모빌",
    "CVX": "셰브론", "JNJ": "존슨앤존슨", "PG": "P&G 프록터앤갬블", "KO": "코카콜라", "PEP": "펩시", "WMT": "월마트",
    "COST": "코스트코", "HD": "홈디포", "MCD": "맥도날드", "NKE": "나이키", "SBUX": "스타벅스", "DIS": "디즈니",
    "NFLX": "넷플릭스", "ADBE": "어도비", "CRM": "세일즈포스", "ORCL": "오라클", "AMD": "AMD", "INTC": "인텔",
    "QCOM": "퀄컴", "MU": "마이크론", "TXN": "텍사스 인스트루먼트", "IBM": "IBM", "CSCO": "시스코", "PLTR": "팔란티어",
    "UBER": "우버", "ABNB": "에어비앤비", "PYPL": "페이팔", "BAC": "뱅크오브아메리카", "WFC": "웰스파고",
    "GS": "골드만삭스", "MS": "모건스탠리", "BA": "보잉", "CAT": "캐터필러", "GE": "GE 에어로스페이스",
    "LMT": "록히드마틴", "PFE": "화이자", "MRK": "머크", "ABBV": "애브비", "T": "AT&T", "VZ": "버라이즌",
    "O": "리얼티인컴 월배당", "F": "포드", "GM": "제너럴모터스", "COIN": "코인베이스", "HOOD": "로빈후드",
    "SMCI": "슈퍼마이크로", "DELL": "델", "PM": "필립모리스", "MO": "알트리아", "ANET": "아리스타",
    "TSM": "TSMC 대만반도체", "ASML": "ASML", "ARM": "ARM 암", "BABA": "알리바바", "PDD": "테무 핀둬둬",
    "MELI": "메르카도리브레", "SHOP": "쇼피파이", "NVO": "노보노디스크 위고비", "SONY": "소니", "TM": "토요타",
    "CPNG": "쿠팡", "MSTR": "마이크로스트래티지 스트래티지", "IONQ": "아이온큐", "RGTI": "리게티", "RKLB": "로켓랩",
    "OKLO": "오클로", "SMR": "뉴스케일", "SOFI": "소파이", "RIVN": "리비안", "NIO": "니오", "SNOW": "스노우플레이크",
    "SPOT": "스포티파이", "SPY": "에스앤피 S&P500", "QQQ": "나스닥", "SCHD": "슈드 배당", "JEPI": "제피 배당",
    "JEPQ": "제피큐 배당", "TQQQ": "티큐 레버리지 나스닥 3배", "SOXL": "속슬 반도체 3배", "TLT": "미국채 장기채",
}
