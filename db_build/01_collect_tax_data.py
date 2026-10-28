# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# =====================================================================
# 마스터 원천 데이터 수집기 (01_collect_data.py)
# =====================================================================
# [설계 및 개편 히스토리]
# 1. 수집기 통합: 기존 tax_master_pipeline법률정보.py[cite: 3]와 tax_master_pipeline판결정보.py[cite: 4]로 분리되어 있던 로직을 하나로 합쳤습니다.
# 2. 명칭 및 경로 표준화: 실무적 직관성을 위해 'precedents'를 'cases'로, 'tax_tribunal'을 'tribunal'로 통일하고 입출력 경로를 고정했습니다.
# 3. 옵션 제어: --target, --limit, --query 옵션을 통해 테스트와 실무 수집의 유연성을 확보했습니다.
# 4. 심판원 버그 원천 방어: 수집 단계에서 ID가 유실되지 않도록 파일명에 강제로 ID를 박아 넣는 로직을 강화했습니다[cite: 4].
# =====================================================================

import os
import re
import json
import time
import argparse
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()
BASE_URL = "https://www.law.go.kr"
OC_KEY = os.getenv("LAW_OC_KEY")
assert OC_KEY, "[FATAL ERROR] LAW_OC_KEY 환경변수가 필요합니다."

# 고정된 입출력 경로 (절대 변경 불가 원칙)
DIR_RAW_LAWS = "./data/01_raw_data/laws"
DIR_RAW_CASES = "./data/01_raw_data/cases"
DIR_RAW_TRIBUNAL = "./data/01_raw_data/tribunal"

for d in [DIR_RAW_LAWS, DIR_RAW_CASES, DIR_RAW_TRIBUNAL]:
    os.makedirs(d, exist_ok=True)

DISPLAY = 100
REQUEST_SLEEP = 0.35

TARGET_TAX_LAWS = [
    "소득세법", "소득세법 시행령", "법인세법", "법인세법 시행령", "부가가치세법", "부가가치세법 시행령",
    "상속세 및 증여세법", "조세특례제한법", "종합부동산세법", "지방세법"
]
TAX_KEYWORDS = ["소득세", "법인세", "부가가치세", "상속세", "조세특례", "종합부동산세", "지방세"]

def fetch_json(url, retries=3):
    for i in range(retries):
        try:
            res = requests.get(url, timeout=30)
            if res.status_code == 200:
                return res.json()
        except Exception as e:
            time.sleep(1.0 + i)
    return None

def _safe_name(s: str) -> str:
    return re.sub(r"[^\w\s\-\[\]()]", "", str(s or "")).strip() or "unknown"

# 1. 법령 수집 로직[cite: 3]
def collect_laws():
    print("\n[수집 시작] 법령 데이터 (laws)")
    for law_name in TARGET_TAX_LAWS:
        search_url = f"{BASE_URL}/DRF/lawSearch.do?OC={OC_KEY}&target=law&query={quote(law_name)}&type=JSON"
        data = fetch_json(search_url)
        if not data or 'LawSearch' not in data or 'law' not in data['LawSearch']: continue
        
        target_item = data['LawSearch']['law'][0] if isinstance(data['LawSearch']['law'], list) else data['LawSearch']['law']
        mst_id = target_item['법령일련번호']
        
        body_url = f"{BASE_URL}/DRF/lawService.do?OC={OC_KEY}&target=law&MST={mst_id}&type=JSON"
        body_data = fetch_json(body_url)
        
        if body_data:
            path = os.path.join(DIR_RAW_LAWS, f"law_{law_name}_raw.json")
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(body_data, f, ensure_ascii=False, indent=2)
            print(f"  └ 💾 {law_name} 저장 완료")

# 2. 판례 수집 로직[cite: 4]
def collect_cases(limit_pages):
    print("\n[수집 시작] 대법원 판례 데이터 (cases)")
    for kw in TAX_KEYWORDS:
        for page in range(1, limit_pages + 1):
            url = f"{BASE_URL}/DRF/lawSearch.do?OC={OC_KEY}&target=prec&type=JSON&display={DISPLAY}&page={page}&search=2&query={quote(kw)}"
            data = fetch_json(url)
            time.sleep(REQUEST_SLEEP)
            if not data or 'PrecSearch' not in data or 'prec' not in data['PrecSearch']: break
            
            items = data['PrecSearch']['prec']
            items = items if isinstance(items, list) else [items]
            
            for item in items:
                prec_id = str(item.get("판례일련번호", ""))
                if not prec_id: continue
                
                body = fetch_json(f"{BASE_URL}/DRF/lawService.do?OC={OC_KEY}&target=prec&ID={prec_id}&type=JSON")
                if body:
                    case_num = item.get("사건번호", prec_id)
                    court = item.get("법원명", "법원")
                    path = os.path.join(DIR_RAW_CASES, f"case_{_safe_name(court)}_{_safe_name(case_num)}_ID{prec_id}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"meta": item, "body": body}, f, ensure_ascii=False, indent=2)
            print(f"  └ 💾 키워드 '{kw}' {page}페이지 수집 완료")

# 3. 조세심판원 수집 로직[cite: 4]
def collect_tribunal(limit_pages):
    print("\n[수집 시작] 조세심판원 재결례 (tribunal)")
    for kw in TAX_KEYWORDS:
        for page in range(1, limit_pages + 1):
            url = f"{BASE_URL}/DRF/lawSearch.do?OC={OC_KEY}&target=ttSpecialDecc&type=JSON&display={DISPLAY}&page={page}&search=2&query={quote(kw)}"
            data = fetch_json(url)
            time.sleep(REQUEST_SLEEP)
            if not data: break
            
            # API 응답 구조 편차 방어 로직[cite: 4]
            root = data.get("TtSpecialDeccSearch") or data.get("LawSearch") or data
            items = root.get("ttSpecialDecc") or root.get("specialDecc") or []
            items = items if isinstance(items, list) else [items]
            if not items: break
            
            for item in items:
                tid = str(item.get("일련번호") or item.get("특별행정심판재결례일련번호", ""))
                if not tid: continue
                
                body = fetch_json(f"{BASE_URL}/DRF/lawService.do?OC={OC_KEY}&target=ttSpecialDecc&ID={tid}&type=JSON")
                if body:
                    case_no = item.get("청구번호", tid)
                    # 수집 시 ID 결함 방어를 위해 파일명에 명시[cite: 4]
                    path = os.path.join(DIR_RAW_TRIBUNAL, f"tribunal_{_safe_name(case_no)}_ID{tid}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"meta": item, "body": body}, f, ensure_ascii=False, indent=2)
            print(f"  └ 💾 키워드 '{kw}' {page}페이지 수집 완료")

def main():
    parser = argparse.ArgumentParser(description="세무 RAG 원천 데이터 마스터 수집기")
    parser.add_argument("--target", choices=["laws", "cases", "tribunal", "all"], default="all", help="수집 대상 지정")
    parser.add_argument("--limit", type=int, default=3, help="검색어당 최대 수집 페이지 수 (기본 3페이지)")
    args = parser.parse_args()

    print("==================================================")
    print("🚀 [세무 RAG] 마스터 원천 데이터 수집 가동")
    print("==================================================")
    
    if args.target in ["laws", "all"]: collect_laws()
    if args.target in ["cases", "all"]: collect_cases(args.limit)
    if args.target in ["tribunal", "all"]: collect_tribunal(args.limit)
    
    print("\n🎉 수집 파이프라인 정상 종료")

if __name__ == "__main__":
    main()