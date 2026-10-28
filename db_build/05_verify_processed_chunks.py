# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# verify_processed_chunks.py
"""
[세무 RAG] 정제 가공 데이터(Processed Chunks) 무결성 및 정합성 전용 대조 검증기

===============================================================================
■ 1. 프로그램 개요 및 도입 목적
===============================================================================
세법 RAG 시스템 구축 시, 원천 데이터(법령, 판례, 심판원, 개정세법)를 파싱·청킹하여
DB에 적재하고 벡터 임베딩을 생성하는 과정은 수십 분 이상의 시간과 막대한 GPU/CPU
자원을 소모합니다.

만약 파싱 로직이나 정규식 규칙이 변경되어 누락이 생겼거나 텍스트가 깨졌다면,
임베딩을 다 돌리고 나서야 문제를 인지하게 되어 시간과 자원을 크게 낭비하게 됩니다.

본 검증기는 DB 적재와 무거운 임베딩 작업을 수행하기 "직전", 파싱/청킹 결과물(JSON)
단계에서 기존 운영 DB 또는 백업 JSON 파일과 1:1로 맞대조하여,
단 1~3초 만에 데이터 누락, ID 변경, 본문 텍스트 변형 여부를 100% 밝혀내는
사전 무결성 검증기(Pre-ingest Integrity Verifier)입니다.

===============================================================================
■ 2. 3가지 핵심 검증 모드 (Execution Modes)
===============================================================================

[모드 1] DB vs 폴더 역대조 (--mode db-dir) ★ 가장 추천하는 방식
  - 대상: 기존 운영 PostgreSQL DB ◀─── 1:1 대조 ───▶ 새로 파싱한 JSON 폴더
  - 용도: 새 DB를 만들지 않고도, 현재 운영 중인 DB의 실제 레코드와
          로컬에 새로 추출한 JSON 파일들의 텍스트/건수/ID가 동일한지 즉시 검증
  - 장점: DB 임베딩 적재(수십 분) 없이 3초 만에 파싱 정상 여부 판정

[모드 2] 폴더 vs 폴더 일괄 대조 (--mode dir)
  - 대상: 이전 가공 폴더 (기준) ◀─── 1:1 대조 ───▶ 신규 가공 폴더 (대조)
  - 용도: 백업해 둔 `./data/02_processed_chunks_backup` 폴더와
          새로 추출한 `./data/02_processed_chunks` 폴더 내 4대 마스터 JSON을 일괄 비교

[모드 3] 파일 vs 파일 단일 대조 (--mode file)
  - 대상: 단일 JSON 파일 1 ◀─── 1:1 대조 ───▶ 단일 JSON 파일 2
  - 용도: 특정 파일(예: laws_old.json vs laws_new.json)만 콕 집어서
          ID 및 본문(full_text) 텍스트를 정밀 대조

===============================================================================
■ 3. 전체 CLI 옵션 파라미터 상세 안내
===============================================================================

[기본 제어 옵션]
  --mode {dir,file,db-dir}
      검증 모드를 선택합니다. (기본값: dir)
      - dir    : 폴더 vs 폴더 일괄 대조
      - file   : 단일 파일 1:1 대조
      - db-dir : 기존 DB vs 신규 로컬 JSON 폴더 대조

  --target {all,laws,cases,tribunal,amendments}
      대조할 세법 도메인 대상을 선택합니다. (기본값: all)
      - all        : 4개 마스터 전체 대조
      - laws       : 법령 (tax_laws_master / laws_processed.json)
      - cases      : 판례 (tax_precedents_master / cases_processed.json)
      - tribunal   : 심판원 (tax_tribunal_master / tribunal_processed.json)
      - amendments : 개정세법 (tax_amendments_2026 / 2026_tax_amendments_processed.json)

  --verbose
      검증 과정에서 상세 진행 로그 및 정보를 콘솔에 추가 출력합니다.

[모드 1: db-dir 전용 옵션]
  --db-name <DB이름>
      대조 기준으로 삼을 기존 PostgreSQL 데이터베이스 이름을 지정합니다.
      (미지정 시 .env의 TAX_DB_URI에 설정된 기본 DB명을 자동으로 사용합니다)
  --processed-dir <폴더경로>
      새로 파싱한 JSON 파일들이 들어있는 로컬 가공 폴더 경로입니다.
      (기본값: ./data/02_processed_chunks)

[모드 2: dir 전용 옵션]
  --dir1 <폴더경로>
      기준이 되는 원본 가공 폴더 경로입니다. (기본값: ./data/02_processed_chunks_backup)
  --dir2 <폴더경로>
      대조 대상이 되는 신규 가공 폴더 경로입니다. (기본값: ./data/02_processed_chunks)

[모드 3: file 전용 옵션]
  --file1 <파일경로>
      기준이 되는 첫 번째 JSON 파일 경로입니다. (필수)
  --file2 <파일경로>
      대조 대상이 되는 두 번째 JSON 파일 경로입니다. (필수)
  --id-key <키이름>
      레코드 식별자로 사용할 필드 이름입니다. (기본값: id)
  --text-key <키이름>
      본문 내용 비교에 사용할 텍스트 필드 이름입니다. (기본값: full_text)

===============================================================================
■ 4. 실전 사용 예시 (PowerShell 복사 & 실행 명령어)
===============================================================================

[상황 A] 파싱 후, 기존 운영 DB와 3초 만에 무결성 역대조하기 (가장 권장)
  # 1. 먼저 파싱만 수행 (임베딩 X, JSON 파일만 생성)
  python 02_build_tax_db.py --step parse --target all
  python 03_build_tax_amendments.py --step parse

  # 2. 기존 DB(tax_rag_db)와 새로 생성된 JSON 파일들 일괄 역대조
  python verify_processed_chunks.py --mode db-dir --db-name tax_rag_db --processed-dir ./data/02_processed_chunks

  # 3. 법령만 콕 집어서 DB와 대조하고 싶을 때
  python verify_processed_chunks.py --mode db-dir --db-name tax_rag_db --target laws

[상황 B] 백업 폴더와 신규 가공 폴더 통째로 비교하기
  python verify_processed_chunks.py --mode dir --dir1 ./data/02_processed_chunks_backup --dir2 ./data/02_processed_chunks

[상황 C] 두 개의 특정 JSON 파일 1:1 정밀 대조하기
  python verify_processed_chunks.py --mode file --file1 ./backup/laws.json --file2 ./data/02_processed_chunks/laws_processed.json

===============================================================================
■ 5. 검증 리포트 해석 방법
===============================================================================
스크립트 실행 시 각 대상별로 아래 3단계 검증 결과가 출력됩니다:

 1. 데이터 총 건수:
    - 기준 건수와 대조 건수가 동일한지 비교 ([PASS] / [FAIL])
 2. ID 집합 정합성:
    - 공통 ID 건수 산출
    - 기준에는 있지만 대조군에서 사라진 [누락 ID] 목록 표시
    - 대조군에 새로 생긴 [신규 추가 ID] 목록 표시
 3. 본문 내용(full_text) 무결성:
    - 공백/개행 정규화 후 MD5 해시를 대조하여 텍스트 내용의 100% 일치 확인
    - 불일치 발생 시 상위 5건의 ID와 텍스트 프리뷰를 즉시 출력하여 원인 파악 지원
 4. 종료 코드(Exit Code):
    - 100% 완벽 일치 시: exit code 0 (자동화 파이프라인 연동 가능)
    - 불일치/에러 발생 시: exit code 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# 매핑 및 설정 상수 (세법 도메인 4대 마스터 테이블)
# -----------------------------------------------------------------------------
DEFAULT_TARGET_MAPPING = {
    "laws": {
        "file_patterns": ["laws_processed.json", "law_processed.json", "*law*.json"],
        "table_name": "tax_laws_master",
        "id_col": "id",
        "text_col": "full_text",
        "extra_cols": ["law_name", "article_number", "article_title"],
    },
    "cases": {
        "file_patterns": ["cases_processed.json", "case_processed.json", "*case*.json"],
        "table_name": "tax_precedents_master",
        "id_col": "id",
        "text_col": "full_text",
        "extra_cols": ["case_name", "case_number", "summary"],
    },
    "tribunal": {
        "file_patterns": ["tribunal_processed.json", "*tribunal*.json"],
        "table_name": "tax_tribunal_master",
        "id_col": "id",
        "text_col": "full_text",
        "extra_cols": ["case_name", "case_number", "tax_item", "summary"],
    },
    "amendments": {
        "file_patterns": [
            "2026_tax_amendments_processed.json",
            "2026_tax_processed.json",
            "amendments_processed.json",
            "*amend*.json",
            "*개정세법*.json",
            "*2026*.json",
        ],
        "table_name": "tax_amendments_2026",
        "id_col": "id",
        "text_col": "full_text",
        "extra_cols": ["statute_area", "item_no", "title"],
    },
}

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")


def _safe_identifier(name: str, label: str) -> str:
    """SQL 인젝션 방지를 위한 식별자 유효성 검증."""
    if not _IDENT_RE.match(name):
        raise SystemExit(
            f"[FATAL ERROR] {label} 값이 안전하지 않습니다: {name!r} "
            f"(영문/숫자/밑줄만 허용, 63자 이내, 숫자로 시작 불가)"
        )
    return name


def _compute_md5(text: str) -> str:
    """공백 정규화 후 MD5 해시 생성 (개행/공백 차이로 인한 거짓 경보 방지)."""
    norm_text = "\n".join(line.rstrip() for line in (text or "").strip().splitlines())
    return hashlib.md5(norm_text.encode("utf-8")).hexdigest()


# -----------------------------------------------------------------------------
# 결과 수집 데이터 모델
# -----------------------------------------------------------------------------
@dataclass
class ComparisonResult:
    target_name: str
    src1_desc: str
    src2_desc: str
    count1: int = 0
    count2: int = 0
    common_count: int = 0
    missing_in_2: List[str] = field(default_factory=list)
    added_in_2: List[str] = field(default_factory=list)
    content_mismatch_count: int = 0
    mismatched_samples: List[Dict[str, Any]] = field(default_factory=list)
    is_success: bool = True
    error_message: Optional[str] = None


# -----------------------------------------------------------------------------
# 1. 파일 기반 데이터 로더 및 비교 로직
# -----------------------------------------------------------------------------
def load_json_records(filepath: str) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """JSON 파일을 읽어 ID 기반 딕셔너리로 반환한다."""
    if not os.path.exists(filepath):
        return {}, f"파일을 찾을 수 없습니다: {filepath}"

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {}, f"JSON 파싱 실패: {filepath} ({e})"

    if not isinstance(data, list):
        return {}, f"JSON 루트가 리스트([]) 형식이 아닙니다: {filepath}"

    records: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        doc_id = str(item.get("id") or item.get("doc_id") or "").strip()
        if not doc_id:
            doc_id = f"__NO_ID_ROW_{idx}__"
        records[doc_id] = item

    return records, None


def compare_records(
    target_name: str,
    src1_desc: str,
    src2_desc: str,
    records1: Dict[str, Dict[str, Any]],
    records2: Dict[str, Dict[str, Any]],
    text_key: str = "full_text",
    max_diff_samples: int = 5,
) -> ComparisonResult:
    """두 딕셔너리 레코드 집합 간의 정밀 대조를 수행한다."""
    res = ComparisonResult(
        target_name=target_name,
        src1_desc=src1_desc,
        src2_desc=src2_desc,
        count1=len(records1),
        count2=len(records2),
    )

    ids1: Set[str] = set(records1.keys())
    ids2: Set[str] = set(records2.keys())

    res.common_count = len(ids1 & ids2)
    res.missing_in_2 = sorted(list(ids1 - ids2))
    res.added_in_2 = sorted(list(ids2 - ids1))

    # 본문 및 주요 필드 대조
    mismatches = []
    for doc_id in ids1 & ids2:
        r1 = records1[doc_id]
        r2 = records2[doc_id]

        t1 = str(r1.get(text_key) or "")
        t2 = str(r2.get(text_key) or "")

        h1 = _compute_md5(t1)
        h2 = _compute_md5(t2)

        if h1 != h2:
            mismatches.append(
                {
                    "id": doc_id,
                    "len1": len(t1),
                    "len2": len(t2),
                    "preview1": t1[:100].replace("\n", " "),
                    "preview2": t2[:100].replace("\n", " "),
                }
            )

    res.content_mismatch_count = len(mismatches)
    res.mismatched_samples = mismatches[:max_diff_samples]

    if res.count1 != res.count2 or res.missing_in_2 or res.added_in_2 or res.content_mismatch_count > 0:
        res.is_success = False

    return res


# -----------------------------------------------------------------------------
# 2. DB 기반 데이터 로더
# -----------------------------------------------------------------------------
def load_db_records(
    db_uri: str, table_name: str, id_col: str = "id", text_col: str = "full_text"
) -> Tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """PostgreSQL 테이블에서 ID와 본문 텍스트를 읽어온다."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except ImportError:
        return {}, "psycopg2 패키지가 설치되어 있지 않습니다."

    try:
        conn = psycopg2.connect(db_uri)
    except Exception as e:
        return {}, f"DB 접속 실패 ({db_uri}): {e}"

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT to_regclass(%s);", (table_name,))
            if not cur.fetchone()["to_regclass"]:
                return {}, f"DB에 '{table_name}' 테이블이 존재하지 않습니다."

            query = f'SELECT "{id_col}", "{text_col}" FROM "{table_name}";'
            cur.execute(query)
            rows = cur.fetchall()

            records: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                k = str(r[id_col])
                records[k] = {id_col: k, text_col: r[text_col]}
            return records, None
    except Exception as e:
        return {}, f"쿼리 실행 실패 ({table_name}): {e}"
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# 3. 파일 매칭 헬퍼
# -----------------------------------------------------------------------------
def find_matched_file(directory: str, patterns: List[str]) -> Optional[str]:
    """지정된 디렉토리에서 패턴 목록에 맞는 첫 번째 파일을 탐색한다."""
    if not os.path.exists(directory):
        return None

    import fnmatch

    all_files = os.listdir(directory)
    for pat in patterns:
        for f in all_files:
            if fnmatch.fnmatch(f, pat):
                return os.path.join(directory, f)
    return None


# -----------------------------------------------------------------------------
# 4. 콘솔 리포트 렌더러
# -----------------------------------------------------------------------------
def print_comparison_result(res: ComparisonResult, verbose: bool = False) -> None:
    """비교 결과를 표준 출력으로 상세하게 출력한다."""
    print("\n" + "=" * 78)
    print(f" [검증 대상] {res.target_name.upper()}")
    print(f" - 원본 1 (기준) : {res.src1_desc}")
    print(f" - 원본 2 (대조) : {res.src2_desc}")
    print("=" * 78)

    if res.error_message:
        print(f" [ERROR] 검증 중단: {res.error_message}")
        print(" [결과] [FAIL] 검증 실패")
        return

    # 1. 행 수 비교
    match_count_str = "[PASS] 건수 일치" if res.count1 == res.count2 else "[FAIL] 건수 불일치"
    print(f" 1. 데이터 총 건수:")
    print(f"    - 기준: {res.count1:,} 건")
    print(f"    - 대조: {res.count2:,} 건  -->  {match_count_str}")

    # 2. 고유 식별자(ID) 집합 비교
    id_status = "[PASS] 모든 ID 일치" if not res.missing_in_2 and not res.added_in_2 else "[FAIL] ID 차이 발생"
    print(f" 2. ID 집합 정합성:")
    print(f"    - 상호 일치 ID 건수 : {res.common_count:,} 건  -->  {id_status}")

    if res.missing_in_2:
        print(f"    - 기준에는 있으나 대조군에서 [누락된 ID] ({len(res.missing_in_2):,}건):")
        sample = res.missing_in_2[:10]
        print(f"      예시: {sample}" + (" ..." if len(res.missing_in_2) > 10 else ""))

    if res.added_in_2:
        print(f"    - 대조군에만 [새로 추가된 ID] ({len(res.added_in_2):,}건):")
        sample = res.added_in_2[:10]
        print(f"      예시: {sample}" + (" ..." if len(res.added_in_2) > 10 else ""))

    # 3. 본문 내용 MD5 대조
    content_status = (
        "[PASS] 100% 동일 (MD5 일치)"
        if res.content_mismatch_count == 0
        else f"[FAIL] 본문 불일치 {res.content_mismatch_count:,}건"
    )
    print(f" 3. 본문 내용(full_text) 무결성:")
    print(f"    - 공통 ID 본문 대조 결과 : {content_status}")

    if res.mismatched_samples:
        print(f"    - 본문 불일치 상세 샘플 (상위 {len(res.mismatched_samples)}건):")
        for idx, smp in enumerate(res.mismatched_samples, 1):
            print(f"      [{idx}] ID: {smp['id']}")
            print(f"          기준 본문(길이 {smp['len1']}): {smp['preview1']}...")
            print(f"          대조 본문(길이 {smp['len2']}): {smp['preview2']}...")

    # 4. 최종 판정
    print("-" * 78)
    if res.is_success:
        print(f" [최종 판정] [PERFECT MATCH] 두 데이터 세트가 100% 완벽히 일치합니다.")
    else:
        print(f" [최종 판정] [MISMATCH DETECTED] 데이터 차이가 발견되었습니다. 적재 전 확인 필요!")


# -----------------------------------------------------------------------------
# 5. CLI 메인 실행기
# -----------------------------------------------------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="[세무 RAG] 정제 가공 데이터(Processed Chunks) 무결성 대조 검증기",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        choices=["dir", "file", "db-dir"],
        default="dir",
        help="검증 모드 선택:\n"
        "  dir    : 두 개의 가공 폴더 내 4개 마스터 JSON 일괄 대조 (기본값)\n"
        "  file   : 두 개의 지정된 JSON 파일 1:1 대조\n"
        "  db-dir : 기존 DB의 실제 데이터와 로컬 가공 JSON 폴더 1:1 대조",
    )

    # 폴더 vs 폴더 모드 인자
    parser.add_argument("--dir1", default="./data/02_processed_chunks_backup", help="기준 가공 폴더 경로")
    parser.add_argument("--dir2", default="./data/02_processed_chunks", help="대조 대상 신규 가공 폴더 경로")

    # 파일 vs 파일 모드 인자
    parser.add_argument("--file1", help="기준 단일 JSON 파일 경로")
    parser.add_argument("--file2", help="대조 대상 단일 JSON 파일 경로")
    parser.add_argument("--id-key", default="id", help="레코드 식별자 키 이름 (기본: id)")
    parser.add_argument("--text-key", default="full_text", help="본문 텍스트 키 이름 (기본: full_text)")

    # DB vs 폴더 모드 인자
    parser.add_argument("--db-name", help="대조할 기존 DB 이름 (미지정시 .env의 TAX_DB_URI 기본 DB 사용)")
    parser.add_argument("--processed-dir", default="./data/02_processed_chunks", help="검증할 신규 JSON 가공 폴더 경로")

    # 필터 및 공통 제어
    parser.add_argument(
        "--target",
        choices=["all", "laws", "cases", "tribunal", "amendments"],
        default="all",
        help="검증 대상 (all, laws, cases, tribunal, amendments)",
    )
    parser.add_argument("--verbose", action="store_true", help="상세 로그 출력")

    return parser.parse_args()


def resolve_db_uri(db_name: Optional[str]) -> str:
    """환경변수에서 DB 접속 URI를 획득하고 필요시 db-name을 교체한다."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    base_uri = os.getenv("TAX_DB_URI") or os.getenv("DB_URI")
    if not base_uri:
        raise SystemExit("[FATAL ERROR] .env 파일에 TAX_DB_URI 또는 DB_URI가 설정되어 있지 않습니다.")

    if db_name:
        _safe_identifier(db_name, "--db-name")
        return re.sub(r"/[^/]+$", f"/{db_name}", base_uri)
    return base_uri


def main():
    args = parse_arguments()
    all_success = True

    print("=" * 78)
    print(" [세무 RAG] 가공 데이터(Processed Chunks) 무결성 대조 검증기 가동")
    print(f" ▶ 실행 모드: {args.mode.upper()}")
    print("=" * 78)

    targets_to_run = (
        ["laws", "cases", "tribunal", "amendments"] if args.target == "all" else [args.target]
    )

    # -------------------------------------------------------------------------
    # CASE 1: 파일 vs 파일 (단일 1:1 대조)
    # -------------------------------------------------------------------------
    if args.mode == "file":
        if not args.file1 or not args.file2:
            print("[FATAL ERROR] --mode file 실행 시 --file1 과 --file2 가 모두 필요합니다.")
            sys.exit(1)

        print(f" - 기준 파일: {args.file1}")
        print(f" - 대조 파일: {args.file2}")

        rec1, err1 = load_json_records(args.file1)
        rec2, err2 = load_json_records(args.file2)

        if err1 or err2:
            print(f"[ERROR] {err1 or err2}")
            sys.exit(1)

        res = compare_records(
            target_name="단일 파일 대조",
            src1_desc=args.file1,
            src2_desc=args.file2,
            records1=rec1,
            records2=rec2,
            text_key=args.text_key,
        )
        print_comparison_result(res, verbose=args.verbose)
        if not res.is_success:
            all_success = False

    # -------------------------------------------------------------------------
    # CASE 2: 폴더 vs 폴더 (4대 마스터 일괄 대조)
    # -------------------------------------------------------------------------
    elif args.mode == "dir":
        print(f" - 기준 폴더: {args.dir1}")
        print(f" - 대조 폴더: {args.dir2}")

        if not os.path.exists(args.dir1):
            print(f"[FATAL ERROR] 기준 폴더가 존재하지 않습니다: {args.dir1}")
            sys.exit(1)
        if not os.path.exists(args.dir2):
            print(f"[FATAL ERROR] 대조 폴더가 존재하지 않습니다: {args.dir2}")
            sys.exit(1)

        for target in targets_to_run:
            info = DEFAULT_TARGET_MAPPING[target]
            f1 = find_matched_file(args.dir1, info["file_patterns"])
            f2 = find_matched_file(args.dir2, info["file_patterns"])

            if not f1 or not f2:
                print(f"\n[-] {target.upper()} 스킵: 양쪽 폴더 중 파일이 없는 곳이 있습니다.")
                if not f1:
                    print(f"    (기준 폴더 {args.dir1}에 일치하는 파일 없음)")
                if not f2:
                    print(f"    (대조 폴더 {args.dir2}에 일치하는 파일 없음)")
                all_success = False
                continue

            rec1, err1 = load_json_records(f1)
            rec2, err2 = load_json_records(f2)

            if err1 or err2:
                print(f"\n[ERROR] {target.upper()} 파일 읽기 실패: {err1 or err2}")
                all_success = False
                continue

            res = compare_records(
                target_name=target,
                src1_desc=os.path.basename(f1),
                src2_desc=os.path.basename(f2),
                records1=rec1,
                records2=rec2,
                text_key=info["text_col"],
            )
            print_comparison_result(res, verbose=args.verbose)
            if not res.is_success:
                all_success = False

    # -------------------------------------------------------------------------
    # CASE 3: 기존 DB vs 신규 로컬 JSON 폴더 (임베딩 전 원천 역대조)
    # -------------------------------------------------------------------------
    elif args.mode == "db-dir":
        db_uri = resolve_db_uri(args.db_name)
        active_db = db_uri.split("/")[-1]
        print(f" - 기준 대상 (DB)   : {active_db}")
        print(f" - 대조 대상 (폴더) : {args.processed_dir}")

        if not os.path.exists(args.processed_dir):
            print(f"[FATAL ERROR] 로컬 가공 폴더가 존재하지 않습니다: {args.processed_dir}")
            sys.exit(1)

        for target in targets_to_run:
            info = DEFAULT_TARGET_MAPPING[target]
            table = info["table_name"]
            fpath = find_matched_file(args.processed_dir, info["file_patterns"])

            if not fpath:
                print(f"\n[-] {target.upper()} 스킵: 폴더 내 대응 파일 없음 (패턴: {info['file_patterns']})")
                all_success = False
                continue

            # 1. DB에서 원문 로드
            db_records, db_err = load_db_records(
                db_uri=db_uri,
                table_name=table,
                id_col=info["id_col"],
                text_col=info["text_col"],
            )
            if db_err:
                print(f"\n[!] {target.upper()} DB 조회 스킵: {db_err}")
                all_success = False
                continue

            # 2. 로컬 JSON 로드
            file_records, file_err = load_json_records(fpath)
            if file_err:
                print(f"\n[ERROR] {target.upper()} JSON 읽기 실패: {file_err}")
                all_success = False
                continue

            # 3. 대조 수행
            res = compare_records(
                target_name=f"{target} (DB: {table} vs 파일: {os.path.basename(fpath)})",
                src1_desc=f"DB 테이블 [{table}]",
                src2_desc=f"로컬 파일 [{os.path.basename(fpath)}]",
                records1=db_records,
                records2=file_records,
                text_key=info["text_col"],
            )
            print_comparison_result(res, verbose=args.verbose)
            if not res.is_success:
                all_success = False

    # -------------------------------------------------------------------------
    # 종합 결과 반환
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    if all_success:
        print(" [전체 요약] 모든 검증 항목이 완벽하게 일치합니다. (무결성 100% 입증 완료)")
        sys.exit(0)
    else:
        print(" [전체 요약] 일부 검증 항목에서 차이 또는 불일치가 발견되었습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
