# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
# =====================================================================
# 마스터 DB 대조 검증기 (verify_db_migration.py) - 세법(Tax) 전용
# =====================================================================
import argparse
import re
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

def _safe_identifier(name, label):
    """f-string으로 SQL에 직접 삽입되는 테이블명이 안전한 식별자인지 검증한다."""
    if not _IDENT_RE.match(name):
        raise SystemExit(f"[FATAL ERROR] {label} 값이 안전하지 않습니다: {name!r} "
                          f"(영문/숫자/밑줄만 허용, 63자 이내, 숫자로 시작 불가)")
    return name

# 기본 타겟 테이블 (세법 도메인)
DEFAULT_TABLES = [
    "tax_laws_master",
    "tax_precedents_master",
    "tax_tribunal_master",
    "tax_amendments_2026"
]

def print_error_and_exit(msg):
    print(f"\n[오류 종료] {msg}")
    sys.exit(1)

def get_connection(dbname, args):
    try:
        return psycopg2.connect(
            dbname=dbname, user=args.user, password=args.password,
            host=args.host, port=args.port
        )
    except psycopg2.OperationalError as e:
        print_error_and_exit(f"'{dbname}' DB에 접속할 수 없습니다.\n(상세: {e})")

def fetch_all(cursor, query, params=None):
    try:
        cursor.execute(query, params)
        return cursor.fetchall()
    except Exception as e:
        print(f"  [쿼리 오류] {e}")  # autocommit 상태이므로 롤백 불필요
        return []

# ---------------------------------------------------------
# 1. 정보 조회 (스키마, 확장, 인덱스)
# ---------------------------------------------------------
def display_db_info(cur_old, cur_new, db_old, db_new, tables):
    print("\n" + "="*70)
    print(" 📊 데이터베이스 스키마 및 환경 비교 정보")
    print("="*70)

    ext_old = {row['extname'] for row in fetch_all(cur_old, "SELECT extname FROM pg_extension;")}
    ext_new = {row['extname'] for row in fetch_all(cur_new, "SELECT extname FROM pg_extension;")}
    
    print(f"\n[확장(Extensions) 설치 현황]")
    print(f"{db_old.ljust(30)} | {db_new}")
    print("-" * 70)
    for ext in sorted(ext_old | ext_new):
        o_mark = "O" if ext in ext_old else "X"
        n_mark = "O" if ext in ext_new else "X"
        print(f" {ext.ljust(28)} | {o_mark.ljust(30)} | {n_mark}")

    print(f"\n[테이블 구조 및 인덱스 비교]")
    for table in tables:
        print(f"\n▶ 테이블: {table}")
        q_cols = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s ORDER BY column_name;"
        cols_old = {r['column_name']: r['data_type'] for r in (fetch_all(cur_old, q_cols, (table,)) or [])}
        cols_new = {r['column_name']: r['data_type'] for r in (fetch_all(cur_new, q_cols, (table,)) or [])}
        
        if not cols_old and not cols_new:
            print("  - 양쪽 DB 모두 테이블이 없습니다.")
            continue
            
        all_cols = sorted(set(cols_old.keys()) | set(cols_new.keys()))
        for col in all_cols:
            o_type = cols_old.get(col, "없음")
            n_type = cols_new.get(col, "없음")
            status = "일치" if o_type == n_type else "불일치!"
            print(f"  {col.ljust(20)} | {db_old}: {o_type.ljust(15)} | {db_new}: {n_type.ljust(15)} | {status}")

# ---------------------------------------------------------
# 2. 데이터 비교 (볼륨 / 내용)
# ---------------------------------------------------------
def compare_table_data(cur_old, cur_new, table, mode):
    print("\n" + "="*70)
    print(f" 🔍 데이터 비교: {table} (모드: {mode})")
    print("="*70)

    cur_old.execute("SELECT to_regclass(%s);", (table,))
    if not cur_old.fetchone()['to_regclass']:
        print(f"[-] {table}: 기존 DB에 테이블이 없습니다. 건너뜁니다.")
        return
    cur_new.execute("SELECT to_regclass(%s);", (table,))
    if not cur_new.fetchone()['to_regclass']:
        print(f"[-] {table}: 신규 DB에 테이블이 없습니다. 건너뜁니다.")
        return

    # 행 갯수 비교
    cnt_old = fetch_all(cur_old, f"SELECT COUNT(*) as c FROM {table}")[0]['c']
    cnt_new = fetch_all(cur_new, f"SELECT COUNT(*) as c FROM {table}")[0]['c']
    print(f"[행 총합] 기존: {cnt_old:,} 건  |  신규: {cnt_new:,} 건  ➔  {'✅ 일치' if cnt_old == cnt_new else '❌ 불일치'}")

    # 임베딩 갯수 비교
    try:
        emb_old = fetch_all(cur_old, f"SELECT COUNT(*) as c FROM {table} WHERE embedding IS NOT NULL")[0]['c']
        emb_new = fetch_all(cur_new, f"SELECT COUNT(*) as c FROM {table} WHERE embedding IS NOT NULL")[0]['c']
        print(f"[임베딩] 기존: {emb_old:,} 건  |  신규: {emb_new:,} 건  ➔  {'✅ 일치' if emb_old == emb_new else '❌ 불일치'}")
    except:
        pass

    if mode == "content":
        # 💡 업그레이드 포인트: _cmp_ 테이블 자동 감지 및 분기 처리
        is_cmp_table = "_cmp_" in table
        id_col = "doc_id" if is_cmp_table else "id"
        
        q_ids = f"SELECT {id_col} FROM {table}"
        
        # 세법 DB는 조문 번호(TEXT) 기준이므로 int 캐스팅 없이 문자열 그대로 대조
        ids_old = set(r[id_col] for r in fetch_all(cur_old, q_ids))
        ids_new = set(r[id_col] for r in fetch_all(cur_new, q_ids))
        
        if ids_old != ids_new:
            print(f"[ID 비교] ❌ 누락/추가 감지! (기존 전용: {len(ids_old - ids_new)}건, 신규 전용: {len(ids_new - ids_old)}건)")
        else:
            print(f"[ID 비교] ✅ 모든 ID가 정확히 일치합니다.")

        # 텍스트 내용 해시(MD5) 대조 로직 분기
        if not is_cmp_table:
            # 모든 마스터 테이블(법령/판례/심판/개정세법)의 본문 컬럼명은 full_text로 통일되어 있음
            content_col = "full_text"
            
            print("[내용 비교] DB 측 해시 연산 진행 중...")
            q_hash = f"SELECT {id_col}, MD5({content_col}::text) as h FROM {table}"
            hash_old = {r[id_col]: r['h'] for r in fetch_all(cur_old, q_hash)}
            hash_new = {r[id_col]: r['h'] for r in fetch_all(cur_new, q_hash)}
            
            diff_count = sum(1 for i in ids_old & ids_new if hash_old[i] != hash_new[i])
                    
            if diff_count == 0:
                print(f"[내용 비교] ✅ 텍스트 본문({content_col}) 내용 100% 일치")
            else:
                print(f"[내용 비교] ❌ 동일한 ID지만 본문 내용이 다른 레코드: {diff_count}건 발견")
        else:
            print("[내용 비교] 비교 전용 테이블이므로 텍스트 본문(content) 해시 대조는 생략합니다.")

def main():
    parser = argparse.ArgumentParser(description="[세법] 데이터베이스 대조 검증기")
    parser.add_argument("--db-old", required=True, help="기존 데이터베이스 이름")
    parser.add_argument("--db-new", required=True, help="신규 데이터베이스 이름")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--password", default="postgres")
    parser.add_argument("--table", help="특정 테이블만 검사할 경우 지정")
    parser.add_argument("--show-info", action="store_true", help="스키마 정보 출력")
    parser.add_argument("--mode", choices=["volume", "content"], default="volume", help="검증 수준")
    
    args = parser.parse_args()
    tables = [_safe_identifier(args.table, "--table")] if args.table else DEFAULT_TABLES

    print("DB 연결 중 (읽기 전용)...")
    conn_old = get_connection(args.db_old, args)
    conn_new = get_connection(args.db_new, args)
    
    conn_old.autocommit = True
    conn_new.autocommit = True

    cur_old = conn_old.cursor(cursor_factory=RealDictCursor)
    cur_new = conn_new.cursor(cursor_factory=RealDictCursor)

    try:
        if args.show_info:
            display_db_info(cur_old, cur_new, args.db_old, args.db_new, tables)
            
        for table in tables:
            compare_table_data(cur_old, cur_new, table, args.mode)
            
    finally:
        cur_old.close()
        cur_new.close()
        conn_old.close()
        conn_new.close()
        print("\n[+] 검증이 안전하게 완료되었습니다.")

if __name__ == "__main__":
    main()