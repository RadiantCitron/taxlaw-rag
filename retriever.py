# === PROJECT: taxlaw-rag | DOMAIN: 세법 | DB: tax_* ===
import logging
from database.postgres_repo import PostgresRepo, normalize_sources
from database.neo4j_repo import Neo4jRepo
from model_registry import use_embed, use_embed_any, use_rerank
from rag_run import (
    RetrievalHit,
    RetrievalResult,
    diversify_tagged,
    format_context,
    law_ids_from_hits,
)
from retriever_kiwi_patch import get_keywords_kiwi
from search_constants import keywords_from_text
from session_manager import raise_if_cancelled

logger = logging.getLogger("my_rag")


def _keywords_from_text(text: str, fallback: str) -> list[str]:
    return keywords_from_text(text, fallback)


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_hit(row, rank, score=None, source_table: str = "") -> RetrievalHit:
    raw_id = row[0]
    fusion = _as_float(row[4]) if len(row) > 4 else None
    hit_score = _as_float(score) if score is not None else fusion
    table = source_table or ""
    doc_id = f"{table}:{raw_id}" if table else str(raw_id)
    return RetrievalHit(
        doc_id=doc_id,
        source=str(row[1] or table or "법령"),
        rank=int(rank),
        score=hit_score,
        content=row[3] or "",
        level=str(row[2] or ""),
        parent_id=raw_id,
    )


def _row_score(row) -> float:
    val = _as_float(row[4]) if len(row) > 4 else None
    return val if val is not None else float("-inf")


def get_hybrid_assembled_context(
    query_text: str,
    llm_optimizer,
    active_prompts,
    query_mode: str = "none",
    fusion_mode: str = "rrf",
    alpha: float = 0.5,
    use_graphdb: bool = False,
    embed_model_name: str = "jhgan/ko-sroberta-multitask",
    use_reranker: bool = False,
    rerank_model_name: str = "Dongjin-kr/ko-reranker",
    top_k: int = 3,
    min_score: float = 0.0,
    rerank_candidate_cap: int = 50,
    base_limit: int = 20,
    return_ids: bool = False,
    trace_prompts: bool = False,
    embed_variant: str | None = None,
    keyword_mode: str = "trigram",
    sources=None,
    source_table: str | None = None,
) -> RetrievalResult:
    """Postgres 하이브리드 검색 + (옵션) Neo4j 보충. 항상 RetrievalResult 를 반환한다.

    sources : 검색할 원본 테이블 목록. 허용 목록의 부분 집합.
        비우면 허용된 네 테이블 전부. source_table 은 예전 단일 인자(호환).

    embed_variant : None(기본)이면 각 원본의 embedding 컬럼.
        비교 모델 id 면 그 원본의 cmp 테이블 벡터를 쓴다.
    """
    _ = return_ids
    if sources is None and source_table:
        sources = [source_table]
    tables = normalize_sources(sources)
    empty = RetrievalResult(expanded_query=query_text or "")
    mode = (query_mode or "none").strip().lower()
    if mode == "vocab_mapping" and embed_variant:
        raise ValueError(
            "query_mode=vocab_mapping 과 embed_variant 는 함께 쓸 수 없습니다. "
            "단어장 매핑은 원본 테이블 검색 경로입니다."
        )
    if mode == "vocab_mapping" and (keyword_mode or "trigram").strip().lower() == "kiwi":
        raise ValueError(
            "query_mode=vocab_mapping 과 keyword_mode=kiwi 는 함께 쓸 수 없습니다. "
            "vocab_mapping 은 LLM 이 키워드를 만드는 경로라 형태소 분석이 개입할 자리가 없습니다."
        )

    probe = PostgresRepo()
    try:
        probe.ensure_search_tables(tables, embed_variant)
    finally:
        probe.close()

    expanded_query = query_text
    shared_keywords = None
    query_vector = None
    with use_embed(embed_model_name) as embed_slot:
        embed_model = embed_slot.model

        if mode == "expansion":
            logger.info("[MODE] Query Expansion 모드 작동")
            expanded_query = llm_optimizer.expand_query(
                query_text, active_prompts.query_expansion, trace_prompts=trace_prompts
            )
            shared_keywords = _keywords_from_text(expanded_query, query_text)
            with embed_slot.infer_lock:
                query_vector = embed_model.get_text_embedding(expanded_query)
        elif mode == "vocab_mapping":
            logger.info("[MODE] Vocab Mapping 모드 작동 (테이블마다 단어장)")
            with embed_slot.infer_lock:
                query_vector = embed_model.get_text_embedding(query_text)
        else:
            logger.info("[MODE] 질문 최적화 생략 (원문 검색)")
            shared_keywords = _keywords_from_text(query_text, query_text)
            with embed_slot.infer_lock:
                query_vector = embed_model.get_text_embedding(query_text)

        if embed_variant:
            _q_for_embed = expanded_query if mode == "expansion" else query_text
            with use_embed_any(embed_variant) as cmp_slot:
                with cmp_slot.infer_lock:
                    query_vector = cmp_slot.model.get_text_embedding(_q_for_embed)
            logger.info(f"[COMPARE] 비교 임베딩으로 질문 재계산: {embed_variant}")

        if keyword_mode == "kiwi" and mode != "vocab_mapping":
            _q_for_kw = expanded_query if mode == "expansion" else query_text
            shared_keywords = get_keywords_kiwi(_q_for_kw, query_text)
            logger.info(f"[SEARCH] Kiwi 형태소 키워드: {shared_keywords}")

    keywords_by_table: dict[str, list] = {}
    if mode == "vocab_mapping":
        vocabs: dict[str, str] = {}
        vocab_repo = PostgresRepo()
        try:
            for table in tables:
                raise_if_cancelled()
                vocabs[table] = vocab_repo.extract_db_vocabulary(table, limit=500)
        finally:
            vocab_repo.close()
        for table, vocab in vocabs.items():
            keywords_by_table[table] = llm_optimizer.optimize_to_sql_keywords(
                query_text, vocab, active_prompts.vocab_mapping,
                trace_prompts=trace_prompts,
            )

    tagged: list[tuple[str, tuple]] = []
    last_keywords = shared_keywords or []
    with PostgresRepo() as repo:
        for table in tables:
            raise_if_cancelled()
            if mode == "vocab_mapping":
                keywords = list(keywords_by_table.get(table) or [])
            else:
                keywords = list(shared_keywords or [])
            last_keywords = keywords
            logger.info(f"[SEARCH] {table} 검색 시작")
            if fusion_mode == "score_based":
                rows = repo.search_hybrid_score_based(
                    keywords, query_vector, alpha, base_limit,
                    source_table=table, embed_variant=embed_variant,
                )
            else:
                rows = repo.search_hybrid_rrf(
                    keywords, query_vector, base_limit,
                    source_table=table, embed_variant=embed_variant,
                )
            for row in rows:
                tagged.append((table, row))
    logger.info("[DB] Postgres 검색 완료 및 커넥션 조기 반납.")

    tagged.sort(key=lambda item: _row_score(item[1]), reverse=True)
    merged_results = [row for _, row in tagged]
    merged_tables = [table for table, _ in tagged]
    logger.info(
        f"[HYBRID] 1차 후보 {len(merged_results)}개 "
        f"(테이블 {len(tables)}개: {', '.join(tables)})"
    )
    if not merged_results:
        empty.keywords = list(last_keywords or [])
        empty.expanded_query = expanded_query or query_text or ""
        return empty

    candidate_hits = [
        _to_hit(row, rank=i, source_table=table)
        for i, (table, row) in enumerate(tagged, 1)
    ]

    top_scored: list[tuple] = []
    if use_reranker:
        cap = max(1, min(int(rerank_candidate_cap or 50), 50))
        pool_results = merged_results[:cap]
        pool_tables = merged_tables[:cap]
        if len(merged_results) > len(pool_results):
            logger.info(
                "[RERANK] 후보 %d건 중 상위 %d건만 리랭킹",
                len(merged_results), len(pool_results),
            )
        with use_rerank(rerank_model_name) as rerank_slot:
            logger.info(
                f"[RERANK] CrossEncoder 리랭킹 진행 중... "
                f"(후보 {len(pool_results)}개 → Top-K: {top_k})"
            )
            pairs = [(expanded_query, r[3]) for r in pool_results]
            with rerank_slot.infer_lock:
                scores = rerank_slot.model.predict(pairs)
            scored_results = sorted(
                zip(pool_results, pool_tables, scores),
                key=lambda x: x[2],
                reverse=True,
            )
            filtered = [
                item for item in scored_results if item[2] >= min_score
            ]
            if scored_results:
                vals = [float(s) for _, _, s in scored_results]
                logger.info(
                    "[RERANK] 점수 범위 min=%.4f max=%.4f min_score=%.4f 통과=%d/%d",
                    min(vals), max(vals), min_score, len(filtered), len(scored_results),
                )
            if not filtered and scored_results:
                logger.warning(
                    "[RERANK] min_score=%.4f 가 후보를 전부 걸렀습니다. "
                    "필터 없이 top_k=%d 를 씁니다.",
                    min_score, top_k,
                )
                top_scored = scored_results[:top_k]
            else:
                top_scored = filtered[:top_k]
            logger.info(f"[RERANK] 최종 {len(top_scored)}개 노드 선정 완료.")
    else:
        logger.info(f"[RERANK] 리랭킹 생략. 상위 {top_k}개 사용.")
        picked = diversify_tagged(tagged, top_k)
        top_scored = [
            (row, table, row[4] if len(row) > 4 else None)
            for table, row in picked
        ]

    hits = []
    for i, item in enumerate(top_scored, 1):
        row, table, score = item[0], item[1], item[2]
        hits.append(_to_hit(row, rank=i, score=score, source_table=table))

    assembled_text = format_context(hits)
    graph_context = ""
    graph_ids = law_ids_from_hits(hits)
    if use_graphdb and graph_ids:
        logger.info(
            f"[GRAPH] Neo4j 보충 컨텍스트 검색 중... "
            f"(법령 ID: {len(graph_ids)}개 / hit {len(hits)}개)"
        )
        neo4j_repo = Neo4jRepo()
        try:
            graph_context = neo4j_repo.get_graph_context(graph_ids) or ""
            if graph_context:
                assembled_text += (
                    f"\n\n{'='*40}\n{graph_context}{'='*40}\n"
                )
                logger.info("[GRAPH] 보충 컨텍스트 병합 완료.")
            else:
                logger.info("[GRAPH] 연결된 보충 정보가 없습니다.")
        finally:
            neo4j_repo.close()
    elif use_graphdb:
        logger.info("[GRAPH] 법령 hit 가 없어 Neo4j 조회를 건너뜁니다.")

    return RetrievalResult(
        context_text=assembled_text,
        hits=hits,
        candidates=candidate_hits,
        keywords=list(last_keywords or []),
        expanded_query=expanded_query or query_text or "",
        graph_context=graph_context,
    )
