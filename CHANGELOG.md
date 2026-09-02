# Changelog

All notable changes to the `retrieval-core` project will be documented in this file.

## [v0.2.0] - 2026-09-02

### Added
- Introduced a parallel `Document` model (containing `doc_id` and `metadata`) to support new agents (e.g., TripAgent).
- Added a `DocumentAdapter` protocol (with `get_document` and `mget_documents`) alongside the existing `CorpusAdapter`.
- Updated `ScoredHit` to support either `product_id` or `doc_id` by making them both optional fields.

## [v0.1.x] - Existing Features

### Features
- **Product Model**: Core schema representing a product with `product_id` and `title` (used primarily by ShopRank).
- **CorpusAdapter Protocol**: Base interface for retrieving product data via `get_product` and `mget_products`.
- **Search Pipeline Models**: Core schemas for search requests and responses, including `Query`, `SearchResponse`, and `PipelineConfig`.
- **Scoring & Ranking Models**: Detailed `ScoredHit` and `ScoreBreakdown` schemas to track `bm25_score`, `dense_score`, `fused_score`, `rerank_score`, and `matched_terms`.
