from typing import Annotated, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class WorkSummary(BaseModel):
    id: str
    title: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None


class MatchRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(
        ...,
        min_length=50,
        max_length=5000,
        description="Abstract or research topic",
        validation_alias=AliasChoices("query", "abstract"),
    )
    top_n: Annotated[int, Field(20, ge=1, le=50)] = 20

    min_h_index: Optional[Annotated[int, Field(ge=0, le=200)]] = None
    min_works: Optional[Annotated[int, Field(ge=0)]] = None
    countries: Optional[list[str]] = Field(
        None, description="ISO country codes, e.g. ['GB', 'US']"
    )
    exclude_openalex_ids: Optional[list[str]] = Field(
        None, description="OpenAlex author IDs to exclude (e.g. co-authors)"
    )
    require_active_since: Optional[int] = Field(
        2022, description="Require a work in or after this year"
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("query cannot be empty")
        return s


class AuthorMatch(BaseModel):
    openalex_id: str
    display_name: str
    orcid: Optional[str] = None

    institution_name: Optional[str] = None
    institution_country: Optional[str] = None

    h_index: int = 0
    i10_index: int = 0
    works_count: int = 0
    cited_by_count: int = 0

    top_concepts: list[str] = Field(default_factory=list)

    score: float = Field(..., description="Combined match score, 0–100")
    similarity: float = Field(..., description="Semantic similarity, 0–1")

    matching_works: list[WorkSummary] = Field(
        ..., description="Works that help explain the match"
    )
    profile_url: str = Field(..., description="OpenAlex profile URL")


class MatchResponse(BaseModel):
    query_summary: str
    total_matched: int
    processing_time_ms: float
    results: list[AuthorMatch]
    data_freshness: str = Field(
        ...,
        description="Latest author row update date (YYYY-MM-DD) or 'no data'",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    authors_in_db: int
    index_built: bool
    faiss_vectors: Optional[int] = None
    last_ingestion: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class AuthorDetail(BaseModel):
    author_id: int
    openalex_id: str
    display_name: str
    institution_name: Optional[str] = None
    h_index: Optional[int] = None
    works_count: Optional[int] = None
    profile_text: Optional[str] = None
    last_known_activity_year: Optional[int] = None
