from __future__ import annotations

from typing import Callable, Dict, List

from .models import PlanQuerySpec, PlanStep, QueryPlan


def _plan_q1_movies_by_genre(spec: PlanQuerySpec) -> QueryPlan:
    step = PlanStep(
        name="filter_movies_by_genre",
        operator_type="filter",
        target_collection="Movie",
        filter_key="genre",
        output_fields=["movieId", "title", "genre"],
        use_sharding=True,
    )
    return QueryPlan(query=spec, steps=[step])


def _plan_q2_reviews_for_movie(spec: PlanQuerySpec) -> QueryPlan:
    step = PlanStep(
        name="filter_reviews_by_movie",
        operator_type="filter",
        target_collection="Review",
        filter_key="movieId",
        output_fields=["movieId", "userId", "rating"],
        use_sharding=True,
    )
    return QueryPlan(query=spec, steps=[step])


def _plan_q3_movie_review_join(spec: PlanQuerySpec) -> QueryPlan:
    step = PlanStep(
        name="join_movies_reviews",
        operator_type="join",
        left_ref="Movie",
        right_ref="Review",
        join_key="movieId",
        join_selectivity=1.0,
        output_fields=["left.movieId", "left.title", "right.rating"],
        use_sharding=True,
    )
    return QueryPlan(query=spec, steps=[step])


def _plan_q4_avg_rating_by_movie(spec: PlanQuerySpec) -> QueryPlan:
    step = PlanStep(
        name="aggregate_reviews_by_movie",
        operator_type="aggregate",
        target_collection="Review",
        grouping_keys=["movieId"],
        output_fields=["movieId", "avg_rating"],
        use_sharding=True,
    )
    return QueryPlan(query=spec, steps=[step])


def _plan_q5_top_movies_with_titles(spec: PlanQuerySpec) -> QueryPlan:
    aggregate_step = PlanStep(
        name="aggregate_reviews_for_titles",
        operator_type="aggregate",
        target_collection="Review",
        grouping_keys=["movieId"],
        output_fields=["movieId", "avg_rating"],
        use_sharding=True,
    )
    join_step = PlanStep(
        name="join_ratings_with_titles",
        operator_type="join",
        left_ref="aggregate_reviews_for_titles",
        right_ref="Movie",
        join_key="movieId",
        join_selectivity=1.0,
        output_fields=["left.movieId", "left.avg_rating", "right.title"],
        use_sharding=True,
    )
    return QueryPlan(query=spec, steps=[aggregate_step, join_step])


_PLANNERS: Dict[str, Callable[[PlanQuerySpec], QueryPlan]] = {
    "Q1_movies_by_genre": _plan_q1_movies_by_genre,
    "Q2_reviews_for_movie": _plan_q2_reviews_for_movie,
    "Q3_movie_review_join": _plan_q3_movie_review_join,
    "Q4_avg_rating_by_movie": _plan_q4_avg_rating_by_movie,
    "Q5_top_movies_with_titles": _plan_q5_top_movies_with_titles,
}


def build_plan(spec: PlanQuerySpec) -> QueryPlan:
    if spec.name not in _PLANNERS:
        raise ValueError(f"No planner available for query: {spec.name}")
    return _PLANNERS[spec.name](spec)


def build_plans(specs: List[PlanQuerySpec]) -> List[QueryPlan]:
    return [build_plan(spec) for spec in specs]
