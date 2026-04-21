"""Domain registry — central lookup for domain instances."""

from __future__ import annotations

from typing import Any

from forge_arena.arena.domains.base import BaseDomain
from forge_arena.arena.domains.code_review import CodeReviewDomain
from forge_arena.arena.domains.customer_support import CustomerSupportDomain
from forge_arena.arena.domains.legal_summarisation import LegalSummarisationDomain
from forge_arena.arena.domains.mixed import MixedDomain
from forge_arena.arena.domains.product_recommendation import ProductRecommendationDomain
from forge_arena.models.tasks import TaskDomain

DOMAIN_REGISTRY: dict[TaskDomain, BaseDomain] = {}


def init_domain_registry(task_bank: list[dict[str, Any]]) -> None:
    """Populate the registry with domain instances loaded from the task bank.

    Called once at application startup from main.py lifespan context.
    """
    DOMAIN_REGISTRY[TaskDomain.CUSTOMER_SUPPORT] = CustomerSupportDomain(task_bank)
    DOMAIN_REGISTRY[TaskDomain.LEGAL_SUMMARISATION] = LegalSummarisationDomain(task_bank)
    DOMAIN_REGISTRY[TaskDomain.CODE_REVIEW] = CodeReviewDomain(task_bank)
    DOMAIN_REGISTRY[TaskDomain.PRODUCT_RECOMMENDATION] = ProductRecommendationDomain(task_bank)
    DOMAIN_REGISTRY[TaskDomain.MIXED] = MixedDomain(task_bank)


def get_domain(domain: TaskDomain) -> BaseDomain:
    if domain not in DOMAIN_REGISTRY:
        raise KeyError(f"Domain '{domain}' is not registered. Call init_domain_registry first.")
    return DOMAIN_REGISTRY[domain]


__all__ = [
    "BaseDomain",
    "DOMAIN_REGISTRY",
    "init_domain_registry",
    "get_domain",
    "CustomerSupportDomain",
    "LegalSummarisationDomain",
    "CodeReviewDomain",
    "ProductRecommendationDomain",
    "MixedDomain",
]
