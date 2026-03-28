"""KV cache management for edgeloop.

Abstracts cache behavior so the agent loop can make cache-aware decisions
without knowing which backend is in use. Each backend reports its cache
state; the CacheManager uses that to decide when to truncate, what
format to send, and logs cache efficiency metrics.

The key insight for local LLMs: prefill is the bottleneck. Every token
we avoid re-prefilling is a direct latency win. This module exists to
minimize prefill at every opportunity.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Stats from a single completion request."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    prefill_ms: float = 0.0
    generation_ms: float = 0.0
    cache_hit_tokens: int = 0  # tokens that were in cache (not re-prefilled)

    @property
    def prefill_tps(self) -> float:
        """Prefill tokens per second."""
        if self.prefill_ms <= 0:
            return 0.0
        return self.prompt_tokens / (self.prefill_ms / 1000)

    @property
    def generation_tps(self) -> float:
        """Generation tokens per second."""
        if self.generation_ms <= 0:
            return 0.0
        return self.generated_tokens / (self.generation_ms / 1000)

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of prompt tokens served from cache."""
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cache_hit_tokens / self.prompt_tokens


@dataclass
class CacheManager:
    """Tracks KV cache state and makes cache-aware decisions.

    Each agent gets a CacheManager. It tracks:
    - How many tokens are currently cached (prefix length)
    - Cache hit ratios over time
    - When truncation is needed to stay within context budget

    The CacheManager doesn't talk to backends directly — it receives
    stats after each completion and advises the agent on what to do.
    """

    max_context_tokens: int = 4096
    system_prompt_tokens: int = 0  # set once at agent init
    _history_tokens: int = 0
    _total_requests: int = 0
    _total_cache_hits: int = 0
    _total_prompt_tokens: int = 0
    _last_stats: CacheStats | None = None
    _per_request_stats: list[CacheStats] = field(default_factory=list)

    def record(self, stats: CacheStats):
        """Record stats from a completion request."""
        self._total_requests += 1
        self._total_cache_hits += stats.cache_hit_tokens
        self._total_prompt_tokens += stats.prompt_tokens
        self._last_stats = stats
        self._per_request_stats.append(stats)

        logger.debug(
            "Cache: prefill=%d tok (%.0fms), gen=%d tok, cache_hit=%.0f%%",
            stats.prompt_tokens, stats.prefill_ms,
            stats.generated_tokens, stats.cache_hit_ratio * 100,
        )

    def update_history_tokens(self, token_count: int):
        """Update the estimated token count of the current history."""
        self._history_tokens = token_count

    @property
    def total_tokens(self) -> int:
        """Estimated total tokens in current prompt (system + history)."""
        return self.system_prompt_tokens + self._history_tokens

    @property
    def remaining_tokens(self) -> int:
        """Tokens available before hitting context limit."""
        return max(0, self.max_context_tokens - self.total_tokens)

    @property
    def needs_truncation(self) -> bool:
        """True if we're over 80% of context budget — truncate soon."""
        return self.total_tokens > self.max_context_tokens * 0.8

    @property
    def overall_cache_hit_ratio(self) -> float:
        """Lifetime cache hit ratio across all requests."""
        if self._total_prompt_tokens <= 0:
            return 0.0
        return self._total_cache_hits / self._total_prompt_tokens

    def truncation_target(self) -> int:
        """How many history messages to keep after truncation.

        Strategy: keep system prompt + most recent messages that fit
        in 60% of context budget. This leaves room for the next
        tool call cycle without another truncation.
        """
        target_tokens = int(self.max_context_tokens * 0.6) - self.system_prompt_tokens
        return max(target_tokens, 0)

    def summary(self) -> dict:
        """Return a summary of cache performance."""
        return {
            "total_requests": self._total_requests,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_cache_hits": self._total_cache_hits,
            "cache_hit_ratio": self.overall_cache_hit_ratio,
            "current_context_tokens": self.total_tokens,
            "max_context_tokens": self.max_context_tokens,
            "last_prefill_ms": self._last_stats.prefill_ms if self._last_stats else 0,
        }

    def log_summary(self):
        """Log a human-readable cache performance summary."""
        s = self.summary()
        logger.info(
            "Cache summary: %d requests, %.0f%% cache hit ratio, "
            "%d/%d context tokens, last prefill %.0fms",
            s["total_requests"],
            s["cache_hit_ratio"] * 100,
            s["current_context_tokens"],
            s["max_context_tokens"],
            s["last_prefill_ms"],
        )
