use serde::Serialize;
use tracing::{debug, info};

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub cache_hit_tokens: usize,
}

impl CacheStats {
    pub fn cache_hit_ratio(&self) -> f64 {
        if self.prompt_tokens == 0 { return 0.0; }
        self.cache_hit_tokens as f64 / self.prompt_tokens as f64
    }
}

#[derive(Debug, Serialize)]
pub struct CacheSummary {
    pub total_requests: usize,
    pub total_prompt_tokens: usize,
    pub cache_hit_ratio: f64,
    pub current_context_tokens: usize,
    pub max_context_tokens: usize,
    pub last_prefill_ms: f64,
}

#[derive(Debug)]
pub struct CacheManager {
    pub max_context_tokens: usize,
    pub system_prompt_tokens: usize,
    pub truncation_threshold: f64,
    history_tokens: usize,
    total_requests: usize,
    total_cache_hits: usize,
    total_prompt_tokens: usize,
    last_stats: Option<CacheStats>,
}

impl CacheManager {
    pub fn new(max_context: usize, truncation_threshold: f64) -> Self {
        Self {
            max_context_tokens: max_context,
            system_prompt_tokens: 0,
            truncation_threshold,
            history_tokens: 0,
            total_requests: 0,
            total_cache_hits: 0,
            total_prompt_tokens: 0,
            last_stats: None,
        }
    }

    pub fn record(&mut self, stats: CacheStats) {
        self.total_requests += 1;
        self.total_cache_hits += stats.cache_hit_tokens;
        self.total_prompt_tokens += stats.prompt_tokens;
        debug!(
            "Cache: prefill={} tok ({:.0}ms), gen={} tok, hit={:.0}%",
            stats.prompt_tokens, stats.prefill_ms, stats.generated_tokens,
            stats.cache_hit_ratio() * 100.0,
        );
        self.last_stats = Some(stats);
    }

    pub fn update_history_tokens(&mut self, count: usize) {
        self.history_tokens = count;
    }

    pub fn total_tokens(&self) -> usize {
        self.system_prompt_tokens + self.history_tokens
    }

    pub fn remaining_tokens(&self) -> usize {
        self.max_context_tokens.saturating_sub(self.total_tokens())
    }

    pub fn needs_truncation(&self) -> bool {
        self.total_tokens() as f64 > self.max_context_tokens as f64 * self.truncation_threshold
    }

    pub fn truncation_target(&self) -> usize {
        let target = (self.max_context_tokens as f64 * 0.6) as usize;
        target.saturating_sub(self.system_prompt_tokens)
    }

    pub fn overall_cache_hit_ratio(&self) -> f64 {
        if self.total_prompt_tokens == 0 { return 0.0; }
        self.total_cache_hits as f64 / self.total_prompt_tokens as f64
    }

    pub fn summary(&self) -> CacheSummary {
        CacheSummary {
            total_requests: self.total_requests,
            total_prompt_tokens: self.total_prompt_tokens,
            cache_hit_ratio: self.overall_cache_hit_ratio(),
            current_context_tokens: self.total_tokens(),
            max_context_tokens: self.max_context_tokens,
            last_prefill_ms: self.last_stats.as_ref().map(|s| s.prefill_ms).unwrap_or(0.0),
        }
    }

    pub fn log_summary(&self) {
        let s = self.summary();
        info!(
            "Cache: {} requests, {:.0}% hit, {}/{} tokens, last prefill {:.0}ms",
            s.total_requests, s.cache_hit_ratio * 100.0,
            s.current_context_tokens, s.max_context_tokens, s.last_prefill_ms,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basics() {
        let mut cm = CacheManager::new(4096, 0.8);
        cm.system_prompt_tokens = 100;
        cm.update_history_tokens(200);
        assert_eq!(cm.total_tokens(), 300);
        assert_eq!(cm.remaining_tokens(), 3796);
        assert!(!cm.needs_truncation());
    }

    #[test]
    fn test_truncation_needed() {
        let mut cm = CacheManager::new(1000, 0.8);
        cm.system_prompt_tokens = 100;
        cm.update_history_tokens(750);
        assert!(cm.needs_truncation());
    }

    #[test]
    fn test_record_stats() {
        let mut cm = CacheManager::new(4096, 0.8);
        cm.record(CacheStats {
            prompt_tokens: 100,
            generated_tokens: 20,
            prefill_ms: 15.0,
            generation_ms: 50.0,
            cache_hit_tokens: 80,
        });
        assert_eq!(cm.total_requests, 1);
        assert_eq!(cm.overall_cache_hit_ratio(), 0.8);
    }
}
