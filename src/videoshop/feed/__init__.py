"""Feed-control layer: exposure selection above the existing intervention env.

The agent picks the next ``user x video x product x treatment`` exposure from a
bounded candidate set produced by the environment. Organic content is a first-class
candidate, so "no commercial intervention" means serving organic content rather
than idling.
"""

from videoshop.feed.candidate_provider import (
    CandidateBatch,
    CandidateProviderConfig,
    ExposureTruth,
    SyntheticCandidateProvider,
)
from videoshop.feed.composite_env import FeedInterventionEnv
from videoshop.feed.control_env import FeedControlEnv, FeedObservation
from videoshop.feed.observation import (
    COUPON_AUTHORITY_FIELDS,
    INTERVENTION_TASK,
    InterventionObservation,
    build_intervention_observation,
    build_intervention_system_prompt,
    intervention_few_shot_messages,
)
from videoshop.feed.policies import (
    FEED_POLICIES,
    AlwaysOrganicPolicy,
    GreedyScorePolicy,
    RandomFeedPolicy,
    RuleBasedFeedPolicy,
    build_feed_policy,
    build_intervention_agent,
    build_intervention_policy,
)
from videoshop.feed.reward import (
    DEFAULT_FEED_REWARD_CONFIG,
    DEFAULT_SCALARIZATION,
    REWARD_COMPONENTS,
    SCALARIZATIONS,
    FeedRewardConfig,
    RewardConfig,
    RewardVector,
    RewardWeights,
    compute_reward_vector,
    expected_reward_vector,
    resolve_weights,
    reward_terms,
)
from videoshop.feed.legacy import is_feed_episode, treatment_for_legacy_action, upgrade_v1_episode
from videoshop.feed.scenarios import build_default_provider, build_feed_scenario, build_smoke_scenarios
from videoshop.feed.schemas import (
    COMMERCIAL_SOURCE_TYPES,
    FEED_ACTION_TYPES,
    FEED_SCHEMA_VERSION,
    FORBIDDEN_PUBLIC_KEYS,
    LEGACY_SCHEMA_VERSION,
    PLACEMENTS,
    SOURCE_TYPES,
    TREATMENTS,
    BaseScores,
    Eligibility,
    ExposureCandidate,
    FeedDecision,
    ScenarioSpec,
    find_forbidden_keys,
    find_leaks,
)
from videoshop.feed.user_model import FeedStateUpdate, FeedUserModel, FeedUserResponse

__all__ = [
    "COMMERCIAL_SOURCE_TYPES",
    "COUPON_AUTHORITY_FIELDS",
    "DEFAULT_FEED_REWARD_CONFIG",
    "DEFAULT_SCALARIZATION",
    "FEED_ACTION_TYPES",
    "FEED_POLICIES",
    "FEED_SCHEMA_VERSION",
    "FORBIDDEN_PUBLIC_KEYS",
    "INTERVENTION_TASK",
    "LEGACY_SCHEMA_VERSION",
    "PLACEMENTS",
    "REWARD_COMPONENTS",
    "SCALARIZATIONS",
    "SOURCE_TYPES",
    "TREATMENTS",
    "AlwaysOrganicPolicy",
    "BaseScores",
    "CandidateBatch",
    "CandidateProviderConfig",
    "Eligibility",
    "ExposureCandidate",
    "ExposureTruth",
    "FeedControlEnv",
    "FeedDecision",
    "FeedInterventionEnv",
    "FeedObservation",
    "FeedRewardConfig",
    "FeedStateUpdate",
    "FeedUserModel",
    "FeedUserResponse",
    "GreedyScorePolicy",
    "InterventionObservation",
    "RandomFeedPolicy",
    "RewardConfig",
    "RewardVector",
    "RewardWeights",
    "RuleBasedFeedPolicy",
    "ScenarioSpec",
    "SyntheticCandidateProvider",
    "build_default_provider",
    "build_feed_policy",
    "build_feed_scenario",
    "build_intervention_agent",
    "build_intervention_observation",
    "build_intervention_policy",
    "build_intervention_system_prompt",
    "intervention_few_shot_messages",
    "build_smoke_scenarios",
    "compute_reward_vector",
    "expected_reward_vector",
    "find_forbidden_keys",
    "find_leaks",
    "is_feed_episode",
    "resolve_weights",
    "reward_terms",
    "treatment_for_legacy_action",
    "upgrade_v1_episode",
]
