from __future__ import annotations

import copy
import json
from itertools import combinations

import pytest

from videoshop.data.synthetic import build_synthetic_catalog
from videoshop.feed.benchmark import (
    FEED_SPLITS,
    OracleStepGreedyPolicy,
    build_split,
    dataset_hash,
    run_feed_benchmark,
)
from videoshop.feed.control_env import FeedControlEnv
from videoshop.feed.splits import FEED_WORLD_CONFIG, partition_world


@pytest.fixture(scope="module")
def smoke_split():
    return build_split("smoke", count=40)


@pytest.fixture(scope="module")
def provider(smoke_split):
    return smoke_split.provider


@pytest.fixture(scope="module")
def scenarios(smoke_split):
    return smoke_split.scenarios


@pytest.fixture(scope="module")
def summaries(provider, scenarios):
    results = {}
    for name in ("random", "always_organic", "greedy_gmv", "rule_based", OracleStepGreedyPolicy.name):
        _episodes, summary = run_feed_benchmark(
            name, scenarios=scenarios, provider=provider, seed=42, max_steps=6
        )
        results[name] = summary
    return results


def test_all_baselines_run_end_to_end(summaries):
    for name, summary in summaries.items():
        assert summary["episodes"] == 40, name
        assert summary["schema_version"] == "v2"
        assert set(summary["avg_reward_vector"]) == {
            "content_value",
            "commerce_value",
            "user_value",
            "ecosystem_value",
            "risk_cost",
        }


def test_baselines_are_ordered_and_leave_headroom(summaries):
    random_reward = summaries["random"]["avg_scalar_reward"]
    rule_based = summaries["rule_based"]["avg_scalar_reward"]
    oracle = summaries[OracleStepGreedyPolicy.name]["avg_scalar_reward"]

    assert rule_based > random_reward, "a sane heuristic must beat random"
    assert oracle > rule_based, "the heuristic must not saturate the ceiling"


def test_organic_only_is_a_strong_but_beatable_floor(summaries):
    organic_only = summaries["always_organic"]["avg_scalar_reward"]
    oracle = summaries[OracleStepGreedyPolicy.name]["avg_scalar_reward"]

    assert organic_only > summaries["random"]["avg_scalar_reward"]
    assert oracle > organic_only, "serving zero commerce must not be optimal"
    assert oracle > summaries["rule_based"]["avg_scalar_reward"], "heuristics must leave headroom"
    assert summaries["always_organic"]["commercial_exposure_share"] == 0.0
    assert summaries[OracleStepGreedyPolicy.name]["commercial_exposure_share"] > 0.0


def test_gmv_chasing_wins_on_gmv_first_relative_to_its_own_average(summaries):
    """Objective profiles must actually change what counts as good."""
    greedy = summaries["greedy_gmv"]["avg_scalar_reward_by_profile"]

    assert greedy["gmv_first"] > greedy["retention_first"]
    assert greedy["gmv_first"] > greedy["content_first"]


def test_commerce_channel_is_not_inert(summaries):
    oracle = summaries[OracleStepGreedyPolicy.name]
    assert oracle["avg_reward_vector"]["commerce_value"] > 0.0
    assert summaries["greedy_gmv"]["purchase_rate"] > 0.01


def test_random_is_the_only_baseline_that_trips_hard_constraints(summaries):
    assert summaries["random"]["blocked_decision_rate"] > 0.0
    for name in ("always_organic", "greedy_gmv", "rule_based"):
        assert summaries[name]["blocked_decision_rate"] == 0.0, name


def test_confidence_interval_brackets_the_mean(summaries):
    for summary in summaries.values():
        low, high = summary["scalar_reward_ci95"]
        assert low <= summary["avg_scalar_reward"] <= high


def test_dataset_hash_is_stable_and_split_sensitive(scenarios):
    assert dataset_hash(scenarios) == dataset_hash(build_split("smoke", count=40).scenarios)
    assert dataset_hash(scenarios) != dataset_hash(build_split("frozen_eval", count=40).scenarios)


def test_split_scenario_ids_are_disjoint():
    ids = {name: {spec.scenario_id for spec in build_split(name, count=10)} for name in FEED_SPLITS}

    assert ids["smoke"] & ids["frozen_eval"] == set()
    assert ids["smoke"] & ids["train"] == set()
    assert ids["train"] & ids["frozen_eval"] == set()


def test_split_entity_pools_share_no_products_videos_or_users():
    """Disjoint scenario ids are not isolation; the underlying entities must differ."""
    pools = partition_world()

    for left, right in combinations(FEED_SPLITS, 2):
        assert pools[left].product_ids() & pools[right].product_ids() == set(), f"{left}/{right} products"
        assert pools[left].video_ids() & pools[right].video_ids() == set(), f"{left}/{right} videos"
        assert pools[left].user_ids() & pools[right].user_ids() == set(), f"{left}/{right} users"


def test_partition_covers_the_whole_world_without_duplication():
    pools = partition_world()
    world_products = build_synthetic_catalog(FEED_WORLD_CONFIG)

    counted = sum(len(pool.products) for pool in pools.values())
    union = set().union(*(pool.product_ids() for pool in pools.values()))
    assert counted == len(world_products)
    assert union == {product.product_id for product in world_products}


def test_every_split_still_covers_every_category():
    pools = partition_world()
    categories = [pool.categories() for pool in pools.values()]

    assert len(set(map(frozenset, categories))) == 1, "splits must differ in entities, not in categories"
    assert len(categories[0]) == FEED_WORLD_CONFIG.category_count


def test_generated_scenarios_only_reference_their_own_pool():
    for name in FEED_SPLITS:
        split = build_split(name, count=15)
        pool_products = split.pool.product_ids()
        pool_videos = split.pool.video_ids()
        pool_users = split.pool.user_ids()

        for scenario in split.scenarios:
            assert scenario.hidden_world_state["user_profile"]["user_id"] in pool_users
            assert set(scenario.hidden_world_state["videos"]) <= pool_videos
            assert set(scenario.hidden_world_state["products"]) <= pool_products


def test_frozen_eval_entities_never_appear_in_train_scenarios():
    train = build_split("train", count=60)
    frozen = build_split("frozen_eval", count=60)

    def referenced(split):
        products, videos = set(), set()
        for scenario in split.scenarios:
            products |= set(scenario.hidden_world_state["products"])
            videos |= set(scenario.hidden_world_state["videos"])
        return products, videos

    train_products, train_videos = referenced(train)
    frozen_products, frozen_videos = referenced(frozen)

    assert train_products & frozen_products == set()
    assert train_videos & frozen_videos == set()
    assert frozen_products, "the guard would pass vacuously on an empty split"


def test_one_episode_uses_a_single_consistent_catalog(provider, scenarios):
    """Step 0 and later steps must not disagree about what a product id means."""
    split = build_split("frozen_eval", count=5)
    env = FeedControlEnv(split.provider, max_steps=6, seed=42)
    runtime_catalog = {product.product_id: product for product in split.provider.products}

    for scenario in split.scenarios:
        for product_id, payload in scenario.hidden_world_state["products"].items():
            runtime = runtime_catalog[product_id]
            assert payload["price"] == runtime.price
            assert payload["review_risk"] == runtime.review_risk
    del env, provider, scenarios


def test_partition_rejects_shares_that_do_not_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        partition_world(shares={"train": 0.5, "frozen_eval": 0.2})


# --- a scenario set can never be paired with a foreign provider -----------


def test_scenarios_without_their_provider_are_rejected():
    """Falling back to the default split's provider ran frozen_eval against the smoke catalog."""
    frozen = build_split("frozen_eval", count=10)

    with pytest.raises(ValueError, match="must be supplied together"):
        run_feed_benchmark("rule_based", scenarios=frozen.scenarios, seed=42, max_steps=6)

    with pytest.raises(ValueError, match="must be supplied together"):
        run_feed_benchmark("rule_based", provider=frozen.provider, seed=42, max_steps=6)


def test_a_feed_split_can_be_passed_directly():
    frozen = build_split("frozen_eval", count=10)
    episodes, summary = run_feed_benchmark("rule_based", split=frozen, seed=42, max_steps=6)

    assert summary["split"] == "frozen_eval"
    assert len(episodes) == 10

    pool = frozen.pool.product_ids()
    served = [
        step["served_exposure"]["product_id"]
        for episode in episodes
        for step in episode["steps"]
        if step["served_exposure"] and step["served_exposure"]["product_id"]
    ]
    assert served, "no products were served, so the guard would be vacuous"
    assert all(product_id in pool for product_id in served)


def test_a_feed_split_cannot_be_combined_with_loose_arguments():
    frozen = build_split("frozen_eval", count=5)
    with pytest.raises(ValueError, match="not both"):
        run_feed_benchmark("rule_based", split=frozen, scenarios=frozen.scenarios)


def test_env_refuses_a_scenario_from_another_split():
    smoke = build_split("smoke", count=5)
    frozen = build_split("frozen_eval", count=5)
    env = FeedControlEnv(smoke.provider, max_steps=6, seed=42)

    with pytest.raises(ValueError, match="products the provider does not own"):
        env.reset(frozen.scenarios[0], seed=42)


def test_env_refuses_a_same_id_different_attributes_catalog():
    """The subtle case: ids line up but the entities behind them are different."""
    from videoshop.data.synthetic import SyntheticCommerceConfig
    from videoshop.feed.scenarios import build_default_provider, build_smoke_scenarios

    scenarios = build_smoke_scenarios(count=3, seed=7, commerce_config=SyntheticCommerceConfig(seed=7))
    other_provider = build_default_provider(commerce_config=SyntheticCommerceConfig(seed=42))
    env = FeedControlEnv(other_provider, max_steps=6, seed=42)

    with pytest.raises(ValueError, match="disagree about"):
        env.reset(scenarios[0], seed=42)


@pytest.mark.parametrize(
    ("entity", "field", "value"),
    [
        ("products", "price", 1234.5),
        ("products", "review_risk", 0.999),
        ("products", "rating", 0.1),
        ("products", "inventory", 999999),
        ("products", "has_coupon", True),
        ("products", "tags", ["tampered"]),
        ("videos", "category", "not_a_category"),
        ("videos", "creator_type", "impostor"),
        ("videos", "objects", ["tampered"]),
    ],
)
def test_every_entity_attribute_is_part_of_the_world_check(entity, field, value):
    """Price was the only field being compared, so any other edit slipped through."""

    split = build_split("smoke", count=3)
    scenario = copy.deepcopy(split.scenarios[0])
    entity_id = next(iter(scenario.hidden_world_state[entity]))
    scenario.hidden_world_state[entity][entity_id][field] = value
    env = FeedControlEnv(split.provider, max_steps=6, seed=42)

    with pytest.raises(ValueError, match=f"disagree about .*{field}"):
        env.reset(scenario, seed=42)


def test_a_json_round_trip_is_not_mistaken_for_a_different_world():
    """Frozen splits are read back from disk, where tuples become lists."""

    split = build_split("smoke", count=3)
    scenario = copy.deepcopy(split.scenarios[0])
    scenario.hidden_world_state = json.loads(json.dumps(scenario.hidden_world_state))
    env = FeedControlEnv(split.provider, max_steps=6, seed=42)

    env.reset(scenario, seed=42)


def test_unknown_split_is_rejected():
    assert set(FEED_SPLITS) == {"smoke", "train", "frozen_eval"}
    with pytest.raises(ValueError, match="Unsupported feed split"):
        build_split("test")


def test_episode_records_the_objective_profile_and_oracle_reference(provider, scenarios):
    episodes, _summary = run_feed_benchmark(
        "rule_based", scenarios=scenarios[:5], provider=provider, seed=42
    )
    for episode in episodes:
        assert episode["objective_profile"]
        assert episode["oracle"]["gold_exposure_id"]
        assert episode["terminated_by"] in {"session_exit", "max_steps"}
