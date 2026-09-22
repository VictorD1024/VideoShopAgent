from __future__ import annotations

import pytest

from videoshop.feed import BaseScores, Eligibility, ExposureCandidate


def organic(**overrides) -> ExposureCandidate:
    payload = {"exposure_id": "E1", "video_id": "V1", "source_type": "organic"}
    payload.update(overrides)
    return ExposureCandidate(**payload)


def seller(**overrides) -> ExposureCandidate:
    payload = {
        "exposure_id": "E2",
        "video_id": "V2",
        "source_type": "seller",
        "product_id": "P1",
        "treatment": "product_anchor",
    }
    payload.update(overrides)
    return ExposureCandidate(**payload)


def test_organic_candidate_defaults_to_no_product_and_no_treatment():
    candidate = organic()
    assert candidate.is_organic
    assert not candidate.is_commercial
    assert candidate.product_id is None
    assert candidate.treatment == "none"
    assert candidate.placement == "for_you"


def test_commercial_source_types_are_commercial():
    for source_type in ("seller", "affiliate", "ad"):
        candidate = seller(source_type=source_type)
        assert candidate.is_commercial
        assert not candidate.is_organic


def test_organic_candidate_rejects_product_binding():
    with pytest.raises(ValueError, match="organic exposures must not carry a product_id"):
        organic(product_id="P1")


def test_organic_candidate_rejects_treatment():
    with pytest.raises(ValueError, match="organic exposures must use treatment=none"):
        ExposureCandidate(
            exposure_id="E1",
            video_id="V1",
            source_type="organic",
            treatment="coupon",
            product_id="P1",
        )


def test_commercial_candidate_with_a_treatment_requires_a_product():
    with pytest.raises(ValueError, match="requires a product_id"):
        ExposureCandidate(
            exposure_id="E2", video_id="V2", source_type="seller", treatment="product_anchor"
        )


def test_commercial_supply_may_carry_no_product_treatment():
    """An ad whose product treatment was declined is still ad supply, not organic."""
    candidate = ExposureCandidate(exposure_id="E2", video_id="V2", source_type="ad")

    assert candidate.treatment == "none"
    assert candidate.product_id is None
    assert candidate.is_commercial
    assert not candidate.is_organic
    assert not candidate.sells_a_product


def test_untreated_commercial_exposure_still_may_not_carry_a_product():
    with pytest.raises(ValueError, match="treatment=none exposures must not carry a product_id"):
        ExposureCandidate(exposure_id="E2", video_id="V2", source_type="ad", product_id="P1")


def test_sells_a_product_separates_supply_from_treatment():
    treated = ExposureCandidate(
        exposure_id="E3", video_id="V3", source_type="ad", treatment="coupon", product_id="P1"
    )
    organic = ExposureCandidate(exposure_id="E4", video_id="V4", source_type="organic")

    assert treated.sells_a_product and treated.is_commercial
    assert not organic.sells_a_product and not organic.is_commercial


def test_coupon_treatment_requires_product():
    with pytest.raises(ValueError, match="requires a product_id for treatment=coupon"):
        ExposureCandidate(exposure_id="E3", video_id="V3", source_type="ad", treatment="coupon")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("source_type", "sponsored", "Unknown source_type"),
        ("treatment", "flash_sale", "Unknown treatment"),
        ("placement", "inbox", "Unknown placement"),
    ],
)
def test_closed_vocabularies_are_enforced(field, value, message):
    with pytest.raises(ValueError, match=message):
        seller(**{field: value})


def test_base_scores_reject_out_of_range_probabilities():
    with pytest.raises(ValueError, match="skip_probability must be within"):
        BaseScores(skip_probability=1.4)
    with pytest.raises(ValueError, match="expected_watch_time must be non-negative"):
        BaseScores(expected_watch_time=-1.0)


def test_eligibility_reports_every_blocking_reason():
    eligibility = Eligibility(in_stock=False, policy_compliant=False, extra_blocks=("region_blocked",))
    assert not eligibility.is_eligible
    assert eligibility.blocking_reasons() == ("in_stock", "policy_compliant", "region_blocked")


def test_eligibility_default_is_servable():
    assert Eligibility().is_eligible
    assert Eligibility().blocking_reasons() == ()


def test_candidate_round_trips_through_dict():
    candidate = seller(
        treatment="coupon",
        placement="shop_tab",
        base_scores=BaseScores(
            expected_watch_time=12.5,
            skip_probability=0.3,
            product_click_probability=0.14,
            purchase_probability=0.05,
            expected_net_gmv=18.4,
            refund_probability=0.07,
        ),
        eligibility=Eligibility(coupon_valid=False),
    )
    payload = candidate.to_dict()
    restored = ExposureCandidate.from_dict(payload)

    assert restored == candidate
    assert restored.to_dict() == payload
    assert restored.base_scores.expected_net_gmv == 18.4
    assert restored.eligibility.blocking_reasons() == ("coupon_valid",)


def test_from_dict_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown ExposureCandidate fields"):
        ExposureCandidate.from_dict({**organic().to_dict(), "gold": True})


def test_candidates_are_hashable_and_immutable():
    candidate = organic()
    assert {candidate}
    with pytest.raises(Exception):
        candidate.video_id = "V9"
