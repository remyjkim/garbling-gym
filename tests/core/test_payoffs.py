"""
Unit tests for payoff structures.

Tests payoff computation for different action-quality pairs.
"""

import pytest
from garbling_gym.core.payoffs import PayoffStructure
from garbling_gym.core.types import Action, AssetQuality


class TestPayoffStructure:
    """Test the PayoffStructure class"""

    def test_default_payoffs(self):
        """Test that default payoffs are set correctly"""
        payoffs = PayoffStructure()

        # BUY + LOW: sender gains, receiver loses
        sender, receiver = payoffs.get_payoffs(Action.BUY, AssetQuality.LOW)
        assert sender == 10
        assert receiver == -15

        # BUY + MEDIUM: both gain
        sender, receiver = payoffs.get_payoffs(Action.BUY, AssetQuality.MEDIUM)
        assert sender == 10
        assert receiver == 5

        # BUY + HIGH: both gain significantly
        sender, receiver = payoffs.get_payoffs(Action.BUY, AssetQuality.HIGH)
        assert sender == 10
        assert receiver == 20

        # PASS: always (0, 0)
        for quality in AssetQuality:
            sender, receiver = payoffs.get_payoffs(Action.PASS, quality)
            assert sender == 0
            assert receiver == 0

    def test_sender_always_gains_from_buy(self):
        """Test that sender always benefits when receiver buys"""
        payoffs = PayoffStructure()

        for quality in AssetQuality:
            sender, _ = payoffs.get_payoffs(Action.BUY, quality)
            assert sender == 10

    def test_receiver_payoffs_increase_with_quality(self):
        """Test that receiver payoffs increase with asset quality"""
        payoffs = PayoffStructure()

        _, low_payoff = payoffs.get_payoffs(Action.BUY, AssetQuality.LOW)
        _, medium_payoff = payoffs.get_payoffs(Action.BUY, AssetQuality.MEDIUM)
        _, high_payoff = payoffs.get_payoffs(Action.BUY, AssetQuality.HIGH)

        assert low_payoff < medium_payoff < high_payoff

    def test_pass_always_safe(self):
        """Test that PASS always gives zero payoff"""
        payoffs = PayoffStructure()

        for quality in AssetQuality:
            sender, receiver = payoffs.get_payoffs(Action.PASS, quality)
            assert sender == 0
            assert receiver == 0

    def test_custom_payoffs(self):
        """Test creating custom payoff structure"""
        custom_payoffs = PayoffStructure(
            payoffs={
                Action.BUY: {
                    AssetQuality.LOW: (5, -10),
                    AssetQuality.MEDIUM: (5, 10),
                    AssetQuality.HIGH: (5, 30),
                },
                Action.PASS: {
                    AssetQuality.LOW: (0, 0),
                    AssetQuality.MEDIUM: (0, 0),
                    AssetQuality.HIGH: (0, 0),
                }
            }
        )

        sender, receiver = custom_payoffs.get_payoffs(Action.BUY, AssetQuality.HIGH)
        assert sender == 5
        assert receiver == 30
