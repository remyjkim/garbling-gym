"""
Integration tests for complete game flow.

Tests that agents, strategies, and game orchestration work together correctly.
"""

import pytest
from garbling_gym.core.config import GameConfig
from garbling_gym.core.game import Game
from garbling_gym.core.agents.sender import SenderAgent
from garbling_gym.core.agents.receiver import ReceiverAgent
from garbling_gym.core.types import AssetQuality, Action


class TestGameIntegration:
    """Integration tests for complete game workflow"""

    def test_game_runs_to_completion(self):
        """Test that a game runs from start to finish"""
        config = GameConfig(num_rounds=5, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        # Verify game completed all rounds
        assert results.total_rounds == 5
        assert len(results.history) == 5

        # Verify results structure
        assert isinstance(results.sender_total, (int, float))
        assert isinstance(results.receiver_total, (int, float))
        assert 0 <= results.buy_rate <= 1
        assert 0 <= results.avg_informativeness <= 1

    def test_game_with_different_round_counts(self):
        """Test games with various round counts"""
        for num_rounds in [1, 10, 50]:
            config = GameConfig(num_rounds=num_rounds, use_llm=False)
            sender = SenderAgent()
            receiver = ReceiverAgent()

            game = Game(config, sender=sender, receiver=receiver)
            results = game.play_game(verbose=False)

            assert results.total_rounds == num_rounds
            assert len(results.history) == num_rounds

    def test_single_round_workflow(self):
        """Test a single round's complete workflow"""
        config = GameConfig(num_rounds=1, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)

        # Play one round
        round_result = game.play_round()

        # Verify round structure
        assert round_result['round'] == 1
        assert round_result['quality'] in ['LOW', 'MEDIUM', 'HIGH']
        assert round_result['strategy'] in [
            'full_revelation', 'complete_noise', 'pool_low_medium',
            'pool_medium_high', 'slight_noise', 'aggressive_pooling'
        ]
        assert round_result['signal'] in ['BAD', 'NEUTRAL', 'GOOD']
        assert round_result['action'] in ['BUY', 'PASS']
        assert isinstance(round_result['sender_payoff'], (int, float))
        assert isinstance(round_result['receiver_payoff'], (int, float))
        assert 0 <= round_result['garbling_info'] <= 1

    def test_payoffs_accumulate_correctly(self):
        """Test that payoffs accumulate over rounds"""
        config = GameConfig(num_rounds=10, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        # Calculate expected totals from history
        expected_sender = sum(h['sender_payoff'] for h in results.history)
        expected_receiver = sum(h['receiver_payoff'] for h in results.history)

        assert results.sender_total == expected_sender
        assert results.receiver_total == expected_receiver

    def test_strategy_usage_tracked(self):
        """Test that strategy usage is tracked correctly"""
        config = GameConfig(num_rounds=20, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        # Verify strategy usage counts
        total_strategy_uses = sum(results.strategies_used.values())
        assert total_strategy_uses == 20

        # All strategies should be valid
        for strategy in results.strategies_used.keys():
            assert strategy in [
                'full_revelation', 'complete_noise', 'pool_low_medium',
                'pool_medium_high', 'slight_noise', 'aggressive_pooling'
            ]

    def test_quality_distribution(self):
        """Test that quality distribution matches prior"""
        config = GameConfig(
            num_rounds=100,
            use_llm=False,
            prior={
                AssetQuality.LOW: 0.3,
                AssetQuality.MEDIUM: 0.4,
                AssetQuality.HIGH: 0.3,
            }
        )
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        # Count quality occurrences
        quality_counts = {
            'LOW': sum(1 for h in results.history if h['quality'] == 'LOW'),
            'MEDIUM': sum(1 for h in results.history if h['quality'] == 'MEDIUM'),
            'HIGH': sum(1 for h in results.history if h['quality'] == 'HIGH'),
        }

        # With 100 rounds, should be roughly distributed according to prior
        # Allow generous margin due to randomness
        assert 15 <= quality_counts['LOW'] <= 45  # 30% ± 15%
        assert 25 <= quality_counts['MEDIUM'] <= 55  # 40% ± 15%
        assert 15 <= quality_counts['HIGH'] <= 45  # 30% ± 15%

    def test_receiver_regret_calculation(self):
        """Test that receiver regret is calculated correctly"""
        config = GameConfig(num_rounds=20, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        # Calculate expected perfect info benchmark
        perfect_info_expected = 0
        for h in results.history:
            if h['quality'] == 'MEDIUM':
                perfect_info_expected += 5
            elif h['quality'] == 'HIGH':
                perfect_info_expected += 20

        assert results.perfect_info_benchmark == perfect_info_expected
        assert results.receiver_regret == perfect_info_expected - results.receiver_total

    def test_informativeness_in_range(self):
        """Test that average informativeness is in valid range"""
        config = GameConfig(num_rounds=20, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        assert 0 <= results.avg_informativeness <= 1

    def test_buy_rate_in_range(self):
        """Test that buy rate is in valid range"""
        config = GameConfig(num_rounds=20, use_llm=False)
        sender = SenderAgent()
        receiver = ReceiverAgent()

        game = Game(config, sender=sender, receiver=receiver)
        results = game.play_game(verbose=False)

        assert 0 <= results.buy_rate <= 1

        # Verify buy rate calculation
        buys = sum(1 for h in results.history if h['action'] == 'BUY')
        expected_buy_rate = buys / len(results.history)
        assert results.buy_rate == expected_buy_rate

    def test_deterministic_with_seed(self):
        """Test that game is deterministic with same random seed"""
        import random
        import numpy as np

        # Run game twice with same seed
        results_list = []
        for _ in range(2):
            random.seed(42)
            np.random.seed(42)

            config = GameConfig(num_rounds=10, use_llm=False)
            sender = SenderAgent()
            receiver = ReceiverAgent()

            game = Game(config, sender=sender, receiver=receiver)
            results = game.play_game(verbose=False)
            results_list.append(results)

        # Results should be identical
        assert results_list[0].sender_total == results_list[1].sender_total
        assert results_list[0].receiver_total == results_list[1].receiver_total
        assert results_list[0].buy_rate == results_list[1].buy_rate

        # History should match round by round
        for i in range(10):
            h1 = results_list[0].history[i]
            h2 = results_list[1].history[i]

            assert h1['quality'] == h2['quality']
            assert h1['strategy'] == h2['strategy']
            assert h1['signal'] == h2['signal']
            assert h1['action'] == h2['action']
