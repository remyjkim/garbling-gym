# ABOUTME: CSV exporter for data analysis
# ABOUTME: Exports round-by-round data and summary statistics

import csv
from pathlib import Path
from typing import Dict, Any, List

from .base import Exporter


class CSVExporter(Exporter):
    """Export results as CSV files for data analysis"""

    def get_extension(self) -> str:
        return ".csv"

    def export(self, results: Dict[str, Any], output_path: Path) -> Path:
        """
        Export results as CSV files.

        Creates two files:
        - {name}_rounds.csv: Round-by-round data
        - {name}_summary.csv: Summary statistics

        Args:
            results: Dictionary with 'results', 'config', 'metadata' keys
            output_path: Base path for output files

        Returns:
            Path to directory containing CSV files
        """
        # Ensure output path is a directory
        if output_path.suffix:
            output_dir = output_path.parent
            base_name = output_path.stem
        else:
            output_dir = output_path
            base_name = "export"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        run_results = self._flatten_results(results['results'])
        config = results.get('config', {})
        metadata = self._format_metadata(results.get('metadata', {}))

        # Export rounds data
        rounds_path = output_dir / f"{base_name}_rounds.csv"
        self._export_rounds(run_results, rounds_path)

        # Export summary data
        summary_path = output_dir / f"{base_name}_summary.csv"
        self._export_summary(run_results, config, metadata, summary_path)

        # Export quality stats
        quality_path = output_dir / f"{base_name}_quality.csv"
        self._export_quality_stats(run_results, quality_path)

        # Export strategy stats
        strategy_path = output_dir / f"{base_name}_strategies.csv"
        self._export_strategy_stats(run_results, strategy_path)

        return output_dir

    def _export_rounds(self, results: Dict[str, Any], output_path: Path):
        """Export round-by-round data"""
        history = results.get('history', [])

        if not history:
            return

        # Define CSV columns
        fieldnames = [
            'round',
            'quality',
            'strategy',
            'signal',
            'action',
            'sender_payoff',
            'receiver_payoff',
            'garbling_info',
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for h in history:
                writer.writerow({
                    'round': h.get('round', 0),
                    'quality': h.get('quality', ''),
                    'strategy': h.get('strategy', ''),
                    'signal': h.get('signal', ''),
                    'action': h.get('action', ''),
                    'sender_payoff': h.get('sender_payoff', 0),
                    'receiver_payoff': h.get('receiver_payoff', 0),
                    'garbling_info': h.get('garbling_info', 0),
                })

    def _export_summary(self, results: Dict[str, Any], config: Dict[str, Any],
                       metadata: Dict[str, Any], output_path: Path):
        """Export summary statistics"""
        rows = [
            ['Metric', 'Value'],
            ['Total Rounds', results['total_rounds']],
            ['Sender Total', results['sender_total']],
            ['Receiver Total', results['receiver_total']],
            ['Buy Rate', results['buy_rate']],
            ['Avg Informativeness', results['avg_informativeness']],
            ['Receiver Regret', results['receiver_regret']],
            ['Perfect Info Benchmark', results.get('perfect_info_benchmark', 0)],
            ['Social Welfare', results['sender_total'] + results['receiver_total']],
            ['', ''],
            ['Metadata', ''],
            ['Run Name', metadata.get('name', 'Unnamed')],
            ['Timestamp', metadata.get('timestamp', 'Unknown')],
            ['Duration (s)', metadata.get('duration', 0)],
            ['Git Commit', metadata.get('git_commit', 'Unknown')],
            ['Tags', ', '.join(metadata.get('tags', []))],
            ['', ''],
            ['Configuration', ''],
            ['LLM Enabled', config.get('use_llm', False)],
            ['LLM Model', config.get('llm_model', 'N/A')],
            ['Prior LOW', config.get('prior', {}).get('LOW', 0)],
            ['Prior MEDIUM', config.get('prior', {}).get('MEDIUM', 0)],
            ['Prior HIGH', config.get('prior', {}).get('HIGH', 0)],
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def _export_quality_stats(self, results: Dict[str, Any], output_path: Path):
        """Export quality distribution statistics"""
        quality_stats = results.get('quality_stats', {})

        if not quality_stats:
            return

        fieldnames = ['quality', 'count', 'bought', 'buy_rate', 'sender_total', 'receiver_total']

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for quality in ['HIGH', 'MEDIUM', 'LOW']:
                stats = quality_stats.get(quality, {})
                writer.writerow({
                    'quality': quality,
                    'count': stats.get('count', 0),
                    'bought': stats.get('bought', 0),
                    'buy_rate': stats.get('buy_rate', 0),
                    'sender_total': stats.get('sender_total', 0),
                    'receiver_total': stats.get('receiver_total', 0),
                })

    def _export_strategy_stats(self, results: Dict[str, Any], output_path: Path):
        """Export strategy usage statistics"""
        strategies = results.get('strategies_used', {})

        if not strategies:
            return

        total = results['total_rounds']

        rows = [['strategy', 'count', 'percentage']]
        for strategy, count in sorted(strategies.items(), key=lambda x: -x[1]):
            pct = count / total if total > 0 else 0
            rows.append([strategy, count, pct])

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
