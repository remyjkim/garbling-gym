# ABOUTME: Base exporter class for export implementations
# ABOUTME: Defines common interface for all export formats

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class Exporter(ABC):
    """Base class for all exporters"""

    @abstractmethod
    def export(self, results: Dict[str, Any], output_path: Path) -> Path:
        """
        Export results to file.

        Args:
            results: Dictionary containing run results
            output_path: Path to write output file

        Returns:
            Path to created file
        """
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Get the file extension for this export format"""
        pass

    def _flatten_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested results structure for easier access.

        Handles both direct results and results with 'summary' key.
        """
        if 'summary' in results:
            flattened = results['summary'].copy()
            flattened['history'] = results.get('history', [])
            return flattened
        return results

    def _format_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Format metadata for display"""
        return {
            'timestamp': metadata.get('timestamp', 'Unknown'),
            'duration': metadata.get('duration', 0),
            'git_commit': metadata.get('git_commit', 'Unknown'),
            'name': metadata.get('name', 'Unnamed'),
            'tags': metadata.get('tags', []),
        }
