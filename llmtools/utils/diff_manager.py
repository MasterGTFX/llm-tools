"""Diff management utilities for tracking changes in knowledge bases."""

import difflib
from pathlib import Path
from typing import Optional, Union

from llmtools.config import DiffManagerConfig


class DiffManager:
    """Manages diffs between versions of text content."""

    def __init__(self, config: Optional[DiffManagerConfig] = None):
        """Initialize DiffManager with configuration.

        Args:
            config: Configuration for diff operations
        """
        self.config = config or DiffManagerConfig(
            diff_format="unified", context_lines=3, ignore_whitespace=False
        )

    def create_diff(
        self,
        old_content: str,
        new_content: str,
        old_label: str = "old",
        new_label: str = "new",
    ) -> str:
        """Create a diff between two text contents.

        Args:
            old_content: Original content
            new_content: Updated content
            old_label: Label for the old content in diff
            new_label: Label for the new content in diff

        Returns:
            Formatted diff string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        if self.config.diff_format == "unified":
            diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=old_label,
                tofile=new_label,
                n=self.config.context_lines,
            )
        elif self.config.diff_format == "context":
            diff = difflib.context_diff(
                old_lines,
                new_lines,
                fromfile=old_label,
                tofile=new_label,
                n=self.config.context_lines,
            )
        else:  # ndiff
            diff = difflib.ndiff(old_lines, new_lines)

        return "".join(diff)

    def apply_diff(self, original: str, diff_content: str) -> str:
        """Apply a diff to original content (basic implementation).

        Args:
            original: Original content
            diff_content: Diff to apply

        Returns:
            Modified content

        Note:
            This is a simplified implementation. For production use,
            consider using more robust diff application libraries.
        """
        # This is a placeholder implementation
        # In practice, you'd want to parse the diff format properly
        return original  # TODO: Implement proper diff application

    def get_change_summary(self, diff_content: str) -> dict[str, int]:
        """Get summary statistics about changes in a diff.

        Args:
            diff_content: The diff content to analyze

        Returns:
            Dictionary with change statistics
        """
        lines = diff_content.splitlines()
        additions = sum(
            1 for line in lines if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in lines if line.startswith("-") and not line.startswith("---")
        )

        return {
            "additions": additions,
            "deletions": deletions,
            "total_changes": additions + deletions,
        }

    def save_diff(self, diff_content: str, output_path: Union[str, Path]) -> None:
        """Save diff content to a file.

        Args:
            diff_content: The diff to save
            output_path: Path where to save the diff
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(diff_content, encoding="utf-8")

    def load_diff(self, diff_path: Union[str, Path]) -> str:
        """Load diff content from a file.

        Args:
            diff_path: Path to the diff file

        Returns:
            Diff content as string
        """
        return Path(diff_path).read_text(encoding="utf-8")
