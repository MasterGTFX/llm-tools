"""Diff management utilities for tracking changes in knowledge bases."""

import difflib
from pathlib import Path
from typing import Optional, Union

from llmtools.config import DiffManagerConfig
from llmtools.utils.logging import setup_logger


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

        # Set up logging
        self.logger = setup_logger(
            __name__,
            level=self.config.logging.level,
            format_string=self.config.logging.format,
            handler_type=self.config.logging.handler_type,
        )
        self.logger.debug(
            f"DiffManager initialized with format: {self.config.diff_format}"
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
        self.logger.info(f"Creating diff between {old_label} and {new_label}")
        self.logger.debug(
            f"Old content: {len(old_content)} characters, New content: {len(new_content)} characters"
        )

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

        diff_result = "".join(diff)
        self.logger.debug(f"Generated diff: {len(diff_result)} characters")
        return diff_result

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
        self.logger.debug("Analyzing diff for change summary")
        lines = diff_content.splitlines()
        additions = sum(
            1 for line in lines if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in lines if line.startswith("-") and not line.startswith("---")
        )

        summary = {
            "additions": additions,
            "deletions": deletions,
            "total_changes": additions + deletions,
        }
        self.logger.info(
            f"Diff summary: +{additions}, -{deletions}, total: {summary['total_changes']} changes"
        )
        return summary

    def save_diff(self, diff_content: str, output_path: Union[str, Path]) -> None:
        """Save diff content to a file.

        Args:
            diff_content: The diff to save
            output_path: Path where to save the diff
        """
        path = Path(output_path)
        self.logger.info(f"Saving diff to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(diff_content, encoding="utf-8")
        self.logger.debug(f"Saved {len(diff_content)} characters to diff file")

    def load_diff(self, diff_path: Union[str, Path]) -> str:
        """Load diff content from a file.

        Args:
            diff_path: Path to the diff file

        Returns:
            Diff content as string
        """
        path = Path(diff_path)
        self.logger.debug(f"Loading diff from {path}")
        content = path.read_text(encoding="utf-8")
        self.logger.debug(f"Loaded {len(content)} characters from diff file")
        return content
