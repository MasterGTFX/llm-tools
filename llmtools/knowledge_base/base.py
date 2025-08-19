"""Knowledge Base implementation for building and maintaining versioned document knowledge bases."""

import json
from pathlib import Path
from typing import Any, Optional, Union

from llmtools.config import KnowledgeBaseConfig, LLMConfig
from llmtools.interfaces.llm import LLMInterface
from llmtools.utils.diff_manager import DiffManager
from llmtools.utils.logging import setup_logger


class KnowledgeBase:
    """Build and iteratively update a knowledge base from a set of documents.

    Supports versioning with incremental updates stored in .history/ directories.
    Uses structured LLM output to manage diffs and track updates over time.
    """

    def __init__(
        self,
        config: Optional[Union[dict[str, Any], KnowledgeBaseConfig]] = None,
        instruction: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
        init: Optional[str] = None,
        llm_provider: Optional[LLMInterface] = None,
    ):
        """Initialize KnowledgeBase.

        Args:
            config: Configuration dictionary or KnowledgeBaseConfig object
            instruction: Custom instruction for knowledge base creation
            output_dir: Directory to store knowledge base and history
            init: Initial knowledge base content
            llm_provider: LLM provider instance (if None, will create from config)
        """
        # Handle config initialization
        if isinstance(config, dict):
            # Convert dict config to proper config objects
            llm_config = LLMConfig(**config)
            self.config = KnowledgeBaseConfig(
                llm=llm_config,
                instruction="Create a comprehensive knowledge base containing all useful information",
                output_dir=None,
                history_dir=".history",
                chunk_size=4000,
                chunk_overlap=200,
                max_versions=10,
            )
        elif isinstance(config, KnowledgeBaseConfig):
            self.config = config
        else:
            # Default config
            default_llm = LLMConfig(
                provider="openai",
                model=None,
                api_key=None,
                base_url=None,
                temperature=0.7,
                max_tokens=None,
                timeout=30,
            )
            self.config = KnowledgeBaseConfig(
                llm=default_llm,
                instruction="Create a comprehensive knowledge base containing all useful information",
                output_dir=None,
                history_dir=".history",
                chunk_size=4000,
                chunk_overlap=200,
                max_versions=10,
            )

        # Override specific fields if provided
        if instruction:
            self.config.instruction = instruction
        if output_dir:
            self.config.output_dir = Path(output_dir)

        self.llm_provider = llm_provider
        self.diff_manager = DiffManager()

        # Set up logging
        self.logger = setup_logger(
            __name__,
            level=self.config.logging.level,
            format_string=self.config.logging.format,
            handler_type=self.config.logging.handler_type,
        )

        # Document storage
        self.documents: list[str] = []
        self.initial_kb = init
        self.versions: list[str] = []

        # Set up output directory
        if self.config.output_dir:
            self.output_dir = Path(self.config.output_dir)
            self.history_dir = self.output_dir / self.config.history_dir
        else:
            self.output_dir = Path.cwd() / "knowledge_base"
            self.history_dir = self.output_dir / self.config.history_dir

        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(
            f"Knowledge base initialized with output directory: {self.output_dir}"
        )

    def add_documents(self, document_paths: list[Union[str, Path]]) -> None:
        """Add documents to be processed into the knowledge base.

        Args:
            document_paths: List of paths to documents to include
        """
        self.logger.info(f"Adding {len(document_paths)} documents to knowledge base")
        for doc_path in document_paths:
            path = Path(doc_path)
            if path.exists():
                content = path.read_text(encoding="utf-8")
                self.documents.append(content)
                self.logger.debug(
                    f"Added document: {doc_path} ({len(content)} characters)"
                )
            else:
                self.logger.error(f"Document not found: {doc_path}")
                raise FileNotFoundError(f"Document not found: {doc_path}")
        self.logger.info(f"Successfully added {len(document_paths)} documents")

    def add_document_content(self, content: str) -> None:
        """Add document content directly (without file path).

        Args:
            content: Document content as string
        """
        self.documents.append(content)
        self.logger.debug(f"Added document content ({len(content)} characters)")

    def process(self) -> list[str]:
        """Process documents and return list of versions ([-1] is the latest).

        Returns:
            List of knowledge base versions, with latest at index -1
        """
        if not self.llm_provider:
            self.logger.error("LLM provider not configured")
            raise ValueError("LLM provider not configured. Cannot process documents.")

        self.logger.info(
            f"Starting knowledge base processing with {len(self.documents)} documents"
        )

        # Combine all documents
        combined_content = "\n\n---\n\n".join(self.documents)
        self.logger.debug(
            f"Combined document content: {len(combined_content)} characters"
        )

        # Create prompt for knowledge base generation
        system_prompt = (
            "You are an expert knowledge base creator. "
            "Create comprehensive, well-structured knowledge bases from provided documents."
        )

        if self.initial_kb:
            prompt = f"""
            Initial knowledge base:
            {self.initial_kb}

            New documents to integrate:
            {combined_content}

            Instructions: {self.config.instruction}

            Please create an updated knowledge base that integrates the new information.
            """
        else:
            prompt = f"""
            Documents to process:
            {combined_content}

            Instructions: {self.config.instruction}

            Please create a comprehensive knowledge base from these documents.
            """

        try:
            # Generate knowledge base using LLM
            self.logger.info("Generating knowledge base content using LLM")
            kb_content = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
            self.logger.debug(
                f"Generated knowledge base content: {len(kb_content)} characters"
            )

            # Save version
            version_num = len(self.versions)
            version_file = self.history_dir / f"v{version_num:03d}.txt"
            version_file.write_text(kb_content, encoding="utf-8")
            self.logger.info(f"Saved version {version_num} to {version_file}")

            # Create diff if there's a previous version
            if self.versions:
                self.logger.debug(
                    f"Creating diff from v{version_num - 1:03d} to v{version_num:03d}"
                )
                previous_content = self.versions[-1]
                diff_content = self.diff_manager.create_diff(
                    previous_content,
                    kb_content,
                    f"v{version_num - 1:03d}",
                    f"v{version_num:03d}",
                )
                diff_file = self.history_dir / f"v{version_num:03d}.diff"
                self.diff_manager.save_diff(diff_content, diff_file)
                self.logger.debug(f"Saved diff to {diff_file}")

            self.versions.append(kb_content)

            # Save metadata
            self._save_metadata()
            self.logger.info(
                f"Knowledge base processing completed successfully. Total versions: {len(self.versions)}"
            )

            return self.versions.copy()

        except Exception as e:
            self.logger.error(f"Failed to process documents: {e}")
            raise RuntimeError(f"Failed to process documents: {e}") from e

    def query(self, question: str) -> str:
        """Query the knowledge base with a question.

        Args:
            question: Question to ask about the knowledge base

        Returns:
            Answer based on the latest knowledge base version
        """
        self.logger.info(
            f"Querying knowledge base: '{question[:50]}{'...' if len(question) > 50 else ''}'"
        )

        if not self.versions:
            self.logger.error("No knowledge base versions available")
            raise ValueError(
                "No knowledge base versions available. Run process() first."
            )

        if not self.llm_provider:
            self.logger.error("LLM provider not configured for querying")
            raise ValueError(
                "LLM provider not configured. Cannot query knowledge base."
            )

        latest_kb = self.versions[-1]

        system_prompt = (
            "You are a knowledgeable assistant. Answer questions based solely on "
            "the provided knowledge base content. Be accurate and cite relevant sections."
        )

        prompt = f"""
        Knowledge Base:
        {latest_kb}

        Question: {question}

        Please provide a comprehensive answer based on the knowledge base above.
        """

        try:
            response = self.llm_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=self.config.llm.temperature,
            )
            self.logger.info("Query completed successfully")
            self.logger.debug(f"Query response length: {len(response)} characters")
            return response
        except Exception as e:
            self.logger.error(f"Failed to query knowledge base: {e}")
            raise RuntimeError(f"Failed to query knowledge base: {e}") from e

    def get_version_history(self) -> list[dict[str, Any]]:
        """Get metadata about all versions.

        Returns:
            List of version metadata dictionaries
        """
        metadata_file = self.history_dir / "metadata.json"
        if metadata_file.exists():
            content = metadata_file.read_text(encoding="utf-8")
            metadata: list[dict[str, Any]] = json.loads(content)
            return metadata
        return []

    def _save_metadata(self) -> None:
        """Save metadata about versions to JSON file."""
        metadata = []
        for i, version in enumerate(self.versions):
            version_info = {
                "version": i,
                "file": f"v{i:03d}.txt",
                "length": len(version),
                "document_count": len(self.documents),
            }

            # Add diff info if available
            if i > 0:
                diff_file = self.history_dir / f"v{i:03d}.diff"
                if diff_file.exists():
                    diff_content = diff_file.read_text(encoding="utf-8")
                    version_info["diff"] = self.diff_manager.get_change_summary(
                        diff_content
                    )

            metadata.append(version_info)

        metadata_file = self.history_dir / "metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
