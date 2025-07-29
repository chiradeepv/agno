import os
from typing import Dict, List, Optional

from agno.knowledge.reader.base import Reader


class ReaderFactory:
    """Factory for creating and managing document readers with lazy loading."""

    # Cache for instantiated readers
    _reader_cache: Dict[str, Reader] = {}

    @classmethod
    def _get_pdf_reader(cls, **kwargs) -> Reader:
        """Get PDF reader instance."""
        from agno.knowledge.reader.pdf_reader import PDFReader

        config = {"chunk": True, "chunk_size": 100}
        config.update(kwargs)
        return PDFReader(**config)

    @classmethod
    def _get_csv_reader(cls, **kwargs) -> Reader:
        """Get CSV reader instance."""
        from agno.knowledge.reader.csv_reader import CSVReader

        config = {"name": "CSV Reader", "description": "Reads CSV files"}
        config.update(kwargs)
        return CSVReader(**config)

    @classmethod
    def _get_docx_reader(cls, **kwargs) -> Reader:
        """Get Docx reader instance."""
        from agno.knowledge.reader.docx_reader import DocxReader

        config = {"name": "Docx Reader", "description": "Reads Docx files"}
        config.update(kwargs)
        return DocxReader(**config)

    @classmethod
    def _get_json_reader(cls, **kwargs) -> Reader:
        """Get JSON reader instance."""
        from agno.knowledge.reader.json_reader import JSONReader

        config = {"name": "JSON Reader", "description": "Reads JSON files"}
        config.update(kwargs)
        return JSONReader(**config)

    @classmethod
    def _get_markdown_reader(cls, **kwargs) -> Reader:
        """Get Markdown reader instance."""
        from agno.knowledge.reader.markdown_reader import MarkdownReader

        config = {"name": "Markdown Reader", "description": "Reads Markdown files"}
        config.update(kwargs)
        return MarkdownReader(**config)

    @classmethod
    def _get_text_reader(cls, **kwargs) -> Reader:
        """Get Text reader instance."""
        from agno.knowledge.reader.text_reader import TextReader

        config = {"name": "Text Reader", "description": "Reads Text files"}
        config.update(kwargs)
        return TextReader(**config)

    @classmethod
    def _get_url_reader(cls, **kwargs) -> Reader:
        """Get URL reader instance."""
        from agno.knowledge.reader.url_reader import URLReader

        config = {"name": "URL Reader", "description": "Reads URLs"}
        config.update(kwargs)
        return URLReader(**config)

    @classmethod
    def _get_website_reader(cls, **kwargs) -> Reader:
        """Get Website reader instance."""
        from agno.knowledge.reader.website_reader import WebsiteReader

        config = {"name": "Website Reader", "description": "Reads Website files"}
        config.update(kwargs)
        return WebsiteReader(**config)

    @classmethod
    def _get_firecrawl_reader(cls, **kwargs) -> Reader:
        """Get Firecrawl reader instance."""
        from agno.knowledge.reader.firecrawl_reader import FirecrawlReader

        config = {
            "api_key": kwargs.get("api_key") or os.getenv("FIRECRAWL_API_KEY"),
            "mode": "crawl",
            "name": "Firecrawl Reader",
            "description": "Crawls websites",
        }
        config.update(kwargs)
        return FirecrawlReader(**config)

    @classmethod
    def _get_youtube_reader(cls, **kwargs) -> Reader:
        """Get YouTube reader instance."""
        from agno.knowledge.reader.youtube_reader import YouTubeReader

        config = {"name": "YouTube Reader", "description": "Reads YouTube videos"}
        config.update(kwargs)
        return YouTubeReader(**config)

    @classmethod
    def _get_pdf_url_reader(cls, **kwargs) -> Reader:
        """Get PDF URL reader instance."""
        from agno.knowledge.reader.pdf_reader import PDFUrlReader

        config = {"name": "PDF URL Reader", "description": "Reads PDF URLs"}
        config.update(kwargs)
        return PDFUrlReader(**config)

    @classmethod
    def _get_csv_url_reader(cls, **kwargs) -> Reader:
        """Get CSV URL reader instance."""
        from agno.knowledge.reader.csv_reader import CSVUrlReader

        config = {"name": "CSV URL Reader", "description": "Reads CSV URLs"}
        config.update(kwargs)
        return CSVUrlReader(**config)

    @classmethod
    def _get_reader_method(cls, reader_key: str):
        """Get the appropriate reader method for the given key."""
        method_name = f"_get_{reader_key}_reader"
        if not hasattr(cls, method_name):
            raise ValueError(f"Unknown reader: {reader_key}")
        return getattr(cls, method_name)

    @classmethod
    def create_reader(cls, reader_key: str, **kwargs) -> Reader:
        """Create a reader instance with the given key and optional overrides."""
        if reader_key in cls._reader_cache:
            return cls._reader_cache[reader_key]

        # Get the reader method and create the instance
        reader_method = cls._get_reader_method(reader_key)
        reader = reader_method(**kwargs)

        # Cache the reader
        cls._reader_cache[reader_key] = reader

        return reader

    @classmethod
    def get_reader_for_extension(cls, extension: str) -> Reader:
        """Get the appropriate reader for a file extension."""
        extension = extension.lower()

        if extension in [".pdf", "application/pdf"]:
            return cls.create_reader("pdf")
        elif extension == ".csv":
            return cls.create_reader("csv")
        elif extension in [".docx", ".doc"]:
            return cls.create_reader("docx")
        elif extension == ".json":
            return cls.create_reader("json")
        elif extension in [".md", ".markdown"]:
            return cls.create_reader("markdown")
        elif extension in [".txt", ".text"]:
            return cls.create_reader("text")
        else:
            # Default to text reader for unknown extensions
            return cls.create_reader("text")

    @classmethod
    def get_reader_for_url(cls, url: str) -> Reader:
        """Get the appropriate reader for a URL."""
        url_lower = url.lower()

        # Check for YouTube URLs
        if any(domain in url_lower for domain in ["youtube.com", "youtu.be"]):
            return cls.create_reader("youtube")

        # Default to URL reader
        return cls.create_reader("url")

    @classmethod
    def get_reader_for_url_file(cls, extension: str) -> Reader:
        """Get the appropriate reader for a URL file extension."""
        extension = extension.lower()

        if extension == ".pdf":
            return cls.create_reader("pdf_url")
        elif extension == ".csv":
            return cls.create_reader("csv_url")
        else:
            return cls.create_reader("url")

    @classmethod
    def get_all_reader_keys(cls) -> List[str]:
        """Get all available reader keys."""
        # Extract reader keys from method names
        reader_keys = []
        for attr_name in dir(cls):
            if attr_name.startswith("_get_") and attr_name.endswith("_reader"):
                reader_key = attr_name[5:-7]  # Remove "_get_" prefix and "_reader" suffix
                reader_keys.append(reader_key)
        return reader_keys

    @classmethod
    def get_reader_info(cls, reader_key: str) -> Dict:
        """Get information about a reader without instantiating it."""
        # Try to create the reader to get its info, but don't cache it
        try:
            reader_method = cls._get_reader_method(reader_key)
            reader = reader_method()
            return {
                "key": reader_key,
                "name": getattr(reader, "name", reader_key.title()),
                "description": getattr(reader, "description", f"Reads {reader_key} files"),
            }
        except Exception:
            raise ValueError(f"Unknown reader: {reader_key}")

    @classmethod
    def get_all_readers_info(cls) -> List[Dict]:
        """Get information about all available readers."""
        return [cls.get_reader_info(key) for key in cls.get_all_reader_keys()]

    @classmethod
    def create_all_readers(cls) -> Dict[str, Reader]:
        """Create all readers and return them as a dictionary."""
        readers = {}
        for reader_key in cls.get_all_reader_keys():
            readers[reader_key] = cls.create_reader(reader_key)
        return readers

    @classmethod
    def clear_cache(cls):
        """Clear the reader cache."""
        cls._reader_cache.clear()

    @classmethod
    def register_reader(
        cls,
        key: str,
        reader_method,
        name: str,
        description: str,
        extensions: Optional[List[str]] = None,
    ):
        """Register a new reader type."""
        # Add the reader method to the class
        method_name = f"_get_{key}_reader"
        setattr(cls, method_name, classmethod(reader_method))
