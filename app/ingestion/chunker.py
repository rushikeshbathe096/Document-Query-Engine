import re
import uuid
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class Tokenizer(ABC):
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass


class TiktokenTokenizer(Tokenizer):
    def __init__(self, model_name: str = "cl100k_base"):
        import tiktoken
        self.encoder = tiktoken.get_encoding(model_name)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))


class Chunker:
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        target_tokens: int = 200,
        max_tokens: int = 300,
        overlap_tokens: int = 50,
    ):
        self.tokenizer = tokenizer or TiktokenTokenizer()
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def _split_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _split_long_sentence(self, sentence: str) -> List[str]:
        if self.tokenizer.count_tokens(sentence) <= self.max_tokens:
            return [sentence]

        words = sentence.split()
        parts = []
        current = []
        current_tokens = 0

        for word in words:
            wt = self.tokenizer.count_tokens(word if not current else " " + word)
            if current and current_tokens + wt > self.max_tokens:
                parts.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(word)
            current_tokens += wt

        if current:
            parts.append(" ".join(current))

        return parts

    def _clamp_overlap(self, sentences: List[str]) -> List[str]:
        overlap = sentences[:] if sentences else []
        while overlap and sum(self.tokenizer.count_tokens(s) for s in overlap) > self.overlap_tokens:
            overlap = overlap[1:]
        return overlap

    def _create_chunk(self, text: str, page_number: int, position: int) -> Dict[str, Any]:
        identifier = f"page_{page_number}_pos_{position}_{text}"
        return {
            "chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, identifier)),
            "text": text,
            "page_number": page_number,
            "position": position,
        }

    def chunk_page(self, page_number: int, text: str) -> List[Dict[str, Any]]:
        raw = self._split_sentences(text)
        sentences = []
        for s in raw:
            sentences.extend(self._split_long_sentence(s))

        chunks = []
        buffer = []
        buffer_tokens = 0
        position = 0

        for sentence in sentences:
            st = self.tokenizer.count_tokens(sentence)

            if buffer and buffer_tokens + st > self.max_tokens:
                chunk_text = " ".join(buffer)
                chunks.append(self._create_chunk(chunk_text, page_number, position))
                position += 1
                buffer = self._clamp_overlap(buffer)
                buffer_tokens = sum(self.tokenizer.count_tokens(s) for s in buffer)

            buffer.append(sentence)
            buffer_tokens += st

            if buffer_tokens >= self.target_tokens:
                chunk_text = " ".join(buffer)
                chunks.append(self._create_chunk(chunk_text, page_number, position))
                position += 1
                buffer = self._clamp_overlap(buffer)
                buffer_tokens = sum(self.tokenizer.count_tokens(s) for s in buffer)

        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(self._create_chunk(chunk_text, page_number, position))

        return chunks

    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for p in pages:
            if "page_number" not in p:
                continue
            out.extend(self.chunk_page(p["page_number"], p.get("text", "")))

        chunk_sizes = [self.tokenizer.count_tokens(chunk["text"]) for chunk in out]
        average_size = round(sum(chunk_sizes) / len(chunk_sizes), 2) if chunk_sizes else 0.0
        logger.debug(
            "Chunking summary: total_chunks=%s average_chunk_size=%s",
            len(out),
            average_size,
        )
        return out


def get_chunker() -> Chunker:
    return Chunker()