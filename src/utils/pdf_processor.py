"""Utility functions for PDF processing and text handling."""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import lru_cache
import warnings
import re

# Fast PDF processing with PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    
# Fallback options
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging
import asyncio
from asyncio import Queue as AsyncQueue
import threading
from dataclasses import dataclass
from enum import Enum

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pdfminer")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handle PDF processing and text extraction with optimizations."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 300, use_cache: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_cache = use_cache
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self._text_cache = {}  # Cache for extracted text
        self._document_cache = {}  # Cache for processed documents
        
        # Determine best extraction method
        self.extraction_method = self._determine_best_method()
        logger.info(f"PDFProcessor initialized with method: {self.extraction_method}")

    def _extract_text_from_dict(self, text_dict: dict) -> str:
        """
        Extracts text from a PyMuPDF text dict structure.
        """
        if not text_dict or "blocks" not in text_dict:
            return ""
        lines = []
        for block in text_dict["blocks"]:
            if block.get("type") == 0:  # text block
                for line in block.get("lines", []):
                    line_text = "".join([span.get("text", "") for span in line.get("spans", [])])
                    if line_text:
                        lines.append(line_text)
        return "\n".join(lines)
    
    def _determine_best_method(self) -> str:
        """Determine the best PDF extraction method available."""
        if PYMUPDF_AVAILABLE:
            return "pymupdf"
        elif PDFPLUMBER_AVAILABLE:
            return "pdfplumber"
        elif PYPDF2_AVAILABLE:
            return "pypdf2"
        else:
            raise ImportError("No PDF processing library available. Install PyMuPDF, pdfplumber, or PyPDF2.")
    
    @lru_cache(maxsize=32)
    def _get_file_info(self, file_path: str) -> Tuple[str, int, float]:
        """Get file hash, size, and modification time for caching."""
        stat = os.stat(file_path)
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read first and last 8KB for faster hashing
            hash_md5.update(f.read(8192))
            f.seek(-8192, 2)
            try:
                hash_md5.update(f.read(8192))
            except:
                pass  # File smaller than 8KB
        return hash_md5.hexdigest(), stat.st_size, stat.st_mtime
    
    def extract_text_pymupdf(self, file_path: str) -> str:
        """Extract text using PyMuPDF (fastest and most robust)."""
        try:
            start_time = time.time()
            text = ""
            
            # Open document with PyMuPDF
            doc = fitz.open(file_path)
            
            # Process pages efficiently
            if len(doc) > 20:  # Large document - use parallel processing
                total_pages = len(doc)
                doc.close()  # Close the document before parallel processing
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    for page_num in range(total_pages):
                        future = executor.submit(self._extract_page_pymupdf_safe, file_path, page_num)
                        futures.append(future)
                    
                    for future in as_completed(futures):
                        try:
                            page_text = future.result()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as e:
                            logger.warning(f"Error extracting page: {e}")
                            
                # Skip the normal doc.close() since we already closed it
                elapsed = time.time() - start_time
                logger.info(f"PyMuPDF extraction took {elapsed:.2f}s for {total_pages} pages")
                return text.strip()
            else:
                # Sequential processing for smaller documents
                total_pages = len(doc)
                for page_num in range(total_pages):
                    try:
                        page = doc.load_page(page_num)
                        # Try different PyMuPDF text extraction methods for compatibility
                        page_text = ""
                        try:
                            # Primary method: get_text() - works in PyMuPDF 1.26.3+
                            page_text = getattr(page, 'get_text', lambda: "")()
                        except (TypeError, AttributeError):
                            try:
                                # Alternative method: get_text("text")
                                page_text = getattr(page, 'get_text', lambda x: "")("text")
                            except (TypeError, AttributeError):
                                try:
                                    # Fallback: get_textpage approach
                                    textpage = getattr(page, 'get_textpage', lambda: None)()
                                    if textpage:
                                        page_text = getattr(textpage, 'extractText', lambda: "")()
                                    else:
                                        page_text = ""
                                except (TypeError, AttributeError):
                                    try:
                                        # Last resort: try legacy getText or get_text_dict
                                        page_text = getattr(page, 'getText', lambda: "")()
                                        if not page_text:
                                            # Try getting dict format and extracting text
                                            text_dict = getattr(page, 'get_text', lambda x: {})("dict")
                                            page_text = self._extract_text_from_dict(text_dict)
                                    except:
                                        page_text = ""
                        
                        if page_text and page_text.strip():
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num}: {e}")
                        continue
                
                doc.close()
                
                elapsed = time.time() - start_time
                logger.info(f"PyMuPDF extraction took {elapsed:.2f}s for {total_pages} pages")
                return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text with PyMuPDF: {e}")
            return ""
    
    def _extract_page_pymupdf(self, doc, page_num: int) -> str:
        """Extract text from a single page using PyMuPDF."""
        try:
            page = doc.load_page(page_num)
            # Try different PyMuPDF text extraction methods for compatibility
            page_text = ""
            try:
                # Primary method: get_text() - works in PyMuPDF 1.26.3+
                page_text = getattr(page, 'get_text', lambda: "")()
            except (TypeError, AttributeError):
                try:
                    # Alternative method: get_text("text")
                    page_text = getattr(page, 'get_text', lambda x: "")("text")
                except (TypeError, AttributeError):
                    try:
                        # Fallback: get_textpage approach
                        textpage = getattr(page, 'get_textpage', lambda: None)()
                        if textpage:
                            page_text = getattr(textpage, 'extractText', lambda: "")()
                        else:
                            page_text = ""
                    except (TypeError, AttributeError):
                        try:
                            # Last resort: try legacy getText or get_text_dict
                            page_text = getattr(page, 'getText', lambda: "")()
                            if not page_text:
                                # Try getting dict format and extracting text
                                text_dict = getattr(page, 'get_text', lambda x: {})("dict")
                                page_text = self._extract_text_from_dict(text_dict)
                        except:
                            page_text = ""
            
            return page_text.strip() if page_text else ""
        except Exception as e:
            logger.warning(f"Error extracting page {page_num}: {e}")
            return ""
    
    def _extract_page_pymupdf_safe(self, file_path: str, page_num: int) -> str:
        """Extract text from a single page using PyMuPDF in a thread-safe way."""
        try:
            # Open a new document instance for this thread
            doc = fitz.open(file_path)
            page = doc.load_page(page_num)
            
            # Try different PyMuPDF text extraction methods for compatibility
            page_text = ""
            try:
                # Primary method: get_text() - works in PyMuPDF 1.26.3+
                page_text = getattr(page, 'get_text', lambda: "")()
            except (TypeError, AttributeError):
                try:
                    # Alternative method: get_text("text")
                    page_text = getattr(page, 'get_text', lambda x: "")("text")
                except (TypeError, AttributeError):
                    try:
                        # Fallback: get_textpage approach
                        textpage = getattr(page, 'get_textpage', lambda: None)()
                        if textpage:
                            page_text = getattr(textpage, 'extractText', lambda: "")()
                        else:
                            page_text = ""
                    except (TypeError, AttributeError):
                        try:
                            # Last resort: try legacy getText or get_text_dict
                            page_text = getattr(page, 'getText', lambda: "")()
                            if not page_text:
                                # Try getting dict format and extracting text
                                text_dict = getattr(page, 'get_text', lambda x: {})("dict")
                                page_text = self._extract_text_from_dict(text_dict)
                        except:
                            page_text = ""
            
            doc.close()  # Close the document instance
            return page_text.strip() if page_text else ""
            
        except Exception as e:
            logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
            return ""
    
    def extract_text_smart(self, file_path: str) -> str:
        """Extract text using the best available method."""
        if self.extraction_method == "pymupdf":
            return self.extract_text_pymupdf(file_path)
        elif self.extraction_method == "pdfplumber":
            return self.extract_text_pdfplumber(file_path)
        elif self.extraction_method == "pypdf2":
            return self.extract_text_pypdf2(file_path)
        else:
            raise RuntimeError("No extraction method available")
    
    def extract_text_pdfplumber(self, file_path: str) -> str:
        """Extract text using pdfplumber with optimization for speed."""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not available")
            
        try:
            import pdfplumber
            text = ""
            start_time = time.time()
            
            # Suppress warnings during extraction
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                with pdfplumber.open(file_path) as pdf:
                    # Process pages in parallel for large PDFs
                    if len(pdf.pages) > 10:
                        with ThreadPoolExecutor(max_workers=4) as executor:
                            page_texts = list(executor.map(self._extract_page_text_pdfplumber, pdf.pages))
                            text = "\n".join(filter(None, page_texts))
                    else:
                        # Sequential processing for small PDFs
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
            
            logger.info(f"pdfplumber extraction took {time.time() - start_time:.2f}s for {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text with pdfplumber: {e}")
            return ""
    
    def _extract_page_text_pdfplumber(self, page) -> str:
        """Extract text from a single page using pdfplumber."""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Error extracting page text: {e}")
            return ""
    
    def extract_text_pypdf2(self, file_path: str) -> str:
        """Extract text using PyPDF2 as fallback with speed optimization."""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 not available")
            
        try:
            import PyPDF2
            text = ""
            start_time = time.time()
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Process pages in parallel for large PDFs
                if len(pdf_reader.pages) > 10:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        page_texts = list(executor.map(lambda p: p.extract_text(), pdf_reader.pages))
                        text = "\n".join(filter(None, page_texts))
                else:
                    # Sequential processing for small PDFs
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            
            logger.info(f"PyPDF2 extraction took {time.time() - start_time:.2f}s for {file_path}")
            return text
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
            return ""
    
    def extract_text(self, file_path: str, use_cache: Optional[bool] = None) -> str:
        """Extract text from PDF using best available method with caching."""
        if use_cache is None:
            use_cache = self.use_cache
        
        # Check cache first
        if use_cache:
            file_hash, file_size, mtime = self._get_file_info(file_path)
            cache_key = f"{file_hash}_{file_size}_{mtime}"
            
            if cache_key in self._text_cache:
                logger.info(f"Using cached text for {file_path}")
                return self._text_cache[cache_key]
        
        # Extract text using smart method selection
        text = self.extract_text_smart(file_path)
        
        if not text.strip():
            raise ValueError(f"Could not extract text from {file_path}")
        
        # Cache the result
        if use_cache:
            self._text_cache[cache_key] = text
            logger.info(f"Cached text for {file_path}")
        
        return text
    
    def get_text_preview(self, file_path: str, max_chars: int = 500) -> str:
        """Get a preview of the document text without full processing."""
        try:
            # Try to get from cache first
            if self.use_cache:
                file_hash, file_size, mtime = self._get_file_info(file_path)
                cache_key = f"{file_hash}_{file_size}_{mtime}"
                
                if cache_key in self._text_cache:
                    full_text = self._text_cache[cache_key]
                    return full_text[:max_chars] + "..." if len(full_text) > max_chars else full_text
            
            # For preview, just extract first few pages quickly
            if not PDFPLUMBER_AVAILABLE:
                logger.error("pdfplumber is not installed. Cannot generate preview.")
                return "Preview unavailable: pdfplumber is not installed."
            with pdfplumber.open(file_path) as pdf:
                preview_text = ""
                max_pages = min(3, len(pdf.pages))  # Only first 3 pages for preview

                for i in range(max_pages):
                    page_text = pdf.pages[i].extract_text()
                    if page_text:
                        preview_text += page_text + "\n"
                        if len(preview_text) >= max_chars:
                            break

                return preview_text[:max_chars] + "..." if len(preview_text) > max_chars else preview_text
                
        except Exception as e:
            logger.error(f"Error getting preview for {file_path}: {e}")
            return f"Preview unavailable: {str(e)}"
    
    def process_pdf(self, file_path: str, metadata: Optional[Dict[str, Any]] = None, use_cache: Optional[bool] = None) -> List[Document]:
        """Process a PDF file and return chunked documents with caching."""
        if use_cache is None:
            use_cache = self.use_cache
        
        try:
            # Check cache first
            if use_cache:
                file_hash, file_size, mtime = self._get_file_info(file_path)
                cache_key = f"{file_hash}_{file_size}_{mtime}_{self.chunk_size}_{self.chunk_overlap}"
                
                if cache_key in self._document_cache:
                    logger.info(f"Using cached documents for {file_path}")
                    cached_docs = self._document_cache[cache_key]
                    
                    # Update metadata if provided
                    if metadata:
                        for doc in cached_docs:
                            doc.metadata.update(metadata)
                    
                    return cached_docs
            
            start_time = time.time()
            
            # Extract text
            text = self.extract_text(file_path, use_cache=use_cache)
            
            # Create base metadata
            file_hash, file_size, mtime = self._get_file_info(file_path)
            base_metadata = {
                "source": file_path,
                "file_name": Path(file_path).name,
                "file_size": file_size,
                "file_hash": file_hash,
                "document_type": "human_rights_law",
                "processing_time": 0  # Will be updated below
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create documents
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = base_metadata.copy()
                doc_metadata.update({
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                })
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            processing_time = time.time() - start_time
            
            # Update processing time in all documents
            for doc in documents:
                doc.metadata["processing_time"] = processing_time
            
            # Cache the result
            if use_cache:
                self._document_cache[cache_key] = documents
                logger.info(f"Cached {len(documents)} documents for {file_path}")
            
            logger.info(f"Processed {file_path}: {len(documents)} chunks created in {processing_time:.2f}s")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate MD5 hash of file for deduplication."""
        # Use the cached version from _get_file_info
        file_hash, _, _ = self._get_file_info(file_path)
        return file_hash
    
    def clear_cache(self):
        """Clear the internal caches."""
        self._text_cache.clear()
        self._document_cache.clear()
        logger.info("PDF processor caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "text_cache_size": len(self._text_cache),
            "document_cache_size": len(self._document_cache)
        }

class TextProcessor:
    """Handle text preprocessing and cleaning."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove special characters but keep legal formatting
        import re
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract key legal terms and concepts."""
        # Simple keyword extraction for legal documents
        import re
        
        # Legal terms pattern
        legal_patterns = [
            r'\barticle\s+\d+\b',
            r'\bsection\s+\d+\b',
            r'\bparagraph\s+\d+\b',
            r'\bhuman\s+rights?\b',
            r'\bfreedom\s+of\b',
            r'\bright\s+to\b',
            r'\bshall\s+not\b',
            r'\bprohibited\b',
            r'\bmandatory\b',
            r'\bobligatory\b'
        ]
        
        keywords = []
        for pattern in legal_patterns:
            matches = re.findall(pattern, text.lower())
            keywords.extend(matches)
        
        # Remove duplicates and return top keywords
        unique_keywords = list(set(keywords))
        return unique_keywords[:max_keywords]

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingJob:
    job_id: str
    file_path: str
    metadata: Optional[Dict[str, Any]]
    status: ProcessingStatus
    result: Optional[List[Document]]
    error: Optional[str]
    progress: float
    start_time: float
    end_time: Optional[float]

class AsyncPDFProcessor(PDFProcessor):
    """PDF Processor with asynchronous processing capabilities."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, use_cache: bool = True):
        super().__init__(chunk_size, chunk_overlap, use_cache)
        self._processing_jobs = {}
        self._job_counter = 0
        
    def submit_processing_job(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Submit a PDF processing job and return job ID."""
        self._job_counter += 1
        job_id = f"job_{self._job_counter}_{int(time.time())}"
        
        job = ProcessingJob(
            job_id=job_id,
            file_path=file_path,
            metadata=metadata,
            status=ProcessingStatus.PENDING,
            result=None,
            error=None,
            progress=0.0,
            start_time=time.time(),
            end_time=None
        )
        
        self._processing_jobs[job_id] = job
        
        # Start processing in background thread
        threading.Thread(target=self._process_job_background, args=(job_id,), daemon=True).start()
        
        return job_id
    
    def _process_job_background(self, job_id: str):
        """Process a job in the background."""
        job = self._processing_jobs[job_id]
        
        try:
            job.status = ProcessingStatus.PROCESSING
            job.progress = 0.1
            
            # Process the PDF
            result = self.process_pdf(job.file_path, job.metadata)
            
            job.result = result
            job.status = ProcessingStatus.COMPLETED
            job.progress = 1.0
            job.end_time = time.time()
            
            logger.info(f"Background processing completed for job {job_id}")
            
        except Exception as e:
            job.error = str(e)
            job.status = ProcessingStatus.FAILED
            job.progress = 0.0
            job.end_time = time.time()
            logger.error(f"Background processing failed for job {job_id}: {e}")
    
    def get_job_status(self, job_id: str) -> Optional[ProcessingJob]:
        """Get the status of a processing job."""
        return self._processing_jobs.get(job_id)
    
    def is_job_complete(self, job_id: str) -> bool:
        """Check if a job is complete."""
        job = self._processing_jobs.get(job_id)
        return job is not None and job.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]
    
    def get_job_result(self, job_id: str) -> Optional[List[Document]]:
        """Get the result of a completed processing job, or None if not complete or not found."""
        job = self._processing_jobs.get(job_id)
        if job and job.status == ProcessingStatus.COMPLETED:
            return job.result
        return None
