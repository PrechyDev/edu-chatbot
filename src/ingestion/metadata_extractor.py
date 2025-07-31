"""
Metadata Extraction Module for EduBot

This module handles document metadata extraction with robust error handling,
rate limiting, retries, and fallback mechanisms for LLM-based extraction.
"""

import logging
import time
import json
from datetime import date, datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.llms.google_genai import GoogleGenAI

from .metadata_schema import EduBotMetadata
from ..config import METADATA_EXTRACTOR_MODEL, GOOGLE_API_KEY


class ExtractionError(Exception):
    """Custom exception for metadata extraction errors."""
    pass


class ExtractionResult(Enum):
    """Enumeration for metadata extraction results."""
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    NETWORK_ERROR = "network_error"
    FALLBACK_USED = "fallback_used"
    FAILED = "failed"


@dataclass
class MetadataExtractionResponse:
    """Response object for metadata extraction operations."""
    success: bool
    metadata: Optional[EduBotMetadata]
    result_type: ExtractionResult
    error_message: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False


class MetadataExtractor:
    """
    Robust metadata extractor with rate limiting, retries, and fallbacks.
    """
    
    def __init__(
        self,
        api_key: str = GOOGLE_API_KEY,
        model_name: str = METADATA_EXTRACTOR_MODEL,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_multiplier: float = 2.0
    ):
        """
        Initialize the metadata extractor.
        
        Args:
            api_key: Google API key
            model_name: Model name for metadata extraction
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_multiplier: Exponential backoff multiplier
        """
        self.api_key = api_key
        self.model_name = model_name
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        
        self.logger = logging.getLogger(__name__)
        self._setup_llm()
        self._setup_extraction_program()
    
    def _setup_llm(self):
        """Initialize the LLM for metadata extraction."""
        try:
            self.llm = GoogleGenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.1  # Low temperature for consistent metadata extraction
            )
            self.logger.info(f"Initialized LLM with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise ExtractionError(f"LLM initialization failed: {e}")
    
    def _setup_extraction_program(self):
        """Setup the Pydantic program for structured metadata extraction."""
        self.extraction_program = LLMTextCompletionProgram.from_defaults(
            output_cls=EduBotMetadata,
            prompt_template_str=self._get_extraction_prompt(),
            llm=self.llm,
            verbose=False,  # Set to False to reduce noise
        )
    
    def _get_extraction_prompt(self) -> str:
        """Get the prompt template for metadata extraction."""
        return (
            "You are an expert librarian specializing in Nigerian educational documents and institutional content.\n"
            "Your task is to analyze the provided text and extract metadata based ONLY on the content.\n"
            "This document could be from any educational institution, government agency, or educational organization.\n"
            "Document types include: handbooks, policies, news articles, brochures, course catalogs, admission guides, research papers, etc.\n"
            "\n"
            "Fill in all fields with appropriate values based on the document content.\n"
            "If you cannot determine a specific value, use reasonable defaults or 'unknown' for strings.\n"
            "IMPORTANT: Do NOT generate values for 'document_id' or 'date_uploaded' - these are auto-generated.\n"
            "\n"
            "For required fields you cannot determine:\n"
            "- Use the actual institution name if mentioned, otherwise 'Unknown Institution'\n"
            "- For content_type: 'handbook', 'policy', 'news', 'brochure', 'catalog', 'guide', 'announcement', or 'unknown'\n"
            "- For geographic_scope: 'institutional', 'state', 'national', or 'international' based on document scope\n"
            "- Use 0.5 for scores when uncertain\n"
            "- For dates, use YYYY-MM-DD format if possible, or descriptive text like 'August 2014'\n"
            "- If you can't determine dates, use null or leave empty\n"
            "- Do NOT use invalid dates like '2008-00-00' or '0000-00-00'\n"
            "- Pay attention to institutional names, department names, and program names\n"
            "- Identify the target audience (students, staff, parents, general public)\n"
            "---------------------\n"
            "Document Content (first 8000 chars):\n"
            "{input}\n"
            "---------------------\n"
            "Please provide the output in a valid JSON format that adheres to the Pydantic schema."
        )
    
    def extract_metadata(
        self,
        document,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> MetadataExtractionResponse:
        """
        Extract metadata from a document with retry logic and fallbacks.
        
        Args:
            document: The document to extract metadata from
            additional_metadata: Optional additional metadata to merge
            
        Returns:
            MetadataExtractionResponse with extraction results
        """
        file_path = document.metadata.get('file_path', 'unknown')
        self.logger.info(f"Starting metadata extraction for: {file_path}")
        
        # Try LLM extraction with retries
        llm_response = self._extract_with_llm_retry(document)
        
        if llm_response.success:
            # Merge additional metadata if provided
            if additional_metadata:
                try:
                    metadata_dict = llm_response.metadata.dict()
                    merged_dict = self._merge_metadata(metadata_dict, additional_metadata)
                    llm_response.metadata = EduBotMetadata(**merged_dict)
                except Exception as e:
                    self.logger.warning(f"Failed to merge additional metadata: {e}")
            
            return llm_response
        
        # If LLM extraction failed, use fallback
        self.logger.warning(f"LLM extraction failed for {file_path}, using fallback")
        fallback_response = self._extract_with_fallback(document, additional_metadata)
        fallback_response.error_message = llm_response.error_message
        
        return fallback_response
    
    def _extract_with_llm_retry(self, document) -> MetadataExtractionResponse:
        """
        Attempt LLM extraction with exponential backoff retry logic.
        """
        content = document.text[:8000] if len(document.text) > 8000 else document.text
        
        for attempt in range(self.max_retries + 1):
            try:
                self.logger.debug(f"LLM extraction attempt {attempt + 1}/{self.max_retries + 1}")
                
                # Extract metadata using LLM
                extracted_metadata = self.extraction_program(input=content)
                
                # Clean up the metadata
                metadata_dict = extracted_metadata.dict()
                metadata_dict.pop('document_id', None)
                metadata_dict.pop('date_uploaded', None)
                
                # Create final metadata object
                final_metadata = EduBotMetadata(**metadata_dict)
                
                self.logger.info(f"LLM extraction successful on attempt {attempt + 1}")
                return MetadataExtractionResponse(
                    success=True,
                    metadata=final_metadata,
                    result_type=ExtractionResult.SUCCESS,
                    retry_count=attempt
                )
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Determine error type
                if "429" in error_msg or "rate" in error_msg:
                    result_type = ExtractionResult.RATE_LIMITED
                elif "quota" in error_msg or "resource_exhausted" in error_msg:
                    result_type = ExtractionResult.QUOTA_EXCEEDED
                elif "network" in error_msg or "connection" in error_msg:
                    result_type = ExtractionResult.NETWORK_ERROR
                else:
                    result_type = ExtractionResult.FAILED
                
                self.logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
                
                # If this is the last attempt, return failure
                if attempt == self.max_retries:
                    return MetadataExtractionResponse(
                        success=False,
                        metadata=None,
                        result_type=result_type,
                        error_message=str(e),
                        retry_count=attempt
                    )
                
                # Calculate delay for next attempt
                delay = min(
                    self.base_delay * (self.backoff_multiplier ** attempt),
                    self.max_delay
                )
                
                self.logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        return MetadataExtractionResponse(
            success=False,
            metadata=None,
            result_type=ExtractionResult.FAILED,
            error_message="Maximum retries exceeded",
            retry_count=self.max_retries
        )
    
    def _extract_with_fallback(
        self,
        document,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> MetadataExtractionResponse:
        """
        Extract metadata using fallback mechanisms when LLM fails.
        """
        try:
            self.logger.info("Using fallback metadata extraction")
            
            # Basic metadata from document properties
            fallback_metadata = self._create_fallback_metadata(document)
            
            # Merge additional metadata if provided
            if additional_metadata:
                metadata_dict = fallback_metadata.dict()
                merged_dict = self._merge_metadata(metadata_dict, additional_metadata)
                fallback_metadata = EduBotMetadata(**merged_dict)
            
            return MetadataExtractionResponse(
                success=True,
                metadata=fallback_metadata,
                result_type=ExtractionResult.FALLBACK_USED,
                fallback_used=True
            )
            
        except Exception as e:
            self.logger.error(f"Fallback metadata extraction failed: {e}")
            return MetadataExtractionResponse(
                success=False,
                metadata=None,
                result_type=ExtractionResult.FAILED,
                error_message=f"Fallback extraction failed: {e}",
                fallback_used=True
            )
    
    def _create_fallback_metadata(self, document) -> EduBotMetadata:
        """
        Create basic metadata using document properties and heuristics.
        """
        file_path = document.metadata.get('file_path', 'unknown')
        file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        # Extract basic information from filename and content
        title = self._extract_title_heuristic(document, file_name)
        institution = self._extract_institution_heuristic(document)
        content_type = self._determine_content_type_heuristic(document, file_name)
        
        # Import the nested classes
        from .metadata_schema import (
            SourceInfo, TemporalInfo, ClassificationInfo, 
            ProcessingInfo, LinguisticInfo, SearchOptimizationInfo
        )
        
        return EduBotMetadata(
            title=title,
            summary=f"Educational document from {institution}. Content extracted using fallback method due to LLM extraction failure.",
            source=SourceInfo(
                organization=institution,
                url=None,
                authority_tier=3,  # Institutional level
                reliability_score=0.6
            ),
            temporal=TemporalInfo(
                publication_date=None,
                last_updated=None,
                validity_period=None,
                version=None
            ),
            classification=ClassificationInfo(
                education_level=['tertiary'],
                content_type=content_type,
                subject_areas=[],
                geographic_scope='institutional',
                institution_types=['university']
            ),
            processing=ProcessingInfo(
                extraction_method='manual',
                quality_score=0.4,  # Lower quality for fallback
                human_verified=False,
                last_reviewed=None
                # Note: last_modified will use default_factory
            ),
            linguistic=LinguisticInfo(
                primary_language='english',
                secondary_languages=[],
                terminology_complexity='intermediate'
            ),
            search_optimization=SearchOptimizationInfo(
                keywords=self._extract_keywords_heuristic(document, title),
                synonyms=[],
                common_queries=[]
            )
        )
    
    def _extract_title_heuristic(self, document, file_name: str) -> str:
        """Extract title using heuristic methods."""
        # Try to find title in first few lines
        lines = document.text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Likely a title if it's not too short or long
                if not line.startswith(('http', 'www', '@', '#')):
                    return line
        
        # Fallback to filename without extension
        return file_name.replace('.pdf', '').replace('_', ' ').title()
    
    def _extract_institution_heuristic(self, document) -> str:
        """Extract institution name using heuristic methods."""
        text_lower = document.text.lower()
        
        # Common Nigerian university patterns
        university_patterns = [
            'obafemi awolowo university', 'oau',
            'university of lagos', 'unilag',
            'university of ibadan', 'ui',
            'university of nigeria', 'unn',
            'ahmadu bello university', 'abu',
            'university of benin', 'uniben',
            'federal university',
            'state university',
            'university'
        ]
        
        for pattern in university_patterns:
            if pattern in text_lower:
                if pattern in ['oau', 'unilag', 'ui', 'unn', 'abu', 'uniben']:
                    # Map abbreviations to full names
                    mapping = {
                        'oau': 'Obafemi Awolowo University',
                        'unilag': 'University of Lagos',
                        'ui': 'University of Ibadan',
                        'unn': 'University of Nigeria',
                        'abu': 'Ahmadu Bello University',
                        'uniben': 'University of Benin'
                    }
                    return mapping.get(pattern, 'Unknown Institution')
                else:
                    return pattern.title()
        
        return 'Unknown Institution'
    
    def _determine_content_type_heuristic(self, document, file_name: str) -> str:
        """Determine content type using heuristic methods."""
        text_lower = document.text.lower()
        file_name_lower = file_name.lower()
        
        # Check filename patterns
        if 'handbook' in file_name_lower:
            return 'handbook'
        elif 'catalog' in file_name_lower:
            return 'curriculum'
        elif 'guide' in file_name_lower:
            return 'admission'
        elif 'policy' in file_name_lower:
            return 'policy'
        
        # Check content patterns
        if 'admission' in text_lower and 'requirement' in text_lower:
            return 'admission'
        elif 'course' in text_lower and ('catalog' in text_lower or 'program' in text_lower):
            return 'curriculum'
        elif 'student' in text_lower and 'handbook' in text_lower:
            return 'handbook'
        elif 'policy' in text_lower or 'regulation' in text_lower:
            return 'policy'
        
        return 'unknown'
    
    def _extract_keywords_heuristic(self, document, title: str) -> List[str]:
        """Extract keywords using heuristic methods."""
        keywords = []
        text_lower = document.text.lower()
        
        # Add title words as keywords
        title_words = [word.strip('.,!?()[]{}') for word in title.lower().split() 
                      if len(word) > 3 and word not in ['the', 'and', 'for', 'with', 'from']]
        keywords.extend(title_words[:5])  # Limit to 5 title words
        
        # Common educational keywords to look for
        educational_terms = [
            'university', 'college', 'admission', 'program', 'course', 'degree',
            'undergraduate', 'postgraduate', 'faculty', 'department', 'school',
            'academic', 'curriculum', 'semester', 'examination', 'student'
        ]
        
        for term in educational_terms:
            if term in text_lower and term not in keywords:
                keywords.append(term)
                if len(keywords) >= 10:  # Limit total keywords
                    break
        
        return keywords[:10]  # Return maximum 10 keywords
    
    def _merge_metadata(
        self,
        extracted_metadata: Dict[str, Any],
        additional_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge additional metadata with extracted metadata.
        Additional metadata takes precedence for overlapping fields.
        """
        merged = extracted_metadata.copy()
        
        for key, value in additional_metadata.items():
            if value is not None:  # Only override with non-None values
                merged[key] = value
        
        return merged
    
    def serialize_metadata(self, metadata: EduBotMetadata) -> Dict[str, Any]:
        """
        Convert metadata to JSON-serializable format.
        """
        def json_serializer(obj):
            if isinstance(obj, (date, datetime)):
                return obj.isoformat()
            return obj
        
        try:
            if hasattr(metadata, 'dict'):
                data = metadata.dict()
            elif isinstance(metadata, dict):
                data = metadata
            else:
                data = metadata
            
            # Convert to JSON string and back to ensure serializability
            json_str = json.dumps(data, default=json_serializer)
            return json.loads(json_str)
            
        except Exception as e:
            self.logger.warning(f"Failed to serialize metadata: {e}")
            return {"error": "Failed to serialize metadata"}


# Factory function for easy instantiation
def create_metadata_extractor(**kwargs) -> MetadataExtractor:
    """
    Create a metadata extractor with optional configuration.
    
    Args:
        **kwargs: Configuration options for MetadataExtractor
        
    Returns:
        Configured MetadataExtractor instance
    """
    return MetadataExtractor(**kwargs)


# Convenience function for direct extraction
def extract_document_metadata(
    document,
    llm=None,
    additional_metadata: Optional[Dict[str, Any]] = None,
    **extractor_kwargs
) -> Optional[EduBotMetadata]:
    """
    Extract metadata from a document with improved error handling.
    
    This is a convenience function that maintains backward compatibility
    while using the new robust extraction system.
    
    Args:
        document: The document to extract metadata from
        llm: Legacy parameter (ignored, uses internal LLM)
        additional_metadata: Optional additional metadata to merge
        **extractor_kwargs: Configuration options for MetadataExtractor
        
    Returns:
        Extracted metadata or None if extraction fails
    """
    try:
        extractor = create_metadata_extractor(**extractor_kwargs)
        response = extractor.extract_metadata(document, additional_metadata)
        
        if response.success:
            logging.info(f"Metadata extraction: {response.result_type.value}")
            if response.fallback_used:
                logging.warning("Used fallback metadata extraction")
            
            # Log final metadata information
            metadata = response.metadata
            serialized_metadata = extractor.serialize_metadata(metadata)
            serialized_size = len(json.dumps(serialized_metadata))
            
            logging.info("metadata extracted: {metadata}")
            return response.metadata
        else:
            logging.error(f"Metadata extraction failed: {response.error_message}")
            return None
            
    except Exception as e:
        logging.error(f"Metadata extraction error: {e}")
        return None
