# Description: This file contains the 'brain' of our system - the
# comprehensive Pydantic schema for metadata extraction.
# =================================================================

from pydantic import BaseModel, Field, UUID4, field_validator
from typing import List, Optional, Literal, Union
from datetime import datetime, date
import uuid

# --- Nested Models for Clarity ---

class SourceInfo(BaseModel):
    """Contains information about the document's origin."""
    organization: str = Field(default="Unknown", description="The institution or body that published the document (e.g., 'National Universities Commission').")
    url: Optional[str] = Field(description="The source URL from which the document was retrieved, if applicable.")
    authority_tier: Literal[1, 2, 3, 4] = Field(default=4, description="Authority level: 1 for federal, 2 for state, 3 for institutional, 4 for other.")
    reliability_score: float = Field(default=0.5, description="A score from 0.0 to 1.0 indicating the trustworthiness of the source.", ge=0.0, le=1.0)

class TemporalInfo(BaseModel):
    """Contains information about the document's timeline."""
    publication_date: Optional[Union[date, str]] = Field(description="The date the document was officially published.")
    last_updated: Optional[Union[date, str]] = Field(description="The date the document content was last modified.")
    validity_period: Optional[str] = Field(description="The academic year or period for which the document is valid, e.g., '2024/2025'.")
    version: Optional[str] = Field(description="The specific version number or name of the document, if any.")
    
    @field_validator('publication_date', 'last_updated', mode='before')
    @classmethod
    def parse_date(cls, v):
        """Parse dates from various formats - handles both API dates and LLM-generated strings"""
        if v is None or v == "":
            return None
        if isinstance(v, (date, datetime)):
            return v
        if isinstance(v, str):
            # Handle invalid dates from LLM
            if v in ['0000-00-00', '2008-00-00', 'unknown', 'Unknown']:
                return None
            # Try to parse valid date strings
            try:
                # Try YYYY-MM-DD format first
                if len(v) == 10 and v.count('-') == 2:
                    year, month, day = v.split('-')
                    if int(month) == 0 or int(day) == 0:
                        return None
                    return date(int(year), int(month), int(day))
                # Return descriptive dates as-is for now
                return v
            except (ValueError, TypeError):
                return v  # Return as string if can't parse
        return v

class ClassificationInfo(BaseModel):
    """Contains classification details for filtering and routing."""
    education_level: List[Literal['primary', 'jss', 'sss', 'tertiary']] = Field(default_factory=list, description="Relevant education levels.")
    content_type: Literal['curriculum', 'policy', 'admission', 'exam', 'prospectus', 'handbook', 'faq', 'unknown'] = Field(default='unknown', description="The primary functional type of the document.")
    subject_areas: List[str] = Field(default_factory=list, description="Specific subject areas mentioned, e.g., 'Mathematics', 'English', 'Sciences'.")
    geographic_scope: Literal['national', 'state', 'institutional'] = Field(default='national', description="The geographic area this document applies to.")
    institution_types: List[Literal['university', 'polytechnic', 'coe']] = Field(default_factory=list, description="Types of institutions this document is relevant for.")

class ProcessingInfo(BaseModel):
    """Contains metadata about the ingestion and quality control process."""
    extraction_method: Literal['web_scrape', 'ocr', 'manual'] = Field(default='manual', description="Method used for initial content extraction.")
    quality_score: float = Field(default=0.5, description="An internal score (0.0-1.0) of the document's quality and clarity.", ge=0.0, le=1.0)
    human_verified: bool = Field(default=False, description="True if a human has reviewed and approved this document and its metadata.")
    last_reviewed: Optional[Union[date, str]] = Field(description="Date of the last human review.")
    last_modified: Optional[datetime] = Field(default_factory=datetime.now, description="Timestamp of the last modification.")
    
    @field_validator('last_reviewed', mode='before')
    @classmethod
    def parse_review_date(cls, v):
        """Parse review dates from various formats"""
        if v is None or v == "":
            return None
        if isinstance(v, (date, datetime)):
            return v
        if isinstance(v, str):
            if v in ['0000-00-00', '2008-00-00', 'unknown', 'Unknown']:
                return None
            try:
                if len(v) == 10 and v.count('-') == 2:
                    year, month, day = v.split('-')
                    if int(month) == 0 or int(day) == 0:
                        return None
                    return date(int(year), int(month), int(day))
                return v
            except (ValueError, TypeError):
                return v
        return v

class LinguisticInfo(BaseModel):
    """Contains linguistic details of the document."""
    primary_language: str = Field(default="english", description="The primary language of the document content.")
    secondary_languages: List[str] = Field(default_factory=list, description="Any other languages present in the document.")
    terminology_complexity: Literal['basic', 'intermediate', 'advanced'] = Field(default='intermediate', description="The complexity level of the terminology used.")

class SearchOptimizationInfo(BaseModel):
    """Contains keywords and questions to improve search relevance."""
    keywords: List[str] = Field(default_factory=list, description="A list of primary keywords for search.")
    synonyms: List[str] = Field(default_factory=list, description="Alternative terms or synonyms for the main keywords.")
    common_queries: List[str] = Field(default_factory=list, description="Example questions this document can answer.")


# --- Main Metadata Schema ---

class EduBotMetadata(BaseModel):
    """
    The canonical metadata schema for all documents ingested by EduBot.
    The LLM extractor will populate this schema based on the document's content.
    """
# Auto-generated fields - exclude from LLM input/output
    document_id: Optional[str] = Field(default=None, description="Unique identifier for the document (auto-generated).")
    date_uploaded: Optional[datetime] = Field(default=None, description="Timestamp when document was uploaded (auto-generated).")
    title: str = Field(description="The official title of the document or web page.")
    summary: str = Field(description="A concise, one-paragraph summary of the document's core purpose and content.")
    
    source: SourceInfo
    temporal: TemporalInfo
    classification: ClassificationInfo
    processing: ProcessingInfo
    linguistic: LinguisticInfo
    search_optimization: SearchOptimizationInfo

    def model_post_init(self, __context):
        """Ensure auto-generated fields are set after model creation"""
        if not self.document_id:
            self.document_id = str(uuid.uuid4())
        if not self.date_uploaded:
            self.date_uploaded = datetime.now()
