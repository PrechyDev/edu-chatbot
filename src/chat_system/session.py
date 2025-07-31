"""
Session management utilities for conversation tracking and persistence.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages conversation sessions and persistence."""
    
    def __init__(self, session_id: str = None):
        """
        Initialize session manager.
        
        Args:
            session_id: Existing session ID or None to create new
        """
        self.session_id = session_id or self._generate_session_id()
        self.conversations_dir = Path("conversations")
        self.conversations_dir.mkdir(exist_ok=True)
        self.conversations_dir.mkdir(exist_ok=True)
    
    def _generate_session_id(self) -> str:
        """Generate a new session ID."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_session_filepath(self, session_id: str = None) -> Path:
        """Get filepath for a session."""
        sid = session_id or self.session_id
        filename = f"conversation_{sid}.json"
        return self.conversations_dir / filename
    
    def save_conversation(
        self, 
        conversation_history: List[Dict[str, Any]], 
        summary: Dict[str, Any],
        filepath: str = None
    ) -> str:
        """
        Save conversation to file.
        
        Args:
            conversation_history: List of conversation messages
            summary: Conversation summary
            filepath: Optional custom filepath
            
        Returns:
            Path to saved file
        """
        if filepath:
            save_path = Path(filepath)
        else:
            save_path = self.get_session_filepath()
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        conversation_data = {
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'summary': summary,
            'history': conversation_history,
            'version': '1.0'
        }
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Conversation saved to: {save_path}")
            return str(save_path)
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
    
    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation from file.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Conversation data or None if not found
        """
        filepath = self.get_session_filepath(session_id)
        
        if not filepath.exists():
            logger.warning(f"Conversation file not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            logger.info(f"📂 Loaded conversation from: {filepath}")
            return conversation_data
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            return None
    
    def get_available_sessions(self) -> List[str]:
        """Get list of available conversation sessions."""
        sessions = []
        
        try:
            for filepath in self.conversations_dir.glob("conversation_*.json"):
                # Extract session ID from filename
                filename = filepath.stem  # Remove .json extension
                if filename.startswith("conversation_"):
                    session_id = filename[len("conversation_"):]
                    sessions.append(session_id)
        
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
        
        return sorted(sessions, reverse=True)  # Most recent first
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if successful
        """
        filepath = self.get_session_filepath(session_id)
        
        try:
            if filepath.exists():
                filepath.unlink()
                logger.info(f"🗑️ Deleted session: {session_id}")
                return True
            else:
                logger.warning(f"Session not found: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a session without loading full conversation.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session info or None if not found
        """
        conversation_data = self.load_conversation(session_id)
        if not conversation_data:
            return None
        
        return {
            'session_id': session_id,
            'created_at': conversation_data.get('created_at'),
            'summary': conversation_data.get('summary', {}),
            'message_count': len(conversation_data.get('history', [])),
            'version': conversation_data.get('version', 'unknown')
        }
    
    def export_sessions(self, output_path: str) -> bool:
        """
        Export all sessions to a single file.
        
        Args:
            output_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            sessions = []
            for session_id in self.get_available_sessions():
                conversation_data = self.load_conversation(session_id)
                if conversation_data:
                    sessions.append(conversation_data)
            
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'session_count': len(sessions),
                'sessions': sessions
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📦 Exported {len(sessions)} sessions to: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export sessions: {e}")
            return False
    
    def cleanup_old_sessions(self, keep_count: int = 50) -> int:
        """
        Clean up old conversation sessions, keeping only the most recent ones.
        
        Args:
            keep_count: Number of sessions to keep
            
        Returns:
            Number of sessions deleted
        """
        sessions = self.get_available_sessions()
        
        if len(sessions) <= keep_count:
            return 0
        
        sessions_to_delete = sessions[keep_count:]
        deleted_count = 0
        
        for session_id in sessions_to_delete:
            if self.delete_session(session_id):
                deleted_count += 1
        
        logger.info(f"🧹 Cleaned up {deleted_count} old sessions")
        return deleted_count
