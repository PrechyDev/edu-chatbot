"""
Integration tests for the complete EduBot system.
Tests end-to-end workflows including ingestion -> storage -> retrieval -> chat.
"""

import sys
import tempfile
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ingestion.ingest import IngestionOrchestrator
from src.chat_system.bot import EduBot


class TestEndToEndWorkflow:
    """Test complete system workflows."""
    
    def test_url_to_chat_workflow(self):
        """Test complete workflow from URL ingestion to chat response."""
        print("🔄 Testing URL -> Chat workflow...")
        
        try:
            # Test URL that should work
            test_url = "https://oauife.edu.ng/admission-undergraduate-studies/"
            
            # Step 1: Test ingestion
            print(f"  📥 Testing ingestion of: {test_url}")
            orchestrator = IngestionOrchestrator()
            
            # This might fail due to network issues, but should handle gracefully
            try:
                results = orchestrator.ingest(test_url)
                print(f"  ✅ Ingestion successful: {results['documents_loaded']} documents")
                ingestion_success = True
            except Exception as e:
                print(f"  ⚠️ Ingestion failed (expected): {e}")
                ingestion_success = False
            
            # Step 2: Test chat system
            print("  💬 Testing chat system...")
            bot = EduBot()
            
            # Test greeting
            response = bot.chat("Hello")
            assert response['query_type'] == 'greeting'
            print("  ✅ Greeting response successful")
            
            # Test institutional query
            response = bot.chat("What are the admission requirements for Computer Science?")
            assert 'response' in response
            print(f"  ✅ Institutional query successful: KB used = {response.get('knowledge_base_used', False)}")
            
            return {
                'ingestion_success': ingestion_success,
                'chat_success': True,
                'workflow_complete': True
            }
            
        except Exception as e:
            print(f"  ❌ Workflow test failed: {e}")
            return {
                'ingestion_success': False,
                'chat_success': False,
                'workflow_complete': False,
                'error': str(e)
            }
    
    def test_document_file_workflow(self):
        """Test workflow with a local document file."""
        print("🔄 Testing File -> Chat workflow...")
        
        try:
            # Create a temporary test document
            test_content = """
            University Admission Requirements
            
            Computer Science Program:
            - Minimum of 5 O'Level credits including English and Mathematics
            - Physics and Chemistry required
            - Minimum JAMB score of 200
            
            Engineering Programs:
            - Similar requirements to Computer Science
            - Additional requirement for Further Mathematics
            
            Business Administration:
            - English and Mathematics required
            - Economics preferred
            - Minimum JAMB score of 180
            """
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_content)
                temp_file = f.name
            
            try:
                # Step 1: Test ingestion
                print(f"  📥 Testing ingestion of temporary file...")
                orchestrator = IngestionOrchestrator()
                
                # This should work with a local file
                results = orchestrator.ingest(temp_file)
                print(f"  ✅ File ingestion successful: {results['documents_loaded']} documents")
                
                # Step 2: Test chat with the ingested content
                print("  💬 Testing chat with ingested content...")
                bot = EduBot()
                
                response = bot.chat("What are the requirements for Computer Science?")
                print(f"  ✅ Query response: KB used = {response.get('knowledge_base_used', False)}")
                
                return {
                    'file_ingestion_success': True,
                    'chat_success': True,
                    'workflow_complete': True
                }
                
            finally:
                # Clean up
                os.unlink(temp_file)
                
        except Exception as e:
            print(f"  ❌ File workflow test failed: {e}")
            return {
                'file_ingestion_success': False,
                'chat_success': False,
                'workflow_complete': False,
                'error': str(e)
            }
    
    def test_conversation_flow(self):
        """Test multi-turn conversation flow."""
        print("🔄 Testing conversation flow...")
        
        try:
            bot = EduBot()
            
            # Start conversation
            responses = []
            
            # Turn 1: Greeting
            response1 = bot.chat("Hello")
            responses.append(response1)
            assert response1['query_type'] == 'greeting'
            
            # Turn 2: Institutional query
            response2 = bot.chat("What programs do you offer?")
            responses.append(response2)
            assert 'response' in response2
            
            # Turn 3: Follow-up
            response3 = bot.chat("Tell me more about Computer Science")
            responses.append(response3)
            assert 'response' in response3
            
            # Turn 4: Thank you
            response4 = bot.chat("Thank you for the information")
            responses.append(response4)
            assert 'response' in response4
            
            print(f"  ✅ Conversation flow successful: {len(responses)} turns")
            
            return {
                'conversation_success': True,
                'turns_completed': len(responses),
                'responses': responses
            }
            
        except Exception as e:
            print(f"  ❌ Conversation flow test failed: {e}")
            return {
                'conversation_success': False,
                'error': str(e)
            }


class TestSystemRobustness:
    """Test system robustness and error handling."""
    
    def test_invalid_inputs(self):
        """Test system behavior with invalid inputs."""
        print("🛡️ Testing system robustness...")
        
        results = {
            'ingestion_errors_handled': 0,
            'chat_errors_handled': 0,
            'total_tests': 0
        }
        
        # Test ingestion with invalid inputs
        invalid_sources = [
            "nonexistent_file.pdf",
            "https://thisdoesnotexist12345.com",
            "",
            None
        ]
        
        orchestrator = IngestionOrchestrator()
        
        for source in invalid_sources:
            if source is None:
                continue
                
            results['total_tests'] += 1
            try:
                orchestrator.ingest(source)
                print(f"  ⚠️ Unexpected success with: {source}")
            except Exception as e:
                print(f"  ✅ Properly handled invalid source '{source}': {type(e).__name__}")
                results['ingestion_errors_handled'] += 1
        
        # Test chat with various inputs
        bot = EduBot()
        
        problematic_queries = [
            "",
            "a" * 10000,  # Very long query
            "🎵🎶🎵🎶🎵",  # Emoji only
            None
        ]
        
        for query in problematic_queries:
            if query is None:
                continue
                
            results['total_tests'] += 1
            try:
                response = bot.chat(query)
                if 'error' not in response:
                    print(f"  ✅ Handled problematic query gracefully")
                    results['chat_errors_handled'] += 1
                else:
                    print(f"  ✅ Properly returned error for problematic query")
                    results['chat_errors_handled'] += 1
            except Exception as e:
                print(f"  ⚠️ Unhandled exception with query: {type(e).__name__}")
        
        return results


def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Integration Tests")
    print("=" * 60)
    
    test_results = {
        'passed': 0,
        'failed': 0,
        'errors': [],
        'workflow_results': {}
    }
    
    # Test complete workflows
    print("\n📋 Testing Complete Workflows")
    workflow_tester = TestEndToEndWorkflow()
    
    # URL workflow test
    try:
        url_result = workflow_tester.test_url_to_chat_workflow()
        test_results['workflow_results']['url_workflow'] = url_result
        if url_result.get('workflow_complete', False):
            test_results['passed'] += 1
        else:
            test_results['failed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(f"URL workflow: {e}")
    
    # File workflow test
    try:
        file_result = workflow_tester.test_document_file_workflow()
        test_results['workflow_results']['file_workflow'] = file_result
        if file_result.get('workflow_complete', False):
            test_results['passed'] += 1
        else:
            test_results['failed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(f"File workflow: {e}")
    
    # Conversation flow test
    try:
        conv_result = workflow_tester.test_conversation_flow()
        test_results['workflow_results']['conversation_flow'] = conv_result
        if conv_result.get('conversation_success', False):
            test_results['passed'] += 1
        else:
            test_results['failed'] += 1
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(f"Conversation flow: {e}")
    
    # Robustness tests
    print("\n📋 Testing System Robustness")
    robustness_tester = TestSystemRobustness()
    
    try:
        robustness_result = robustness_tester.test_invalid_inputs()
        test_results['workflow_results']['robustness'] = robustness_result
        
        # Consider it passed if most errors were handled
        error_handling_rate = (
            robustness_result['ingestion_errors_handled'] + 
            robustness_result['chat_errors_handled']
        ) / max(robustness_result['total_tests'], 1)
        
        if error_handling_rate > 0.8:  # 80% error handling rate
            test_results['passed'] += 1
        else:
            test_results['failed'] += 1
            
    except Exception as e:
        test_results['failed'] += 1
        test_results['errors'].append(f"Robustness test: {e}")
    
    # Print results
    print(f"\n🏁 Integration Test Results:")
    print(f"   ✅ Passed: {test_results['passed']}")
    print(f"   ❌ Failed: {test_results['failed']}")
    
    if test_results['errors']:
        print(f"\n❌ Errors:")
        for error in test_results['errors']:
            print(f"   - {error}")
    
    return test_results


if __name__ == "__main__":
    run_integration_tests()
