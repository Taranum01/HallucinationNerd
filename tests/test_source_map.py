"""Source labels and parsed metadata stay local to one verification response."""
import os
import sys


def test_source_map_uses_document_reference_metadata(mock_openai, tmp_path, monkeypatch):
    web_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')
    if web_path not in sys.path:
        sys.path.insert(0, web_path)
    import app
    import citation_resolver
    paper = tmp_path / 'paper.txt'
    paper.write_text('A sample observation is reported [1].\n\nReferences\n\n[1] Doe, J. Example Source Title. 2024. https://example.org/source\n')
    monkeypatch.setattr(citation_resolver, 'resolve_and_fetch_all', lambda *args, **kwargs: {'1': 'Sample observation reported in the example source title.'})
    mock_openai.next_verdict = {'verdict': 'SUPPORTED', 'confidence': .9, 'evidence_quote': 'Sample observation', 'reasoning': 'Illustrative test'}
    result = app._run_verification(str(paper), paper.name, '.txt', 'auto')
    assert '1' in result['source_map']
    assert result['source_map']['1']['url'] == 'https://example.org/source'
    assert result['source_map']['1']['title']
    assert '2' not in result['source_map']
