from types import SimpleNamespace

import requests

from pdfchat.qa import get_answer


class DummyVectorStore:
    def similarity_search(self, question):
        return [SimpleNamespace(page_content="The contract says invoices are due in 30 days.")]


def test_get_answer_retries_with_fallback_model(monkeypatch):
    calls = []

    def fake_make_api_request(method, path, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if len(calls) == 1:
            raise requests.exceptions.HTTPError(response=SimpleNamespace(status_code=404))

        class DummyResponse:
            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "Invoices are due in 30 days."}}
                    ]
                }

        return DummyResponse()

    monkeypatch.setattr("pdfchat.qa.make_api_request", fake_make_api_request)

    answer = get_answer("When are invoices due?", DummyVectorStore())

    assert answer == "Invoices are due in 30 days."
    assert calls == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
