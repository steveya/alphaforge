from __future__ import annotations


class FakeHttpClient:
    def __init__(self, responses: dict[str, bytes]):
        self.responses = responses

    def get_bytes(
        self,
        *,
        url: str,
        source: str,
        artifact_name: str,
        params=None,
        force: bool = False,
    ) -> bytes:
        for key, payload in self.responses.items():
            if key in url or key == artifact_name:
                return payload
        raise KeyError(f"No fake response for url={url} artifact={artifact_name}")
