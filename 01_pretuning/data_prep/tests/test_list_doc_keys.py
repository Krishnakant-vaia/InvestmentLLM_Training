from list_doc_keys import list_corpus_doc_keys


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, **kwargs):
        return iter(self.pages)


class FakeS3Client:
    def __init__(self, pages):
        self._paginator = FakePaginator(pages)

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self._paginator


def test_list_corpus_doc_keys_filters_by_extension_and_formats_as_s3_uri():
    pages = [
        {"Contents": [
            {"Key": "prefix/a.md"},
            {"Key": "prefix/folder/.keep"},
            {"Key": "prefix/b.md"},
        ]},
    ]
    client = FakeS3Client(pages)
    keys = list_corpus_doc_keys("my-bucket", "prefix/", client)
    assert keys == ["s3://my-bucket/prefix/a.md", "s3://my-bucket/prefix/b.md"]

def test_list_corpus_doc_keys_spans_multiple_pages():
    pages = [
        {"Contents": [{"Key": "prefix/a.md"}]},
        {"Contents": [{"Key": "prefix/b.md"}]},
    ]
    client = FakeS3Client(pages)
    keys = list_corpus_doc_keys("my-bucket", "prefix/", client)
    assert keys == ["s3://my-bucket/prefix/a.md", "s3://my-bucket/prefix/b.md"]

def test_list_corpus_doc_keys_handles_empty_page():
    pages = [{"Contents": []}, {"Contents": [{"Key": "prefix/a.md"}]}]
    client = FakeS3Client(pages)
    keys = list_corpus_doc_keys("my-bucket", "prefix/", client)
    assert keys == ["s3://my-bucket/prefix/a.md"]

def test_list_corpus_doc_keys_respects_custom_extensions():
    pages = [{"Contents": [{"Key": "prefix/a.md"}, {"Key": "prefix/b.txt"}]}]
    client = FakeS3Client(pages)
    keys = list_corpus_doc_keys("my-bucket", "prefix/", client, extensions={".txt"})
    assert keys == ["s3://my-bucket/prefix/b.txt"]
