from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient

api = "api/v1/documents"


@pytest.mark.anyio
async def test_post_document(client: AsyncClient, auth_headers: dict[str, str], mocked_minio):
    test_doc_path = Path(__file__).parent / "assets/sample-pdf-invoice.pdf"
    pdf_bytes = test_doc_path.read_bytes()

    with patch("app.service.document.document.start_processing.delay") as mock_task:
        response = await client.post(
            api,
            files={
                "file": (
                    "sample-pdf-invoice.pdf",
                    BytesIO(pdf_bytes),
                    "application/pdf",
                )
            },
            data={
                "processing_type": "DOCUMENT_SUMMARY",
                "instructions": "summarize",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        mocked_minio.put_object.assert_called_once()
        mock_task.assert_called_once()
        data = response.json()
        assert "document_id" in data
        assert "processing_request_id" in data
        assert "status" in data


@pytest.mark.anyio
async def test_get_all_documents(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.get(api, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == []
    assert data["total"] == 0
    assert data["has_more"] is False


@pytest.mark.anyio
async def test_document_not_found(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.get(f"{api}/999", headers=auth_headers)

    assert response.status_code == 404


@pytest.mark.anyio
async def test_delete_document_by_id(
    client: AsyncClient,
    auth_headers: dict[str, str],
    mocked_minio,
):
    test_doc_path = Path(__file__).parent / "assets/sample-pdf-invoice.pdf"
    pdf_bytes = test_doc_path.read_bytes()

    with patch("app.service.document.document.start_processing.delay") as mock_task:
        doc_response = await client.post(
            api,
            files={
                "file": (
                    "sample-pdf-invoice.pdf",
                    BytesIO(pdf_bytes),
                    "application/pdf",
                )
            },
            data={
                "processing_type": "DOCUMENT_SUMMARY",
                "instructions": "summarize",
            },
            headers=auth_headers,
        )

        assert doc_response.status_code == 201

        mocked_minio.put_object.assert_called_once()
        mock_task.assert_called_once()

    doc_data = doc_response.json()
    document_id = doc_data["document_id"]

    response = await client.get(
        f"{api}/{document_id}",
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == document_id


@pytest.mark.anyio
async def test_get_document_by_id(
    client: AsyncClient,
    auth_headers: dict[str, str],
    mocked_minio,
):
    test_doc_path = Path(__file__).parent / "assets/sample-pdf-invoice.pdf"
    pdf_bytes = test_doc_path.read_bytes()

    with patch("app.service.document.document.start_processing.delay") as mock_task:
        doc_response = await client.post(
            api,
            files={
                "file": (
                    "sample-pdf-invoice.pdf",
                    BytesIO(pdf_bytes),
                    "application/pdf",
                )
            },
            data={
                "processing_type": "DOCUMENT_SUMMARY",
                "instructions": "summarize",
            },
            headers=auth_headers,
        )

        assert doc_response.status_code == 201

        mocked_minio.put_object.assert_called_once()
        mock_task.assert_called_once()

    doc_data = doc_response.json()
    document_id = doc_data["document_id"]

    response = await client.delete(
        f"{api}/{document_id}",
        headers=auth_headers,
    )
    doc_data = doc_response.json()

    assert response.status_code == 200
