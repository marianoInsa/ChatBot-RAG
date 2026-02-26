"""Tests para app/loaders/web.py"""
from unittest.mock import patch, MagicMock, PropertyMock
from langchain_core.documents import Document
import pytest


class TestSeleniumURLLoaderWithWait:
    """Tests para SeleniumURLLoaderWithWait."""

    @patch("app.loaders.web.webdriver.Chrome")
    @patch("app.loaders.web.webdriver.ChromeOptions")
    def test_carga_url_con_contenido(self, mock_options_class, mock_chrome_class):
        """Debe extraer contenido de una URL y generar un Document."""
        mock_driver = MagicMock()
        mock_driver.page_source = "<html><body><p>Texto de prueba</p></body></html>"
        mock_driver.title = "Página de Prueba"
        mock_chrome_class.return_value = mock_driver

        from app.loaders.web import SeleniumURLLoaderWithWait
        loader = SeleniumURLLoaderWithWait(
            urls=["https://example.com"],
            headless=True
        )
        docs = loader.load()

        assert len(docs) == 1
        assert "Texto de prueba" in docs[0].page_content
        assert docs[0].metadata["source"] == "https://example.com"
        assert docs[0].metadata["source_type"] == "web"
        assert docs[0].metadata["title"] == "Página de Prueba"
        mock_driver.quit.assert_called_once()

    @patch("app.loaders.web.webdriver.Chrome")
    @patch("app.loaders.web.webdriver.ChromeOptions")
    def test_contenido_vacio_no_genera_documento(self, mock_options_class, mock_chrome_class):
        """No debe generar documentos si la página no tiene texto."""
        mock_driver = MagicMock()
        mock_driver.page_source = "<html><body><script>var x=1;</script></body></html>"
        mock_driver.title = "Vacía"
        mock_chrome_class.return_value = mock_driver

        from app.loaders.web import SeleniumURLLoaderWithWait
        loader = SeleniumURLLoaderWithWait(urls=["https://example.com"])
        docs = loader.load()

        assert len(docs) == 0
        mock_driver.quit.assert_called_once()

    @patch("app.loaders.web.webdriver.Chrome")
    @patch("app.loaders.web.webdriver.ChromeOptions")
    def test_error_en_url_continua(self, mock_options_class, mock_chrome_class):
        """Debe continuar con las demás URLs si una falla."""
        mock_driver = MagicMock()
        mock_driver.get.side_effect = [
            Exception("Error en URL 1"),  # primera URL falla
            None,  # segunda URL OK
        ]
        mock_driver.page_source = "<html><body><p>Contenido OK</p></body></html>"
        mock_driver.title = "OK"
        mock_chrome_class.return_value = mock_driver

        from app.loaders.web import SeleniumURLLoaderWithWait
        loader = SeleniumURLLoaderWithWait(
            urls=["https://fail.com", "https://ok.com"]
        )
        docs = loader.load()

        # La primera URL falla, la segunda se carga
        assert len(docs) == 1
        mock_driver.quit.assert_called_once()

    @patch("app.loaders.web.webdriver.Chrome")
    @patch("app.loaders.web.webdriver.ChromeOptions")
    def test_espera_selector_wait_map(self, mock_options_class, mock_chrome_class):
        """Debe esperar el selector si la URL está en wait_map."""
        mock_driver = MagicMock()
        mock_driver.page_source = "<html><body><p>Con espera</p></body></html>"
        mock_driver.title = "Wait"
        mock_chrome_class.return_value = mock_driver

        from selenium.webdriver.common.by import By
        from app.loaders.web import SeleniumURLLoaderWithWait

        loader = SeleniumURLLoaderWithWait(
            urls=["https://example.com"],
            wait_map={"https://example.com": (By.TAG_NAME, "body")},
            wait_time=5
        )

        with patch("app.loaders.web.WebDriverWait") as mock_wait:
            mock_wait_instance = MagicMock()
            mock_wait.return_value = mock_wait_instance
            docs = loader.load()

        assert len(docs) == 1
        mock_wait.assert_called_once()


class TestWebLoader:
    """Tests para WebLoader."""

    @patch("app.loaders.web.SeleniumURLLoaderWithWait")
    def test_delega_a_selenium_loader(self, mock_selenium_class):
        """WebLoader debe delegar la carga a SeleniumURLLoaderWithWait."""
        mock_loader = MagicMock()
        expected_docs = [Document(page_content="Web", metadata={"source": "url"})]
        mock_loader.load.return_value = expected_docs
        mock_selenium_class.return_value = mock_loader

        from app.loaders.web import WebLoader
        loader = WebLoader(urls=["https://example.com/1", "https://example.com/2"])
        docs = loader.load()

        mock_selenium_class.assert_called_once()
        assert docs == expected_docs
