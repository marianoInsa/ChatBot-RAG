import logging
import time
from typing import List
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langchain_core.documents import Document
from app.loaders.base import BaseLoader

logger = logging.getLogger(__name__)

class SeleniumURLLoaderWithWait:
    def __init__(
        self,
        urls: List[str],
        wait_map: dict[str, tuple] | None = None,
        wait_time: int = 10,
        headless: bool = True,
        arguments: List[str] | None = None
    ):
        self.urls = urls
        self.wait_map = wait_map or {}
        self.wait_time = wait_time
        self.headless = headless
        self.arguments = arguments or []

    def _get_driver(self):
        options = webdriver.ChromeOptions()
        
        if self.headless:
            options.add_argument("--headless=new")
        
        # User-Agent real para evitar bloqueos
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        for arg in self.arguments:
            options.add_argument(arg)

        driver = webdriver.Chrome(options=options)
        
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            },
        )
        return driver

    def load(self) -> List[Document]:
        documents = []
        driver = None

        try:
            driver = self._get_driver()

            for url in self.urls:
                logger.info(f"Cargando URL: {url}")
                try:
                    driver.get(url)

                    wait_selector = self.wait_map.get(url)

                    if wait_selector:
                        # Espera extra para React
                        logger.info(f"Esperando selector {wait_selector}...")
                        WebDriverWait(driver, self.wait_time).until(
                            EC.presence_of_element_located(wait_selector)
                        )
                        time.sleep(2) 
                    else:
                        # Espera mínima para carga básica
                        logger.info(f"No se requiere espera")
                        driver.implicitly_wait(2)

                    # Parseo manual
                    page_source = driver.page_source
                    soup = BeautifulSoup(page_source, "html.parser")

                    # Eliminar scripts y estilos que meten ruido
                    for script in soup(["script", "style", "noscript", "svg"]):
                        script.extract()

                    text = soup.get_text(separator="\n", strip=True)

                    if text:
                        metadata = {
                            "source": url,
                            "title": driver.title,
                            "source_type": "web",
                            "language": "es" 
                        }
                        docs = [Document(page_content=text, metadata=metadata)]
                        documents.extend(docs)
                        logger.info(f"Contenido de {url} extraído.")
                    else:
                        logger.warning(f"Contenido vacío en {url}")

                except Exception as e:
                    logger.error(f"Error en {url}: {type(e).__name__}: {e}")
                    continue

        finally:
            if driver:
                driver.quit()

        return documents

class WebLoader(BaseLoader):
    def __init__(self, urls: List[str]):
        self.urls = urls

    def load(self) -> List[Document]:
        wait_map = {
            self.urls[0]: (By.TAG_NAME, "body"),
            self.urls[1]: (By.TAG_NAME, "body"),
        }
        
        loader = SeleniumURLLoaderWithWait(
            urls=self.urls,
            wait_time=20,
            wait_map=wait_map,
            headless=True,
            arguments=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--window-size=1920,1080",
                "--ignore-certificate-errors"
            ]
        )
        return loader.load()

if __name__ == "__main__":
    from app.config.config import get_settings
    settings = get_settings()

    if settings.urls:
        web_loader = WebLoader(settings.urls)
        docs = web_loader.load()
        print("--- RESUMEN ---")
        for doc in docs:
            print(f"Fuente: {doc.metadata.get('source')}") 
            print(f"Contenido: {doc.page_content.replace(chr(10), ' ')}...\n")