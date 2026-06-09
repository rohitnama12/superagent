from setuptools import setup, find_packages

setup(
    name="superagent",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "rich",
        "python-dotenv",
        "chromadb",
        "langchain",
        "langchain-text-splitters",
        "langchain-huggingface",
        "watchdog",
        "sentence-transformers",
        "duckduckgo-search",
        "tavily-python",
        "pymupdf",
        "pdfplumber",
        "easyocr",
        "python-docx"
    ],
    entry_points={
        "console_scripts": [
            "superagent=superagent.main:main",
        ],
    },
)
