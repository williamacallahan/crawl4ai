from pathlib import Path

from setuptools import find_packages, setup

# Note: Most configuration is now in pyproject.toml
# This setup.py is kept for backwards compatibility

version = "0.0.0"  # This will be overridden by pyproject.toml's dynamic version
version_file = Path("crawl4ai/__version__.py")
if version_file.is_file():
    for line in version_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break

setup(
    name="Crawl4AI",
    version=version,
    description="🚀🤖 Crawl4AI: Open-source LLM Friendly Web Crawler & scraper",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/unclecode/crawl4ai",
    author="Unclecode",
    author_email="unclecode@kidocode.com",
    license="Apache-2.0",
    packages=find_packages(),
    package_data={"crawl4ai": ["js_snippet/*.js"]},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.10",
)
