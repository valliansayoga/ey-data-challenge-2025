from setuptools import setup, find_packages

# Read the dependencies from requirements.txt
def parse_requirements(filename: str) -> list[str]:
    with open(filename, encoding="utf-8") as f:
        lines = f.read().splitlines()
        return [line.strip() for line in lines if line and not line.startswith("#")]

setup(
    name="src",
    version="1.0.0",
    description="A geospatial and solar feature extraction toolkit",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=parse_requirements("requirements.txt"),
    python_requires=">=3.8",
)
