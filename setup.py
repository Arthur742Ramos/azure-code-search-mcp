from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = [                         # this now includes the Git URL
        line.strip() for line in f
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="code-search-mcp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'code-search-mcp=code_search_mcp:main',
        ],
    },
)