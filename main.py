from fastapi import FastAPI

from rag.api import create_app


app: FastAPI = create_app()
