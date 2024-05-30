import os
import dotenv

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from _haystack.minimal_KB import HaystackKnowledgeBase
from _langchain.minimal_KB import LangchainKnowledgeBase

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run")
async def run_pipeline(request: Request, pipeline: str = Form(...)):
    query = "Give me detailed information about sulfuric acid based on the documents provided."
    if pipeline == "haystack":
        kb = HaystackKnowledgeBase(
            docs_dir="../docs/",
            azure_api_key="ba4c5c522b134b5c8e0728f8a4264003",
            cohere_api_key="VZsa4drkzj1Di2ki7n8nboLHuFQg3y7m03ktlFpf",
            endpoint="https://anonymus-manatee.cognitiveservices.azure.com/",
        )
        result = kb.run(query=query, output_file="output_haystack.txt")
    elif pipeline == "langchain":
        kb = LangchainKnowledgeBase(
            docs_dir="../docs/",
            cohere_api_key="VZsa4drkzj1Di2ki7n8nboLHuFQg3y7m03ktlFpf",
            azure_api_key="ba4c5c522b134b5c8e0728f8a4264003",
            endpoint="https://anonymus-manatee.cognitiveservices.azure.com/",
        )
        result = kb.run(query=query, output_file="output_langchain.txt")

    return templates.TemplateResponse(
        "result.html", {"request": request, "result": result}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
