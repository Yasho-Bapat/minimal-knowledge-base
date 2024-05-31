import os
import dotenv

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from _haystack.minimal_KB import HaystackKnowledgeBase
from _langchain.minimal_KB import LangchainKnowledgeBase

dotenv.load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/run")
async def run_pipeline(request: Request, pipeline: str = Form(...)):
    query = "Give me detailed information about copper sulphate based on the documents provided."
    if pipeline == "haystack":
        kb = HaystackKnowledgeBase(
            docs_dir="../docs/",
            azure_api_key=os.getenv("AZURE_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            endpoint=os.getenv("ENDPOINT"),
        )
        result, time = kb.run(query=query, output_file="../_haystack/output.txt")
    elif pipeline == "langchain":
        kb = LangchainKnowledgeBase(
            docs_dir="../docs/",
            azure_api_key=os.getenv("AZURE_API_KEY"),
            cohere_api_key=os.getenv("COHERE_API_KEY"),
            endpoint=os.getenv("ENDPOINT"),
        )
        result, time = kb.run(query=query, output_file="../_langchain/output.txt")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "framework": pipeline,
            "result": result.split(". "),
            "time": time,
        },
    )


@app.post("/requery")
async def requery(query: str = Form(...)):
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
