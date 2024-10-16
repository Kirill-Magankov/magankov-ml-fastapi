from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root():
    return {"msg": "ok"}


@router.get("/hello/{name}")
async def say_hello(name: str):
    return {"msg": f"Hello {name}"}
