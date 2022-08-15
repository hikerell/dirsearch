"""
pip3 install fastapi uvicorn
uvicorn "fake_200_app:app" --port 8001
"""
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

app = FastAPI()
users = [
    {"uid": 1, "name": "admin", "scopes": ["users:all", "orders:all"]},
    {"uid": 2, "name": "alan@foxmail.com.cn", "scopes": []}
]


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc):
    body = {"code": 404, "message": f'{request.url.path} not found'}
    return JSONResponse(body, status_code=200)


@app.get('/user/{uid}')
async def get_user_by_id(uid: int):
    target = None
    for user in users:
        if uid == user.get('uid'):
            target = user
    return {"code": 0, "message": None, "data": target}