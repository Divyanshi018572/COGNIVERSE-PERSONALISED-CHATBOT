# auth/routes.py
# Two FastAPI routes: POST /auth/register and POST /auth/login
# Mounted into the existing FastAPI app in backend/main.py
# Does not touch any existing route.

from fastapi import APIRouter, HTTPException, status
from auth.models import RegisterRequest, LoginRequest, TokenResponse
from auth.db_queries import create_user, authenticate_user
from auth.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """
    Register a new user.
    Returns success message — user must then login to get token.
    """
    try:
        user = create_user(req)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    return {"message": f"Account created. Welcome, {user['full_name'].split()[0]}!"}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """
    Authenticate and return a JWT token.
    Frontend stores this token in a cookie.
    """
    user = authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({
        "sub": user["username"],
        "name": user["full_name"],
    })
    return TokenResponse(
        access_token=token,
        full_name=user["full_name"],
        username=user["username"],
    )
