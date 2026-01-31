"""
Security and Authentication Module
Provides JWT authentication, API key validation, and security middleware
"""

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from functools import wraps

from fastapi import HTTPException, Security, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
import jwt

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7
API_KEY_HEADER = "X-API-Key"

# Security scheme instances
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_HEADER, auto_error=False)


@dataclass
class User:
    """User model for authentication"""
    id: str
    username: str
    email: str
    role: str  # 'admin', 'physician', 'researcher', 'viewer'
    permissions: List[str]
    is_active: bool = True
    
    def has_permission(self, permission: str) -> bool:
        return permission in self.permissions or 'admin' in self.permissions


@dataclass
class TokenPayload:
    """JWT token payload"""
    sub: str  # user id
    username: str
    role: str
    permissions: List[str]
    exp: datetime
    iat: datetime
    type: str  # 'access' or 'refresh'


# In-memory user store (replace with database in production)
_users_db: Dict[str, Dict] = {
    "admin": {
        "id": "usr_001",
        "username": "admin",
        "email": "admin@medimind.local",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "permissions": ["read", "write", "delete", "admin", "analyze", "export"],
        "is_active": True
    },
    "physician": {
        "id": "usr_002", 
        "username": "physician",
        "email": "physician@medimind.local",
        "password_hash": hashlib.sha256("doctor123".encode()).hexdigest(),
        "role": "physician",
        "permissions": ["read", "write", "analyze", "export"],
        "is_active": True
    },
    "researcher": {
        "id": "usr_003",
        "username": "researcher",
        "email": "researcher@medimind.local",
        "password_hash": hashlib.sha256("research123".encode()).hexdigest(),
        "role": "researcher",
        "permissions": ["read", "analyze"],
        "is_active": True
    }
}

# API Keys (for service-to-service auth)
_api_keys: Dict[str, Dict] = {
    hashlib.sha256("medimind-dev-key-2024".encode()).hexdigest(): {
        "name": "Development Key",
        "permissions": ["read", "write", "analyze"],
        "rate_limit": 1000
    },
    hashlib.sha256("medimind-prod-key-2024".encode()).hexdigest(): {
        "name": "Production Key", 
        "permissions": ["read", "write", "analyze", "export"],
        "rate_limit": 10000
    }
}


class AuthenticationError(HTTPException):
    """Custom authentication exception"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(HTTPException):
    """Custom authorization exception"""
    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(plain_password) == hashed_password


def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "permissions": user.permissions,
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(user: User) -> str:
    """Create a JWT refresh token"""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    payload = {
        "sub": user.id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def get_user_by_id(user_id: str) -> Optional[User]:
    """Get user by ID from database"""
    for user_data in _users_db.values():
        if user_data["id"] == user_id:
            return User(
                id=user_data["id"],
                username=user_data["username"],
                email=user_data["email"],
                role=user_data["role"],
                permissions=user_data["permissions"],
                is_active=user_data["is_active"]
            )
    return None


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Authenticate user with username and password"""
    user_data = _users_db.get(username)
    
    if not user_data:
        return None
    
    if not verify_password(password, user_data["password_hash"]):
        return None
    
    if not user_data["is_active"]:
        return None
    
    return User(
        id=user_data["id"],
        username=user_data["username"],
        email=user_data["email"],
        role=user_data["role"],
        permissions=user_data["permissions"],
        is_active=user_data["is_active"]
    )


def validate_api_key(api_key: str) -> Optional[Dict]:
    """Validate an API key"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return _api_keys.get(key_hash)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header)
) -> Optional[User]:
    """
    Get current user from JWT token or API key.
    Returns None if no authentication provided (for public endpoints).
    """
    # Try JWT token first
    if credentials:
        try:
            payload = decode_token(credentials.credentials)
            
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")
            
            user = get_user_by_id(payload["sub"])
            if not user:
                raise AuthenticationError("User not found")
            
            if not user.is_active:
                raise AuthenticationError("User account is disabled")
            
            return user
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(f"Token validation failed: {str(e)}")
    
    # Try API key
    if api_key:
        key_info = validate_api_key(api_key)
        if key_info:
            # Return a system user for API key auth
            return User(
                id="sys_api_key",
                username=f"api_key:{key_info['name']}",
                email="api@medimind.local",
                role="service",
                permissions=key_info["permissions"],
                is_active=True
            )
    
    # No authentication provided
    return None


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme),
    api_key: str = Security(api_key_header)
) -> User:
    """Require authentication - raises error if not authenticated"""
    user = await get_current_user(credentials, api_key)
    if not user:
        raise AuthenticationError("Authentication required")
    return user


def require_permission(permission: str):
    """Decorator factory for requiring specific permissions"""
    async def permission_checker(user: User = Depends(require_auth)) -> User:
        if not user.has_permission(permission):
            raise AuthorizationError(f"Permission '{permission}' required")
        return user
    return permission_checker


def require_role(roles: List[str]):
    """Decorator factory for requiring specific roles"""
    async def role_checker(user: User = Depends(require_auth)) -> User:
        if user.role not in roles and 'admin' not in user.permissions:
            raise AuthorizationError(f"One of roles {roles} required")
        return user
    return role_checker


# Rate limiting (simple in-memory implementation)
_rate_limit_store: Dict[str, List[datetime]] = {}
RATE_LIMIT_WINDOW = 60  # seconds
DEFAULT_RATE_LIMIT = 100  # requests per window


async def check_rate_limit(request: Request, user: Optional[User] = None):
    """Check and enforce rate limits"""
    # Get client identifier
    if user:
        client_id = user.id
    else:
        client_id = request.client.host if request.client else "unknown"
    
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    
    # Clean old entries and get recent requests
    if client_id in _rate_limit_store:
        _rate_limit_store[client_id] = [
            t for t in _rate_limit_store[client_id] 
            if t > window_start
        ]
    else:
        _rate_limit_store[client_id] = []
    
    # Check limit
    if len(_rate_limit_store[client_id]) >= DEFAULT_RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Record this request
    _rate_limit_store[client_id].append(now)


# Security headers middleware
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    return response


# HIPAA audit logging
class AuditLogger:
    """HIPAA-compliant audit logging"""
    
    def __init__(self):
        self.audit_log: List[Dict] = []
        
    def log_access(
        self,
        user: Optional[User],
        action: str,
        resource: str,
        patient_id: Optional[str] = None,
        details: Dict = None,
        success: bool = True
    ):
        """Log an access event for HIPAA compliance"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user.id if user else "anonymous",
            "username": user.username if user else "anonymous",
            "action": action,
            "resource": resource,
            "patient_id": patient_id,
            "details": details or {},
            "success": success,
            "ip_address": None  # Would be filled from request context
        }
        
        self.audit_log.append(entry)
        logger.info(f"AUDIT: {action} on {resource} by {entry['username']} - {'SUCCESS' if success else 'FAILED'}")
        
        # In production, write to secure audit database
        return entry
    
    def get_audit_log(self, filters: Dict = None) -> List[Dict]:
        """Get audit log entries with optional filtering"""
        if not filters:
            return self.audit_log
        
        filtered = self.audit_log
        
        if "user_id" in filters:
            filtered = [e for e in filtered if e["user_id"] == filters["user_id"]]
        if "action" in filters:
            filtered = [e for e in filtered if e["action"] == filters["action"]]
        if "start_date" in filters:
            filtered = [e for e in filtered if e["timestamp"] >= filters["start_date"]]
        if "end_date" in filters:
            filtered = [e for e in filtered if e["timestamp"] <= filters["end_date"]]
            
        return filtered


# Singleton audit logger
audit_logger = AuditLogger()


# Token blacklist for logout/revocation
_token_blacklist: set = set()


def blacklist_token(token: str):
    """Add a token to the blacklist"""
    _token_blacklist.add(token)


def is_token_blacklisted(token: str) -> bool:
    """Check if a token is blacklisted"""
    return token in _token_blacklist
