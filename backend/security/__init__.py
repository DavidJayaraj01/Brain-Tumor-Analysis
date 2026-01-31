"""
Security Package
Provides authentication, authorization, and HIPAA compliance features
"""

from .auth import (
    # Models
    User,
    TokenPayload,
    
    # Exceptions
    AuthenticationError,
    AuthorizationError,
    
    # Functions
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    authenticate_user,
    validate_api_key,
    
    # Dependencies
    get_current_user,
    require_auth,
    require_permission,
    require_role,
    check_rate_limit,
    
    # Middleware
    add_security_headers,
    
    # Audit
    AuditLogger,
    audit_logger,
    
    # Token management
    blacklist_token,
    is_token_blacklisted,
    
    # Constants
    SECRET_KEY,
    ALGORITHM
)

__all__ = [
    "User",
    "TokenPayload",
    "AuthenticationError",
    "AuthorizationError",
    "hash_password",
    "verify_password", 
    "create_access_token",
    "create_refresh_token",
    "decode_token",
    "authenticate_user",
    "validate_api_key",
    "get_current_user",
    "require_auth",
    "require_permission",
    "require_role",
    "check_rate_limit",
    "add_security_headers",
    "AuditLogger",
    "audit_logger",
    "blacklist_token",
    "is_token_blacklisted",
    "SECRET_KEY",
    "ALGORITHM"
]
