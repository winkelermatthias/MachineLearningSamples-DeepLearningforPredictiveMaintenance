import time
from relspec_service import auth

def test_normalize():
    assert auth.normalize('  a  b   c ') == 'a b c'

def test_strength():
    import pytest
    with pytest.raises(auth.AuthError):
        auth.validate_strength('short one')
    auth.validate_strength('four little words here')
    auth.validate_strength('x'*20)

def test_wsid_deterministic():
    a = auth.workspace_id('four little words here')
    b = auth.workspace_id('four little words here')
    c = auth.workspace_id('four little words there')
    assert a == b and a != c and len(a) == 16

def test_verifier_roundtrip():
    v = auth.make_verifier('a strong enough test phrase')
    assert auth.check_verifier('a strong enough test phrase', v)
    assert not auth.check_verifier('a wrong phrase entirely no', v)

def test_token_roundtrip_and_expiry():
    t = auth.mint_token('ws123')
    assert auth.check_token(t) == 'ws123'
    old = auth.mint_token('ws123', now=time.time()-auth.config.TOKEN_TTL_S-10)
    assert auth.check_token(old) is None
    assert auth.check_token(t[:-4]+'AAAA') is None
